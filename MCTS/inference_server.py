import logging
import queue
import time
import uuid
import os
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

# Set to True to enable inference server logging (startup, throughput, exceptions)
INFERENCE_SERVER_LOGGING_ENABLED = False


def _create_network_from_cfg(cfg):
    if cfg.env.type == "chess":
        from MCTS.training_modules.chess import create_chess_network as create_network
    elif cfg.env.type == "gomoku":
        from MCTS.training_modules.gomoku import create_gomoku_network as create_network
    else:
        raise ValueError(f"Unsupported environment type: {cfg.env.type}")
    return create_network


def inference_server_worker(
    checkpoint_path: str,
    cfg,
    device_str: str,
    request_queue,
    stop_event,
    max_batch_size: int,
    min_batch_size: int,
    max_wait_ms: int,
    reply_queues_by_worker: dict,
    initial_state_dict: Optional[dict] = None,
):
    # Ensure logging works in spawned subprocess
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    if isinstance(cfg, dict) and not OmegaConf.is_config(cfg):
        cfg = OmegaConf.create(cfg)

    create_network = _create_network_from_cfg(cfg)
    device = torch.device(device_str)
    network = create_network(cfg, device)
    if initial_state_dict is not None:
        try:
            network.load_state_dict(initial_state_dict)
        except Exception:
            pass
    network.to(device).eval()

    last_mtime = 0.0
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            last_mtime = os.path.getmtime(checkpoint_path)
        except Exception:
            last_mtime = 0.0

    max_wait_s = max(0.001, max_wait_ms / 1000.0)
    batch_count = 0
    total_inferences = 0
    log_interval = 100
    log_interval_start = time.monotonic() if INFERENCE_SERVER_LOGGING_ENABLED else 0.0

    if INFERENCE_SERVER_LOGGING_ENABLED:
        logger.info(f"InferenceServer: max_wait_ms={max_wait_ms} (max_wait_s={max_wait_s})")

    while not stop_event.is_set():
        # Hot-reload weights if checkpoint updated
        try:
            if checkpoint_path and os.path.exists(checkpoint_path):
                mtime = os.path.getmtime(checkpoint_path)
                if mtime > last_mtime:
                    try:
                        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
                        network.load_state_dict(ckpt["model_state_dict"])
                        network.to(device).eval()
                        last_mtime = mtime
                    except Exception:
                        pass
        except Exception:
            pass

        # Collect a batch of requests
        pending = []
        try:
            req = request_queue.get(timeout=max_wait_s)
            pending.append(req)
        except queue.Empty:
            continue
        except Exception:
            continue

        # Wait up to max_wait_s for more requests. Use explicit polling + sleep so we
        # actually spend the wait time (multiprocessing.Queue.get(timeout) can return
        # early on some platforms). This allows requests from multiple workers to accumulate.
        # When we have fewer than min_batch_size, extend deadline once to avoid small batches.
        deadline = time.monotonic() + max_wait_s
        deadline_extended = False
        while len(pending) < max_batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if len(pending) < min_batch_size and not deadline_extended:
                    deadline = time.monotonic() + max_wait_s
                    deadline_extended = True
                else:
                    break
            try:
                req = request_queue.get(timeout=min(0.05, remaining))
                pending.append(req)
            except queue.Empty:
                time.sleep(max(0, min(0.05, remaining)))
            except Exception as e:
                if INFERENCE_SERVER_LOGGING_ENABLED:
                    logger.warning(f"InferenceServer: collection loop exception: {e}")
                break

        obs_list = []
        sizes = []
        for req in pending:
            obs = req.get("obs")
            if obs is None:
                sizes.append(0)
                obs_list.append(None)
                continue
            if obs.ndim == 3:
                obs = obs[None, ...]
            sizes.append(obs.shape[0])
            obs_list.append(obs)

        # Filter out invalid requests
        valid_pairs = [(req, obs, size) for req, obs, size in zip(pending, obs_list, sizes) if size > 0]
        if not valid_pairs:
            continue

        obs_batch = np.concatenate([obs for _, obs, _ in valid_pairs], axis=0)
        obs_tensor = torch.from_numpy(obs_batch).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            policy_logits, value_preds = network(obs_tensor)
        policy_np = policy_logits.detach().cpu().numpy()
        value_np = value_preds.detach().cpu().numpy()

        offset = 0
        for req, _obs, size in valid_pairs:
            worker_id = req.get("worker_id")
            request_id = req.get("id")
            if size <= 0:
                offset += size
                continue
            reply_queue = reply_queues_by_worker.get(worker_id) if worker_id is not None else None
            if reply_queue is None:
                offset += size
                continue
            pol_slice = policy_np[offset:offset + size]
            val_slice = value_np[offset:offset + size]
            try:
                reply_queue.put((request_id, pol_slice, val_slice))
            except Exception:
                pass
            offset += size

        if INFERENCE_SERVER_LOGGING_ENABLED:
            batch_count += 1
            num_requests = len(valid_pairs)
            combined_obs = sum(s for _, _, s in valid_pairs)
            total_inferences += combined_obs
            avg_obs = total_inferences / batch_count
            worker_ids = [req.get("worker_id") for req, _, _ in valid_pairs if req.get("worker_id") is not None]
            unique_workers = len(set(worker_ids)) if worker_ids else "?"
            if batch_count % log_interval == 0:
                elapsed_s = time.monotonic() - log_interval_start
                batches_per_s = log_interval / elapsed_s if elapsed_s > 0 else 0
                log_interval_start = time.monotonic()
                logger.info(
                    f"InferenceServer: obs={combined_obs} (from {num_requests} requests, {unique_workers} workers), "
                    f"avg_obs={avg_obs:.1f} | last {log_interval} batches in {elapsed_s:.2f}s (~{batches_per_s:.1f} batch/s)"
                )

        if device.type == "cuda":
            try:
                torch.cuda.synchronize(device)
            except Exception:
                pass


class InferenceClient:
    def __init__(self, request_queue, reply_queue, timeout_s: float = 30.0, worker_id: Optional[int] = None):
        self.request_queue = request_queue
        self.reply_queue = reply_queue
        self.timeout_s = timeout_s
        self.worker_id = worker_id

    def predict(self, obs_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        obs_np = obs_batch.detach().cpu().numpy()
        request_id = uuid.uuid4().hex
        req = {
            "id": request_id,
            "obs": obs_np,
        }
        if self.worker_id is not None:
            req["worker_id"] = self.worker_id
        self.request_queue.put(req)
        while True:
            resp_id, pol_np, val_np = self.reply_queue.get(timeout=self.timeout_s)
            if resp_id == request_id:
                break
        policy_logits = torch.from_numpy(pol_np)
        value_preds = torch.from_numpy(val_np)
        return policy_logits, value_preds

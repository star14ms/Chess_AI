import queue
import time
import uuid
import os
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf


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
    max_wait_ms: int,
    initial_state_dict: Optional[dict] = None,
):
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

        start_time = time.time()
        while len(pending) < max_batch_size:
            remaining = max_wait_s - (time.time() - start_time)
            if remaining <= 0:
                break
            try:
                req = request_queue.get(timeout=remaining)
                pending.append(req)
            except queue.Empty:
                break
            except Exception:
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
            reply_queue = req.get("reply_queue")
            request_id = req.get("id")
            if reply_queue is None or size <= 0:
                offset += size
                continue
            pol_slice = policy_np[offset:offset + size]
            val_slice = value_np[offset:offset + size]
            try:
                reply_queue.put((request_id, pol_slice, val_slice))
            except Exception:
                pass
            offset += size

        if device.type == "cuda":
            try:
                torch.cuda.synchronize(device)
            except Exception:
                pass


class InferenceClient:
    def __init__(self, request_queue, reply_queue, timeout_s: float = 30.0):
        self.request_queue = request_queue
        self.reply_queue = reply_queue
        self.timeout_s = timeout_s

    def predict(self, obs_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        obs_np = obs_batch.detach().cpu().numpy()
        request_id = uuid.uuid4().hex
        self.request_queue.put({
            "id": request_id,
            "obs": obs_np,
            "reply_queue": self.reply_queue,
        })
        while True:
            resp_id, pol_np, val_np = self.reply_queue.get(timeout=self.timeout_s)
            if resp_id == request_id:
                break
        policy_logits = torch.from_numpy(pol_np)
        value_preds = torch.from_numpy(val_np)
        return policy_logits, value_preds

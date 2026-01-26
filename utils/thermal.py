import re
import shutil
import subprocess
import sys
import time
from typing import Any

from omegaconf import DictConfig, OmegaConf
from rich.progress import Progress


def _parse_temperatures_from_text(text: str) -> list[float]:
    temps: list[float] = []
    if not text:
        return temps
    for match in re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*C", text):
        try:
            temps.append(float(match))
        except Exception:
            pass
    return temps


def _read_macos_temperature_c() -> float | None:
    if sys.platform != "darwin":
        return None
    command_candidates: list[list[str]] = []
    pmset_path = shutil.which("pmset")
    if pmset_path:
        command_candidates.append([pmset_path, "-g", "thermlog"])
    powermetrics_path = shutil.which("powermetrics")
    if powermetrics_path:
        command_candidates.append([powermetrics_path, "--samplers", "smc", "-n", "1"])

    for cmd in command_candidates:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        except Exception:
            continue
        output = (result.stdout or "") + "\n" + (result.stderr or "")
        temps = _parse_temperatures_from_text(output)
        if temps:
            return max(temps)
    return None


def _read_macos_thermal_pressure_level() -> str | None:
    if sys.platform != "darwin":
        return None
    powermetrics_path = shutil.which("powermetrics")
    if not powermetrics_path:
        return None
    try:
        # Use sudo -n to avoid blocking for password.
        result = subprocess.run(
            ["sudo", "-n", powermetrics_path, "--samplers", "thermal", "-n", "1"],
            capture_output=True,
            text=True,
            timeout=6,
        )
    except Exception:
        return None
    output = (result.stdout or "") + "\n" + (result.stderr or "")
    match = re.search(r"Current pressure level:\s*([A-Za-z]+)", output)
    if match:
        return match.group(1).strip()
    return None


def _get_thermal_throttle_settings(cfg: DictConfig) -> dict[str, Any] | None:
    if not hasattr(cfg, "training"):
        return None
    settings: dict[str, Any] = {}
    thermal_cfg = cfg.training.get("thermal_throttle", None)
    if thermal_cfg is not None:
        try:
            if OmegaConf.is_config(thermal_cfg):
                thermal_cfg = OmegaConf.to_container(thermal_cfg, resolve=True)
        except Exception:
            pass
        if isinstance(thermal_cfg, dict):
            settings.update(thermal_cfg)

    # Back-compat direct keys
    for key in ("max_temp_c", "resume_temp_c", "check_interval_seconds"):
        if key not in settings:
            val = cfg.training.get(f"thermal_throttle_{key}", None)
            if val is not None:
                settings[key] = val

    enabled = settings.get("enabled", None)
    if enabled is False:
        return None
    max_temp_c = settings.get("max_temp_c", None)
    if max_temp_c is None:
        return None
    try:
        max_temp_c = float(max_temp_c)
    except Exception:
        return None
    resume_temp_c = settings.get("resume_temp_c", max_temp_c - 5.0)
    try:
        resume_temp_c = float(resume_temp_c)
    except Exception:
        resume_temp_c = max_temp_c - 5.0
    if resume_temp_c >= max_temp_c:
        resume_temp_c = max_temp_c - 1.0
    check_interval = settings.get("check_interval_seconds", 10.0)
    try:
        check_interval = float(check_interval)
    except Exception:
        check_interval = 10.0
    check_interval = max(1.0, check_interval)

    return {
        "max_temp_c": max_temp_c,
        "resume_temp_c": resume_temp_c,
        "check_interval_seconds": check_interval,
        "pressure_pause_level": settings.get("pressure_pause_level", "Heavy"),
        "pressure_resume_level": settings.get("pressure_resume_level", "Nominal"),
    }


def maybe_pause_for_thermal_throttle(
    cfg: DictConfig,
    progress: Progress | None = None,
    phase: str = "self-play",
) -> None:
    settings = _get_thermal_throttle_settings(cfg)
    if settings is None:
        return
    current_temp = _read_macos_temperature_c()
    if current_temp is None:
        pressure_level = _read_macos_thermal_pressure_level()
        if pressure_level is None:
            return
        pressure_rank = {"Nominal": 0, "Moderate": 1, "Heavy": 2, "Critical": 3}
        pause_rank = pressure_rank.get(settings["pressure_pause_level"], 2)
        current_rank = pressure_rank.get(pressure_level, 0)
        if current_rank < pause_rank:
            return
        resume_rank = pressure_rank.get(settings["pressure_resume_level"], 0)
        if progress is not None:
            progress.print(
                f"Thermal throttle: pressure {pressure_level} >= {settings['pressure_pause_level']}. "
                f"Pausing {phase} until <= {settings['pressure_resume_level']}."
            )
        while True:
            time.sleep(settings["check_interval_seconds"])
            pressure_level = _read_macos_thermal_pressure_level()
            if pressure_level is None:
                break
            current_rank = pressure_rank.get(pressure_level, 0)
            if current_rank <= resume_rank:
                if progress is not None:
                    progress.print(
                        f"Thermal throttle cleared: pressure {pressure_level} <= {settings['pressure_resume_level']}. "
                        f"Resuming {phase}."
                    )
                break
        return

    if current_temp < settings["max_temp_c"]:
        return

    if progress is not None:
        progress.print(
            f"Thermal throttle: {current_temp:.1f}C >= {settings['max_temp_c']:.1f}C. "
            f"Pausing {phase} until <= {settings['resume_temp_c']:.1f}C."
        )
    while True:
        time.sleep(settings["check_interval_seconds"])
        current_temp = _read_macos_temperature_c()
        if current_temp is None or current_temp <= settings["resume_temp_c"]:
            if progress is not None and current_temp is not None:
                progress.print(
                    f"Thermal throttle cleared: {current_temp:.1f}C <= {settings['resume_temp_c']:.1f}C. "
                    f"Resuming {phase}."
                )
            break

from pathlib import Path
import os

from .config import get_settings


ENV_PATH = Path(".env")


def save_deepseek_api_key(api_key: str) -> None:
    if not api_key or not api_key.strip():
        return
    updates = {
        "DEEPSEEK_API_KEY": api_key.strip(),
        "LLM_PROVIDER": "deepseek",
    }
    lines = []
    if ENV_PATH.exists():
        lines = ENV_PATH.read_text(encoding="utf-8").splitlines()

    seen: set[str] = set()
    next_lines = []
    for line in lines:
        key = line.split("=", 1)[0].strip() if "=" in line and not line.strip().startswith("#") else None
        if key in updates:
            next_lines.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            next_lines.append(line)

    for key, value in updates.items():
        if key not in seen:
            next_lines.append(f"{key}={value}")

    ENV_PATH.write_text("\n".join(next_lines).rstrip() + "\n", encoding="utf-8")
    os.environ.update(updates)
    get_settings.cache_clear()


def has_deepseek_api_key() -> bool:
    if not ENV_PATH.exists():
        return False
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("DEEPSEEK_API_KEY="):
            value = line.split("=", 1)[1].strip()
            return bool(value and value != "replace-if-needed")
    return False


def delete_deepseek_api_key() -> None:
    if not ENV_PATH.exists():
        return
    next_lines = []
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        key = line.split("=", 1)[0].strip() if "=" in line and not line.strip().startswith("#") else None
        if key == "DEEPSEEK_API_KEY":
            continue
        next_lines.append(line)
    ENV_PATH.write_text("\n".join(next_lines).rstrip() + "\n", encoding="utf-8")
    os.environ.pop("DEEPSEEK_API_KEY", None)
    get_settings.cache_clear()

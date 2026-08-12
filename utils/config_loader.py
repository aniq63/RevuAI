import yaml
from pathlib import Path

# Locate the config file path relative to this script
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"

# Prefer common names, but allow either .yaml or .yml
_candidates = [CONFIG_DIR / "settings.yaml", CONFIG_DIR / "settings.yml"]
CONFIG_PATH = None
for _p in _candidates:
    if _p.exists():
        CONFIG_PATH = _p
        break

# Fallback: find any file in the config dir that looks like settings.*yml
if CONFIG_PATH is None and CONFIG_DIR.exists():
    for _p in CONFIG_DIR.iterdir():
        if _p.is_file() and _p.stem.startswith("settings") and _p.suffix in (".yml", ".yaml"):
            CONFIG_PATH = _p
            break

if CONFIG_PATH is None:
    tried = ", ".join(str(p) for p in _candidates)
    available = list(CONFIG_DIR.iterdir()) if CONFIG_DIR.exists() else "(config dir missing)"
    raise FileNotFoundError(f"No settings YAML found. Tried: {tried}. Available: {available}")

# Load the YAML data into a standard Python dictionary
with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    settings = yaml.safe_load(file)

# Normalize file paths in settings to be absolute where appropriate
try:
    if isinstance(settings, dict):
        data_cfg = settings.get("data")
        if isinstance(data_cfg, dict) and "path" in data_cfg:
            raw_path = data_cfg["path"]
            if isinstance(raw_path, str):
                p = Path(raw_path)
                if not p.is_absolute():
                    # Interpret relative paths as relative to project BASE_DIR
                    p = (BASE_DIR / raw_path).resolve()
                settings["data"]["path"] = str(p)
except Exception:
    # If normalization fails, fall back to original settings and allow caller to handle errors
    pass

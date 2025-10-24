import joblib
import yaml
from pathlib import Path

CONFIG_PATH = Path("configs/config.yaml")

def load_config():
    """Загружает YAML-конфиг"""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg

def load_model():
    """Загружает модель из пути, указанного в YAML-конфиге"""
    cfg = load_config()
    model_path = cfg["output"]["model_path"]
    model = joblib.load(model_path)
    return model, cfg

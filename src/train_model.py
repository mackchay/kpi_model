import pandas as pd
import joblib
import yaml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from pathlib import Path

CONFIG_PATH = Path("configs/config.yaml")

def load_config():
    """Загружает конфиг YAML"""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg

def train_model():
    cfg = load_config()

    # Загружаем данные
    df = pd.read_csv(cfg["data"]["path"])

    # Выбираем признаки
    if "features" in cfg["data"]:
        features = cfg["data"]["features"]
    else:
        features = [col for col in df.columns if col != cfg["data"]["target"]]

    X = df[features]
    y = df[cfg["data"]["target"]]

    # Делим на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["data"].get("test_size", 0.2),
        random_state=cfg["model"].get("random_state", 42)
    )

    # Создаём модель
    model = RandomForestRegressor(
        n_estimators=cfg["model"].get("n_estimators", 100),
        max_depth=cfg["model"].get("max_depth", None),
        random_state=cfg["model"].get("random_state", 42),
        n_jobs=-1
    )

    # Обучаем
    model.fit(X_train, y_train)

    # Предсказания на тесте
    y_pred = model.predict(X_test)
    print(f"R²: {r2_score(y_test, y_pred):.3f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")

    # Сохраняем модель
    model_path = cfg["output"]["model_path"]
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"✅ Модель обучена и сохранена в {model_path}")

if __name__ == "__main__":
    train_model()

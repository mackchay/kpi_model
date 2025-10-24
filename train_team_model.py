import pandas as pd
import joblib
import yaml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from pathlib import Path
import numpy as np

CONFIG_PATH = Path("configs/config.yaml")


def load_config():
    """Загружает конфиг YAML"""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def train_team_model():
    """
    Обучает модель для расчета командного KPI на основе улучшенных метрик.
    """
    cfg = load_config()

    # Загружаем данные командного KPI
    df = pd.read_csv("data/team_kpi_dataset.csv")

    # Проверяем наличие всех необходимых признаков
    required_features = [
        'commits_total', 'merge_conflicts', 'bus_factor', 'team_size', 'active_days',
        'refactor_ratio', 'fix_ratio', 'feature_ratio', 'docs_ratio', 'test_ratio', 'avg_commits_per_dev'
    ]

    # Проверяем и создаем отсутствующие признаки
    for feature in required_features:
        if feature not in df.columns:
            print(f"⚠️  Признак {feature} отсутствует в датасете, создаем...")
            if feature == 'avg_commits_per_dev':
                df['avg_commits_per_dev'] = df['commits_total'] / df['team_size']
            else:
                df[feature] = 0  # или другое значение по умолчанию

    # Выбираем признаки для командного KPI
    team_features = required_features

    X = df[team_features]
    y = df['kpi_score']

    # Проверяем данные на наличие NaN и бесконечных значений
    print("🔍 Проверка данных перед обучением:")
    print(f"Размерность X: {X.shape}")
    print(f"Количество NaN в X: {X.isna().sum().sum()}")
    print(f"Количество NaN в y: {y.isna().sum()}")
    print(f"Диапазон KPI: {y.min():.2f} - {y.max():.2f}")

    # Очистка данных
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    y = y.fillna(0)

    # Делим на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["data"].get("test_size", 0.2),
        random_state=cfg["model"].get("random_state", 42),
        stratify=pd.cut(y, bins=5, labels=range(5))  # Стратификация по KPI
    )

    # Создаём модель для командного KPI
    team_model = RandomForestRegressor(
        n_estimators=cfg["model"].get("n_estimators", 200),
        max_depth=cfg["model"].get("max_depth", 12),
        min_samples_split=cfg["model"].get("min_samples_split", 5),
        min_samples_leaf=cfg["model"].get("min_samples_leaf", 2),
        random_state=cfg["model"].get("random_state", 42),
        n_jobs=-1
    )

    # Обучаем
    print("🎯 Начинаем обучение командной модели...")
    team_model.fit(X_train, y_train)

    # Предсказания на тесте
    y_pred = team_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"\n📊 Team Model Performance:")
    print(f"R²: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {np.sqrt(np.mean((y_test - y_pred) ** 2)):.4f}")

    # Сохраняем модель
    team_model_path = "models/team_kpi_regressor.pkl"
    Path(team_model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(team_model, team_model_path)
    print(f"✅ Team модель обучена и сохранена в {team_model_path}")

    # Метрики на train и test
    train_score = team_model.score(X_train, y_train)
    test_score = team_model.score(X_test, y_test)
    print(f"R² train: {train_score:.4f}")
    print(f"R² test: {test_score:.4f}")
    print(f"Разница: {train_score - test_score:.4f}")

    # Анализ важности признаков
    feature_importance = pd.DataFrame({
        'feature': team_features,
        'importance': team_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\n🔝 Важность признаков:")
    for _, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    # Сохраняем информацию о признаках
    feature_info = {
        "model_type": "team_kpi",
        "features": team_features,
        "target": "kpi_score",
        "performance": {
            "r2_score": float(r2),
            "mae": float(mae),
            "train_r2": float(train_score),
            "test_r2": float(test_score)
        },
        "feature_importance": feature_importance.to_dict('records')
    }

    import json
    with open("models/team_model_info.json", "w") as f:
        json.dump(feature_info, f, indent=2)

    return team_model, team_features


if __name__ == "__main__":
    train_team_model()
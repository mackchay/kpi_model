from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

from src.model_loader import load_model
from src.utils import calculate_kpi

app = FastAPI(title="Team KPI Predictor")

# Загружаем модель и конфиг один раз при старте сервера
model, cfg = load_model()

# Берём список признаков из данных (все колонки кроме target)
FEATURES = [col for col in pd.read_csv(cfg["data"]["path"]).columns if col != cfg["data"]["target"]]

class GitMetrics(BaseModel):
    commits_total: int
    merge_conflicts: int
    code_review_ratio: float
    bus_factor: int
    avg_commit_size: float
    comment_density: float
    churn_rate: float
    workload_balance: float
    refactor_commits: int
    active_days: int
    team_size: int


@app.post("/predict_kpi")
def predict_kpi(metrics: GitMetrics):
    data = pd.DataFrame([metrics.model_dump()])

    # Аналитический KPI (по формуле)
    analytical_kpi = calculate_kpi(data).iloc[0]

    # ML KPI (по модели)
    ml_kpi = float(model.predict(data)[0])

    return {
        "analytical_kpi": round(analytical_kpi, 2),
        "ml_predicted_kpi": round(ml_kpi, 2),
        "difference": round(ml_kpi - analytical_kpi, 2)
    }


@app.get("/feature_importance")
def get_feature_importance():
    """
    Возвращает важность признаков модели (JSON).
    """
    if not hasattr(model, "feature_importances_"):
        return {"error": "Модель не поддерживает feature_importances_"}

    importance = model.feature_importances_
    importance_dict = {feature: round(float(score), 4) for feature, score in zip(FEATURES, importance)}
    sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    return {"feature_importance": sorted_importance}

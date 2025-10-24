from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd

from src.model_loader import load_model
from src.utils import calculate_kpi
from src.preprocess import extract_features_from_project

app = FastAPI(
    title="Team KPI Predictor",
    description="API для расчета KPI команд разработчиков на основе данных GitHub",
    version="1.0.0"
)

# Загружаем модель и конфиг один раз при старте сервера
model, cfg = load_model()

# Берём список признаков из данных (все колонки кроме target)
FEATURES = [col for col in pd.read_csv(cfg["data"]["path"]).columns if col != cfg["data"]["target"]]

# Pydantic models for GitHub data structure
class Author(BaseModel):
    name: str
    email: Optional[str] = None

class Committer(BaseModel):
    name: str
    email: Optional[str] = None

class Commit(BaseModel):
    hash: Optional[str] = None
    message: str
    author: Author
    committer: Optional[Committer] = None
    created_at: str
    parents: List[str] = []
    branch_names: Optional[List[str]] = None

class Repo(BaseModel):
    name: Optional[str] = None
    commits: List[Commit] = []

class Project(BaseModel):
    name: Optional[str] = None
    repos: List[Repo] = []

class GitHubData(BaseModel):
    project: Project

@app.get("/")
def root():
    """
    Корневой endpoint с информацией об API
    """
    return {
        "message": "Team KPI Predictor API",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict_kpi": "Основной endpoint для расчета KPI из GitHub данных",
            "POST /predict_kpi_metrics": "Альтернативный endpoint для прямого ввода метрик",
            "GET /feature_importance": "Получить важность признаков модели",
            "GET /docs": "Swagger документация API"
        },
        "data_format": {
            "description": "Данные должны быть в формате project -> repos -> commits с полной структурой GitHub",
            "example": {
                "project": {
                    "name": "team26",
                    "repos": [
                        {
                            "name": "repo1",
                            "commits": [
                                {
                                    "hash": "abc123def456",
                                    "message": "feat: add new feature",
                                    "author": {"name": "Developer Name", "email": "dev@example.com"},
                                    "committer": {"name": "GitHub", "email": "noreply@github.com"},
                                    "created_at": "2025-08-12T12:35:28Z",
                                    "parents": ["751a32d..."],
                                    "branch_names": ["main"]
                                }
                            ]
                        }
                    ]
                }
            }
        }
    }

# Keep GitMetrics for backward compatibility
class GitMetrics(BaseModel):
    commits_total: int
    merge_conflicts: int
    bus_factor: int
    refactor_commits: int
    fix_commits: int
    feature_commits: int
    docs_commits: int
    test_commits: int
    refactor_ratio: float
    fix_ratio: float
    feature_ratio: float
    docs_ratio: float
    test_ratio: float
    active_days: int
    team_size: int


@app.post("/predict_kpi")
def predict_kpi(github_data: GitHubData):
    """
    Принимает данные GitHub в формате project -> repos -> commits
    и возвращает KPI на основе аналитических и ML методов.
    
    Входные данные:
    - project: объект проекта с репозиториями
    - repos: список репозиториев с коммитами
    - commits: список коммитов с сообщениями, авторами и датами
    
    Возвращает:
    - analytical_kpi: KPI рассчитанный по аналитической формуле
    - ml_predicted_kpi: KPI предсказанный ML моделью
    - difference: разность между ML и аналитическим KPI
    - extracted_features: извлеченные признаки из данных
    """
    try:
        # Преобразуем Pydantic модель в словарь для preprocess.py
        data_dict = github_data.model_dump()
        
        # Извлекаем признаки используя preprocess.py
        features_df = extract_features_from_project(data_dict)
        
        # Аналитический KPI (по формуле)
        analytical_kpi = calculate_kpi(features_df).iloc[0]

        # ML KPI (по модели)
        ml_kpi = float(model.predict(features_df)[0])

        return {
            "analytical_kpi": round(analytical_kpi, 2),
            "ml_predicted_kpi": round(ml_kpi, 2),
            "difference": round(ml_kpi - analytical_kpi, 2),
            "extracted_features": features_df.iloc[0].to_dict()
        }
    except Exception as e:
        return {"error": f"Ошибка обработки данных: {str(e)}"}


@app.post("/predict_kpi_metrics")
def predict_kpi_metrics(metrics: GitMetrics):
    """
    Альтернативный endpoint для прямого ввода метрик (обратная совместимость).
    Принимает уже извлеченные признаки в формате GitMetrics.
    """
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

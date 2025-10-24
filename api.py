from dataclasses import field
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict
import pandas as pd
import joblib


from model_loader import load_model, load_config
from utils import calculate_individual_kpi, calculate_team_kpi, filter_features, calculate_team_and_individual_kpis
from preprocess import extract_features_from_project, extract_individual_metrics, parse_date, \
    calculate_commit_complexity, improved_classify_commit

app = FastAPI(
    title="Team KPI Predictor",
    description="API для расчета KPI команд разработчиков на основе данных GitHub",
    version="1.0.0"
)

# === Загрузка моделей ===
cfg = load_config()

def try_load_model(path: str):
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        print(f"[WARN] Не удалось загрузить модель: {path} → {e}")
        return None

# Основные модели
team_model = try_load_model(cfg["output"].get("team_model_path", ""))
individual_model = try_load_model(cfg["output"].get("individual_model_path", ""))

# Legacy модель (для старой совместимости)
try:
    legacy_model, legacy_cfg = load_model()
    LEGACY_FEATURES = [
        col for col in pd.read_csv(legacy_cfg["data"]["path"]).columns
        if col != legacy_cfg["data"]["target"]
    ]
except Exception:
    legacy_model = None
    LEGACY_FEATURES = []

# Фичи для ML моделей (если есть)
TEAM_FEATURES = getattr(team_model, "feature_names_in_", []) if team_model else []
INDIVIDUAL_FEATURES = getattr(individual_model, "feature_names_in_", []) if individual_model else []

# === Pydantic модели для входных данных ===
class Author(BaseModel):
    name: str
    email: str

class Committer(BaseModel):
    name: str
    email: str

class Commit(BaseModel):
    hash: Optional[str] = None
    message: str
    author: Author
    committer: Optional[Committer] = None
    createdAt: str
    parents: List[str] = field(default_factory=list)
    branches: Optional[List[str]] = None
    branch_names: Optional[List[str]] = None

class Repo(BaseModel):
    name: Optional[str] = None
    commits: List[Commit] = field(default_factory=list)

class Project(BaseModel):
    key: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None

class Repository(BaseModel):
    name: Optional[str] = None
    createdAt: Optional[str] = None

class BackendResponse(BaseModel):
    success: bool
    message: str
    project: Project
    repository: Repository
    commits: List[Commit] = field(default_factory=list)

class GitHubData(BaseModel):
    project: Project

# === Корневой endpoint ===
@app.get("/")
def root():
    return {
        "message": "Team KPI Predictor API",
        "version": "1.0.0",
        "models": {
            "team_model_loaded": team_model is not None,
            "individual_model_loaded": individual_model is not None,
            "legacy_model_loaded": legacy_model is not None
        }
    }

# === Совместимость с GitMetrics ===
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
    avg_commits_per_dev: float

@app.post("/predict_team_kpi")
def predict_team_kpi(backend_data: BackendResponse):
    """Рассчитывает командный KPI"""
    try:
        data_dict = backend_data.model_dump()
        team_features = extract_features_from_project(data_dict)
        team_data = team_features.iloc[0].to_dict()

        analytical_team_kpi = calculate_team_kpi(team_data)

        if team_model is not None:
            # Используем filter_features для правильного выравнивания признаков
            team_df = filter_features(team_features, team_model)
            ml_team_kpi = float(team_model.predict(team_df)[0])
        else:
            ml_team_kpi = analytical_team_kpi

        return {
            "team_kpi": {
                "analytical": round(analytical_team_kpi, 2),
                "ml_predicted": round(ml_team_kpi, 2),
                "difference": round(ml_team_kpi - analytical_team_kpi, 2)
            },
            "team_metrics": team_data,
            "formula_used": "Productivity + Code Quality + Technical Health"
        }
    except Exception as e:
        return {"error": f"Ошибка расчета командного KPI: {e}"}

def calculate_developer_metrics(commits: List[Dict]) -> Dict:
    """Вычисляет метрики для индивидуального разработчика с использованием улучшенной классификации"""
    total = len(commits)
    if total == 0:
        return {
            "feature_ratio": 0.0, "fix_ratio": 0.0, "refactor_ratio": 0.0,
            "test_ratio": 0.0, "docs_ratio": 0.0, "total_commits": 0,
            "active_days": 0, "feature_commits": 0, "fix_commits": 0,
            "refactor_commits": 0, "test_commits": 0, "docs_commits": 0,
            "avg_complexity": 1.0  # Добавляем сложность по умолчанию
        }

    # Используем улучшенную классификацию из preprocess.py
    commit_types = {
        "feature": 0, "fix": 0, "refactor": 0,
        "test": 0, "docs": 0, "other": 0
    }
    total_complexity = 0.0
    active_days = set()

    for commit in commits:
        # Классифицируем коммит с улучшенной функцией
        msg = commit.get("message", "")
        commit_type = improved_classify_commit(msg)
        commit_types[commit_type] += 1

        # Рассчитываем сложность коммита
        complexity = calculate_commit_complexity(commit)
        total_complexity += complexity

        # Считаем активные дни
        created_at = commit.get("createdAt") or commit.get("created_at")
        if created_at:
            dt = parse_date(created_at)
            if dt:
                active_days.add(dt.date())

    # Рассчитываем соотношения
    feature_ratio = commit_types["feature"] / total
    fix_ratio = commit_types["fix"] / total
    refactor_ratio = commit_types["refactor"] / total
    test_ratio = commit_types["test"] / total
    docs_ratio = commit_types["docs"] / total
    avg_complexity = total_complexity / total if total > 0 else 1.0

    return {
        "feature_ratio": feature_ratio,
        "fix_ratio": fix_ratio,
        "refactor_ratio": refactor_ratio,
        "test_ratio": test_ratio,
        "docs_ratio": docs_ratio,
        "total_commits": total,
        "active_days": len(active_days),
        "feature_commits": commit_types["feature"],
        "fix_commits": commit_types["fix"],
        "refactor_commits": commit_types["refactor"],
        "test_commits": commit_types["test"],
        "docs_commits": commit_types["docs"],
        "avg_complexity": avg_complexity  # Добавляем среднюю сложность
    }

@app.post("/predict_individual_kpi")
def predict_individual_kpi(backend_data: BackendResponse):
    """Рассчитывает индивидуальные KPI разработчиков с объединением KPI и метрик"""
    try:
        data_dict = backend_data.model_dump()
        individual_metrics = extract_individual_metrics(data_dict)

        developers = {}

        for author, commits in individual_metrics.items():
            # Вычисляем метрики разработчика
            metrics = calculate_developer_metrics(commits)

            # Аналитический KPI
            analytical_kpi = calculate_individual_kpi(commits)

            # ML KPI
            if individual_model is not None and metrics['total_commits'] > 0:
                individual_df = pd.DataFrame([metrics])
                individual_df = filter_features(individual_df, individual_model)
                ml_kpi = float(individual_model.predict(individual_df)[0])
            else:
                ml_kpi = analytical_kpi

            # Округляем метрики для вывода
            rounded_metrics = {k: round(v, 2) if isinstance(v, float) else v
                               for k, v in metrics.items()}

            # Объединяем все в один объект
            developers[author] = {
                "analytical_kpi": round(analytical_kpi, 2),
                "ml_kpi": round(ml_kpi, 2),
                "metrics": rounded_metrics
            }

        return {
            "developers": developers,
            "formula_used": "Activity + Features + Fixes + Quality + Process"
        }

    except Exception as e:
        return {"error": f"Ошибка расчета индивидуальных KPI: {e}"}

@app.post("/predict_comprehensive_kpi")
def predict_comprehensive_kpi(backend_data: BackendResponse):
    """Полный анализ: командный + индивидуальные KPI с объединением KPI и метрик"""
    try:
        data_dict = backend_data.model_dump()

        # 1️⃣ Аналитические расчёты
        comprehensive = calculate_team_and_individual_kpis(data_dict)
        individual_metrics = comprehensive["individual_metrics"]

        # 2️⃣ ML-командный KPI
        team_features = extract_features_from_project(data_dict)
        model_to_use = team_model or legacy_model
        if model_to_use is not None:
            team_df = filter_features(team_features, model_to_use)
            ml_team_kpi = float(model_to_use.predict(team_df)[0])
        else:
            ml_team_kpi = comprehensive["team_kpi"]

        # 3️⃣ Индивидуальные KPI с метриками
        developers = {}
        for author, commits in individual_metrics.items():
            # Вычисляем метрики разработчика
            metrics = calculate_developer_metrics(commits)

            analytical_kpi = comprehensive["individual_kpis"][author]

            # ML KPI
            if individual_model is not None and metrics['total_commits'] > 0:
                individual_df = pd.DataFrame([metrics])
                individual_df = filter_features(individual_df, individual_model)
                ml_kpi = float(individual_model.predict(individual_df)[0])
            else:
                ml_kpi = analytical_kpi

            # Округляем метрики для вывода
            rounded_metrics = {k: round(v, 2) if isinstance(v, float) else v
                               for k, v in metrics.items()}

            developers[author] = {
                "analytical_kpi": round(analytical_kpi, 2),
                "ml_kpi": round(ml_kpi, 2),
                "metrics": rounded_metrics
            }

        return {
            "team_kpi": {
                "analytical": round(comprehensive["team_kpi"], 2),
                "ml_predicted": round(ml_team_kpi, 2),
                "difference": round(ml_team_kpi - comprehensive["team_kpi"], 2)
            },
            "developers": developers,
            "team_metrics": comprehensive["team_metrics"],
            "team_size": comprehensive["team_size"],
            "formulas": {
                "team": "Productivity + Code Quality + Technical Health",
                "individual": "Activity + Features + Fixes + Quality + Process"
            }
        }

    except Exception as e:
        return {"error": f"Ошибка полного анализа KPI: {e}"}

@app.get("/feature_importance")
def get_feature_importance():
    """Возвращает важность признаков из доступной модели"""
    model_to_check = team_model or individual_model or legacy_model
    if not model_to_check or not hasattr(model_to_check, "feature_importances_"):
        return {"error": "Нет доступных моделей с поддержкой feature_importances_"}

    if model_to_check == team_model:
        features = TEAM_FEATURES
    elif model_to_check == individual_model:
        features = INDIVIDUAL_FEATURES
    else:
        features = LEGACY_FEATURES

    importance = model_to_check.feature_importances_
    importance_dict = {f: round(float(v), 4) for f, v in zip(features, importance)}
    sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    return {"feature_importance": sorted_importance}

@app.get("/model_info")
def get_model_info():
    """Возвращает информацию о загруженных моделях"""

    def get_model_features(model, model_name):
        if model is None:
            return f"{model_name}: не загружена"

        features = getattr(model, "feature_names_in_", [])
        return f"{model_name}: {len(features)} признаков"

    return {
        "team_model": get_model_features(team_model, "Командная модель"),
        "individual_model": get_model_features(individual_model, "Индивидуальная модель"),
        "legacy_model": get_model_features(legacy_model, "Legacy модель")
    }
import pandas as pd
from src.preprocess import extract_features_from_project, extract_individual_metrics


def filter_features(df: pd.DataFrame, model) -> pd.DataFrame:
    """
    Приводит DataFrame к тому же набору признаков, что и у модели.
    """
    if not hasattr(model, "feature_names_in_"):
        return df

    model_features = list(model.feature_names_in_)
    df_aligned = pd.DataFrame(columns=model_features)

    for col in model_features:
        if col in df.columns:
            df_aligned[col] = df[col]
        else:
            df_aligned[col] = 0

    return df_aligned[model_features]


def calculate_individual_kpi(developer_commits: list) -> float:
    """Улучшенный расчет индивидуального KPI"""
    if not developer_commits:
        return 0.0

    total_commits = len(developer_commits)

    # Улучшенная классификация
    def classify_commit(message):
        if not message:
            return "other"

        msg_lower = message.lower()
        if any(word in msg_lower for word in ["feat", "feature", "add", "implement", "new"]):
            return "feature"
        elif any(word in msg_lower for word in ["fix", "bug", "error", "issue", "resolve"]):
            return "fix"
        elif any(word in msg_lower for word in ["refactor", "cleanup", "optimize", "improve"]):
            return "refactor"
        elif any(word in msg_lower for word in ["test", "spec", "coverage"]):
            return "test"
        elif any(word in msg_lower for word in ["doc", "readme", "comment", "document"]):
            return "docs"
        else:
            return "other"

    # Анализ коммитов
    commit_types = {}
    task_related = 0
    release_commits = 0

    for commit in developer_commits:
        msg = commit.get("message", "")
        commit_type = classify_commit(msg)
        commit_types[commit_type] = commit_types.get(commit_type, 0) + 1

        # Связь с задачами
        if '#' in msg:
            task_related += 1

        # Релизные ветки
        branches = commit.get("branches", [])
        branch_names = [str(branch).lower() for branch in branches]
        release_keywords = ["main", "master", "release", "prod", "stable"]
        if any(keyword in " ".join(branch_names) for keyword in release_keywords):
            release_commits += 1

    # Балльная система (макс 100)
    scores = {
        "activity": min(total_commits / 50 * 25, 25),  # 25% за активность
        "features": min(commit_types.get("feature", 0) / total_commits * 25, 25),  # 25% за фичи
        "fixes": min(commit_types.get("fix", 0) / total_commits * 20, 20),  # 20% за исправления
        "quality": min(commit_types.get("test", 0) / total_commits * 15, 15),  # 15% за тесты
        "process": min(task_related / total_commits * 15, 15)  # 15% за связь с задачами
    }

    # Бонус за работу в релизных ветках
    release_bonus = min(release_commits / total_commits * 10, 10)

    total_score = sum(scores.values()) + release_bonus

    return min(total_score, 100.0)


def calculate_team_kpi(team_data: dict) -> float:
    """Улучшенный расчет командного KPI"""
    commits_total = team_data.get("commits_total", 0)
    if commits_total == 0:
        return 0.0

    # Основные метрики
    team_size = team_data.get("team_size", 1)
    bus_factor = team_data.get("bus_factor", 1)

    # Соотношения
    feature_ratio = team_data.get("feature_ratio", 0)
    fix_ratio = team_data.get("fix_ratio", 0)
    test_ratio = team_data.get("test_ratio", 0)
    refactor_ratio = team_data.get("refactor_ratio", 0)
    docs_ratio = team_data.get("docs_ratio", 0)
    avg_commits_per_dev = team_data.get("avg_commits_per_dev", 0)

    # 1. Производительность (40%)
    productivity_score = 0
    # Нормализованная активность (целевой показатель 20 коммитов/дев)
    productivity_score += min(avg_commits_per_dev / 20 * 25, 25)

    # Баланс нагрузки (чем ближе к 1, тем лучше)
    load_balance = bus_factor / team_size if team_size > 0 else 0
    productivity_score += load_balance * 15

    # 2. Качество кода (35%)
    quality_score = 0
    quality_score += min(feature_ratio * 20, 20)  # фичи - главный вклад
    quality_score += min(test_ratio * 8, 8)  # тесты
    quality_score += min(docs_ratio * 7, 7)  # документация

    # Штраф за высокий процент фиксов (>30% - тревожный знак)
    if fix_ratio > 0.3:
        quality_score -= (fix_ratio - 0.3) * 20
    quality_score = max(quality_score, 0)

    # 3. Техническое здоровье (25%)
    health_score = 0
    # Рефакторинг в меру (10-20% - оптимально)
    if 0.1 <= refactor_ratio <= 0.2:
        health_score += 15
    else:
        health_score += max(0, 15 - abs(refactor_ratio - 0.15) * 50)  # менее строгий штраф

    # Стабильность (меньше мерж-конфликтов лучше)
    merge_conflicts = team_data.get("merge_conflicts", 0)
    conflict_score = max(0, 10 - min(merge_conflicts, 10))  # максимум 10 конфликтов
    health_score += conflict_score

    total_score = productivity_score + quality_score + health_score
    return min(total_score, 100.0)


def calculate_team_and_individual_kpis(data_dict: dict) -> dict:
    """
    Расчёт как командного, так и индивидуальных KPI (аналитическая часть).
    Вынесено из preprocess.py для разделения логики.
    """
    # Командные метрики
    team_features = extract_features_from_project(data_dict)
    team_data = team_features.iloc[0].to_dict()
    analytical_team_kpi = calculate_team_kpi(team_data)

    # Индивидуальные метрики
    individual_metrics = extract_individual_metrics(data_dict)
    individual_kpis = {
        author: calculate_individual_kpi(commits)
        for author, commits in individual_metrics.items()
    }

    return {
        "team_kpi": analytical_team_kpi,
        "team_metrics": team_data,
        "team_size": len(individual_metrics),
        "individual_kpis": individual_kpis,
        "individual_metrics": individual_metrics
    }
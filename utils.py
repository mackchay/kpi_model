import pandas as pd
import re
from preprocess import extract_features_from_project, extract_individual_metrics, calculate_commit_complexity, \
    improved_classify_commit


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
    """Улучшенный расчет индивидуального KPI с учетом сложности"""
    if not developer_commits:
        return 0.0

    total_commits = len(developer_commits)

    # Рассчитываем общую сложность работы разработчика
    total_complexity = 0
    weighted_commits = 0

    # Анализ коммитов
    commit_types = {}
    task_related = 0
    release_commits = 0
    high_complexity_commits = 0

    for commit in developer_commits:
        msg = commit.get("message", "")
        commit_type = improved_classify_commit(msg)
        commit_types[commit_type] = commit_types.get(commit_type, 0) + 1

        # Рассчитываем сложность коммита
        complexity = calculate_commit_complexity(commit)
        total_complexity += complexity
        weighted_commits += complexity

        # Считаем сложные коммиты (сложность > 1.5)
        if complexity > 1.5:
            high_complexity_commits += 1

        # Связь с задачами
        if '#' in msg:
            task_related += 1

        # Релизные ветки
        branches = commit.get("branches", [])
        branch_names = [str(branch).lower() for branch in branches]
        release_keywords = ["main", "master", "release", "prod", "stable"]
        if any(keyword in " ".join(branch_names) for keyword in release_keywords):
            release_commits += 1

    # Средняя сложность работы разработчика
    avg_complexity = total_complexity / total_commits if total_commits > 0 else 1.0

    # Балльная система (макс 100)
    scores = {
        "activity": min(weighted_commits / 40 * 20, 20),  # 20% за взвешенную активность
        "features": min(commit_types.get("feature", 0) / total_commits * 20, 20),  # 20% за фичи
        "fixes": min(commit_types.get("fix", 0) / total_commits * 18, 18),  # 18% за исправления
        "quality": min(commit_types.get("test", 0) / total_commits * 12, 12),  # 12% за тесты
        "complexity": min(avg_complexity * 10, 15),  # 15% за среднюю сложность
        "process": min(task_related / total_commits * 15, 15)  # 15% за связь с задачами
    }

    # Бонусы
    bonuses = {
        "release": min(release_commits / total_commits * 8, 8),  # до 8% за релизы
        "high_complexity": min(high_complexity_commits / total_commits * 12, 12)  # до 12% за сложные коммиты
    }

    total_score = sum(scores.values()) + sum(bonuses.values())
    return min(total_score, 100.0)


def calculate_team_kpi(team_data: dict) -> float:
    """Улучшенный расчет командного KPI с расширенными метриками"""
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

    # Дополнительные метрики для улучшенной оценки
    complexity_estimate = min(feature_ratio * 1.5 + refactor_ratio * 1.8 + fix_ratio * 1.3, 2.0)

    # 1. Производительность и эффективность (40%)
    productivity_score = 0
    # Взвешенная активность с учетом сложности
    productivity_score += min(avg_commits_per_dev * complexity_estimate / 15 * 20, 20)

    # Баланс нагрузки и устойчивость команды
    load_balance = bus_factor / team_size if team_size > 0 else 0
    stability_bonus = min(bus_factor * 2, 10)  # Поощряем высокий bus factor
    productivity_score += load_balance * 10 + stability_bonus

    # 2. Качество и ценность кода (35%)
    quality_score = 0
    quality_score += min(feature_ratio * 15, 15)  # Бизнес-ценность
    quality_score += min(test_ratio * 7, 7)  # Надежность
    quality_score += min(docs_ratio * 5, 5)  # Поддерживаемость
    quality_score += min(refactor_ratio * 8, 8)  # Техническое здоровье

    # Адаптивный штраф за баги (зависит от контекста)
    critical_fix_threshold = 0.35  # Более гибкий порог
    if fix_ratio > critical_fix_threshold:
        penalty_severity = min((fix_ratio - critical_fix_threshold) * 25, 20)
        quality_score -= penalty_severity
    quality_score = max(quality_score, 0)

    # 3. Техническое совершенство (25%)
    excellence_score = 0

    # Оптимальный рефакторинг с адаптивными диапазонами
    optimal_ranges = {
        "greenfield": (0.05, 0.15),  # Новые проекты
        "standard": (0.1, 0.2),  # Стандартные проекты
        "legacy": (0.15, 0.25)  # Legacy системы
    }

    # Автоматически определяем контекст проекта
    if feature_ratio > 0.6:
        context = "greenfield"
    elif fix_ratio > 0.4:
        context = "legacy"
    else:
        context = "standard"

    low, high = optimal_ranges[context]
    if low <= refactor_ratio <= high:
        excellence_score += 12
    else:
        # Мягкий штраф с учетом контекста
        deviation = min(abs(refactor_ratio - (low + high) / 2), 0.3)
        excellence_score += max(0, 12 - deviation * 30)

    # Стабильность разработки
    merge_conflicts = team_data.get("merge_conflicts", 0)
    conflicts_per_dev = merge_conflicts / team_size if team_size > 0 else merge_conflicts
    conflict_score = max(0, 8 - min(conflicts_per_dev * 2, 8))
    excellence_score += conflict_score

    # Инновации (комбинация feature_ratio и refactor_ratio)
    innovation_ratio = (feature_ratio + refactor_ratio) / 2
    excellence_score += min(innovation_ratio * 5, 5)

    total_score = productivity_score + quality_score + excellence_score
    return min(max(total_score, 0), 100.0)


def calculate_team_and_individual_kpis(data_dict: dict) -> dict:
    """
    Расчёт как командного, так и индивидуальных KPI (аналитическая часть).
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
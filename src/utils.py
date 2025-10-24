import pandas as pd


def calculate_kpi(df: pd.DataFrame) -> pd.Series:
    """
    Расчёт составного KPI (0–100) на основе метрик из preprocess.py.

    Включает признаки из extract_features_from_project:
      - merge_conflicts: количество merge-коммитов
      - bus_factor: число активных участников
      - fix_ratio: доля коммитов с исправлениями
      - feature_ratio: доля коммитов с новыми фичами
      - refactor_ratio: доля рефакторингов
      - docs_ratio: доля документационных коммитов
      - test_ratio: доля тестовых коммитов
      - active_days: количество активных дней
      - team_size: размер команды

    Интерпретация:
      - меньше merge_conflicts → лучше
      - выше bus_factor → лучше
      - меньше fix_ratio → лучше (меньше багов)
      - выше feature_ratio → лучше (команда делает новые фичи)
      - умеренный refactor_ratio → показатель технического здоровья
      - выше docs_ratio и test_ratio → лучше качество кода
      - больше active_days → лучше активность команды
    """

    df = df.copy()

    # --- нормализация ---
    df["merge_conflicts_norm"] = (10 - df["merge_conflicts"].clip(0, 10)) / 10
    df["bus_factor_norm"] = df["bus_factor"].clip(0, 5) / 5
    df["fix_ratio_norm"] = 1 - df["fix_ratio"].clip(0, 1)
    df["feature_ratio_norm"] = df["feature_ratio"].clip(0, 1)
    df["refactor_ratio_norm"] = 1 - abs(df["refactor_ratio"] - 0.1) / 0.1  # идеал ≈10% рефакторингов
    df["docs_ratio_norm"] = df["docs_ratio"].clip(0, 1)
    df["test_ratio_norm"] = df["test_ratio"].clip(0, 1)
    df["active_days_norm"] = df["active_days"].clip(0, 365) / 365  # нормализуем к году
    df["team_size_norm"] = df["team_size"].clip(1, 20) / 20  # нормализуем к разумному максимуму

    # --- формула KPI ---
    # веса можно настраивать под проект
    kpi = (
        0.2 * df["bus_factor_norm"] +
        0.15 * df["merge_conflicts_norm"] +
        0.15 * df["fix_ratio_norm"] +
        0.15 * df["feature_ratio_norm"] +
        0.1 * df["refactor_ratio_norm"] +
        0.1 * df["docs_ratio_norm"] +
        0.1 * df["test_ratio_norm"] +
        0.05 * df["active_days_norm"]
    ) * 100

    return kpi.clip(0, 100)

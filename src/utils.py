import pandas as pd


def calculate_kpi(df: pd.DataFrame) -> pd.Series:
    """
    Формула составного KPI (0–100).
    Весовые коэффициенты можно будет менять под конкретный проект.
    """
    # Нормализуем и ограничиваем метрики
    df = df.copy()
    df["merge_conflicts_norm"] = (10 - df["merge_conflicts"].clip(0, 10)) / 10
    df["bus_factor_norm"] = df["bus_factor"].clip(0, 5) / 5
    df["code_review_norm"] = df["code_review_ratio"].clip(0, 1)

    # Формула KPI
    kpi = (
                0.4 * df["code_review_norm"] +
                0.3 * df["merge_conflicts_norm"] +
                0.3 * df["bus_factor_norm"]
    ) * 100

    return kpi.clip(0, 100)

import pandas as pd
import numpy as np

# Количество команд в датасете
N = 1000
np.random.seed(42)

# Генерация случайных данных для признаков
df = pd.DataFrame({
    "commits_total": np.random.randint(10, 200, size=N),
    "merge_conflicts": np.random.randint(0, 20, size=N),
    "code_review_ratio": np.random.rand(N),
    "bus_factor": np.random.randint(1, 5, size=N),
    "avg_commit_size": np.random.uniform(10, 500, size=N),
    "comment_density": np.random.uniform(0, 1, size=N),
    "churn_rate": np.random.uniform(0, 0.5, size=N),
    "workload_balance": np.random.uniform(0, 1, size=N),
    "refactor_commits": np.random.randint(0, 50, size=N),
    "active_days": np.random.randint(10, 120, size=N),
    "team_size": np.random.randint(2, 10, size=N)
})

# Нормализация признаков для расчёта KPI (только внутри скрипта)
code_review_norm = (df["code_review_ratio"] - df["code_review_ratio"].min()) / (df["code_review_ratio"].max() - df["code_review_ratio"].min())
merge_conflicts_norm = (df["merge_conflicts"] - df["merge_conflicts"].min()) / (df["merge_conflicts"].max() - df["merge_conflicts"].min())
bus_factor_norm = (df["bus_factor"] - df["bus_factor"].min()) / (df["bus_factor"].max() - df["bus_factor"].min())

# Вычисление KPI
df["kpi_score"] = (0.4 * code_review_norm + 0.3 * merge_conflicts_norm + 0.3 * bus_factor_norm) * 100

# Сохраняем только исходные признаки + kpi_score
df.to_csv("team_kpi_dataset.csv", index=False)

print("✅ Датасет создан: team_kpi_dataset.csv")
print(df.head())

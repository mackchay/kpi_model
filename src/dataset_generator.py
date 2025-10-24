import pandas as pd
import numpy as np
from typing import Dict, List
import random
from preprocess import extract_features_from_project
from utils import calculate_kpi

def generate_team_metrics(num_samples: int = 1000) -> pd.DataFrame:
    """
    Генерирует синтетические метрики команд для обучения модели KPI,
    используя структуру данных из preprocess.py.
    
    Args:
        num_samples: Количество образцов для генерации
        
    Returns:
        DataFrame с метриками команд и KPI
    """
    
    data = []
    
    for _ in range(num_samples):
        # Генерируем синтетические данные в формате preprocess.py
        commits_total = random.randint(50, 500)
        merge_conflicts = random.randint(0, 20)
        bus_factor = random.randint(1, 8)
        
        # Генерируем коммиты разных типов
        refactor_commits = random.randint(5, 100)
        fix_commits = random.randint(10, 150)
        feature_commits = random.randint(20, 200)
        docs_commits = random.randint(0, 50)
        test_commits = random.randint(5, 80)
        
        # Вычисляем соотношения
        total_commits = refactor_commits + fix_commits + feature_commits + docs_commits + test_commits
        refactor_ratio = refactor_commits / total_commits if total_commits > 0 else 0
        fix_ratio = fix_commits / total_commits if total_commits > 0 else 0
        feature_ratio = feature_commits / total_commits if total_commits > 0 else 0
        docs_ratio = docs_commits / total_commits if total_commits > 0 else 0
        test_ratio = test_commits / total_commits if total_commits > 0 else 0
        
        active_days = random.randint(10, 365)
        team_size = random.randint(2, 15)
        
        # Создаем DataFrame с метриками
        metrics_df = pd.DataFrame([{
            'commits_total': commits_total,
            'merge_conflicts': merge_conflicts,
            'bus_factor': bus_factor,
            'refactor_commits': refactor_commits,
            'fix_commits': fix_commits,
            'feature_commits': feature_commits,
            'docs_commits': docs_commits,
            'test_commits': test_commits,
            'refactor_ratio': refactor_ratio,
            'fix_ratio': fix_ratio,
            'feature_ratio': feature_ratio,
            'docs_ratio': docs_ratio,
            'test_ratio': test_ratio,
            'active_days': active_days,
            'team_size': team_size
        }])
        
        # Вычисляем KPI используя функцию из utils.py
        kpi_score = calculate_kpi(metrics_df).iloc[0]
        
        data.append({
            'commits_total': commits_total,
            'merge_conflicts': merge_conflicts,
            'bus_factor': bus_factor,
            'refactor_commits': refactor_commits,
            'fix_commits': fix_commits,
            'feature_commits': feature_commits,
            'docs_commits': docs_commits,
            'test_commits': test_commits,
            'refactor_ratio': refactor_ratio,
            'fix_ratio': fix_ratio,
            'feature_ratio': feature_ratio,
            'docs_ratio': docs_ratio,
            'test_ratio': test_ratio,
            'active_days': active_days,
            'team_size': team_size,
            'kpi_score': kpi_score
        })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Генерируем датасет
    df = generate_team_metrics(1000)
    
    # Сохраняем в CSV
    df.to_csv("../data/team_kpi_dataset.csv", index=False)
    print(f"Generated dataset with {len(df)} samples")
    print(f"KPI range: {df['kpi_score'].min():.2f} - {df['kpi_score'].max():.2f}")
    print(f"Mean KPI: {df['kpi_score'].mean():.2f}")
    print("\nDataset columns:")
    print(df.columns.tolist())
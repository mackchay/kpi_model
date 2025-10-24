import pandas as pd
import random
from utils import calculate_team_kpi, calculate_individual_kpi, calculate_commit_complexity


def generate_team_metrics(num_samples: int = 1000) -> pd.DataFrame:
    """
    Генерирует синтетические метрики команд для обучения модели KPI,
    используя улучшенную структуру данных из preprocess.py.
    """
    data = []

    for _ in range(num_samples):
        # Генерируем реалистичные данные команды
        team_size = random.randint(2, 15)
        commits_total = random.randint(50, 500)

        # Генерируем коммиты разных типов с реалистичными соотношениями
        feature_commits = random.randint(int(commits_total * 0.3), int(commits_total * 0.6))  # 30-60%
        fix_commits = random.randint(int(commits_total * 0.1), int(commits_total * 0.3))  # 10-30%
        refactor_commits = random.randint(int(commits_total * 0.05), int(commits_total * 0.2))  # 5-20%
        test_commits = random.randint(int(commits_total * 0.05), int(commits_total * 0.15))  # 5-15%
        docs_commits = random.randint(int(commits_total * 0.02), int(commits_total * 0.1))  # 2-10%

        # Корректируем общее количество коммитов
        actual_total = feature_commits + fix_commits + refactor_commits + test_commits + docs_commits
        if actual_total > commits_total:
            # Масштабируем до нужного общего количества
            scale = commits_total / actual_total
            feature_commits = int(feature_commits * scale)
            fix_commits = int(fix_commits * scale)
            refactor_commits = int(refactor_commits * scale)
            test_commits = int(test_commits * scale)
            docs_commits = commits_total - (feature_commits + fix_commits + refactor_commits + test_commits)

        # Вычисляем соотношения
        refactor_ratio = refactor_commits / commits_total if commits_total > 0 else 0
        fix_ratio = fix_commits / commits_total if commits_total > 0 else 0
        feature_ratio = feature_commits / commits_total if commits_total > 0 else 0
        docs_ratio = docs_commits / commits_total if commits_total > 0 else 0
        test_ratio = test_commits / commits_total if commits_total > 0 else 0

        # Генерируем остальные метрики
        merge_conflicts = random.randint(0, min(20, commits_total // 25))  # Конфликты пропорционально активности
        active_days = random.randint(max(10, commits_total // 10), 365)  # Активные дни связаны с коммитами
        avg_commits_per_dev = commits_total / team_size if team_size > 0 else 0

        # Генерируем реалистичную среднюю сложность
        avg_complexity = random.uniform(0.8, 2.5)  # Реалистичный диапазон сложности

        # Реалистичный bus factor (1-5, зависит от распределения работы)
        if team_size <= 2:
            bus_factor = random.randint(1, 2)
        else:
            bus_factor = random.randint(2, min(5, team_size))

        # Создаем данные команды для расчета KPI
        team_data = {
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
            'avg_commits_per_dev': avg_commits_per_dev,
            'avg_complexity': avg_complexity  # Добавляем новый признак
        }

        # Вычисляем командный KPI используя улучшенную функцию
        kpi_score = calculate_team_kpi(team_data)

        # Добавляем в датасет
        data.append({
            **team_data,
            'kpi_score': kpi_score
        })

    return pd.DataFrame(data)


def generate_individual_metrics(num_samples: int = 1000) -> pd.DataFrame:
    """
    Генерирует синтетические метрики для индивидуальных разработчиков
    с учетом новых улучшенных формул.
    """
    data = []

    for _ in range(num_samples):
        # Генерируем реалистичные данные разработчика
        commits_total = random.randint(10, 200)

        # Генерируем коммиты с реалистичными соотношениями
        feature_commits = random.randint(int(commits_total * 0.2), int(commits_total * 0.5))
        fix_commits = random.randint(int(commits_total * 0.1), int(commits_total * 0.3))
        refactor_commits = random.randint(int(commits_total * 0.05), int(commits_total * 0.15))
        test_commits = random.randint(int(commits_total * 0.05), int(commits_total * 0.2))
        docs_commits = random.randint(int(commits_total * 0.02), int(commits_total * 0.1))

        # Корректируем общее количество
        actual_total = feature_commits + fix_commits + refactor_commits + test_commits + docs_commits
        if actual_total > commits_total:
            scale = commits_total / actual_total
            feature_commits = int(feature_commits * scale)
            fix_commits = int(fix_commits * scale)
            refactor_commits = int(refactor_commits * scale)
            test_commits = int(test_commits * scale)
            docs_commits = commits_total - (feature_commits + fix_commits + refactor_commits + test_commits)

        # Вычисляем соотношения
        feature_ratio = feature_commits / commits_total if commits_total > 0 else 0
        fix_ratio = fix_commits / commits_total if commits_total > 0 else 0
        refactor_ratio = refactor_commits / commits_total if commits_total > 0 else 0
        test_ratio = test_commits / commits_total if commits_total > 0 else 0
        docs_ratio = docs_commits / commits_total if commits_total > 0 else 0

        active_days = random.randint(max(5, commits_total // 5), 180)

        # Создаем реалистичные коммиты с разными типами сообщений и ветками
        commits = []

        # Создаем коммиты для каждого типа с улучшенной структурой
        commits.extend(create_commits_with_type("feature", feature_commits))
        commits.extend(create_commits_with_type("fix", fix_commits))
        commits.extend(create_commits_with_type("refactor", refactor_commits))
        commits.extend(create_commits_with_type("test", test_commits))
        commits.extend(create_commits_with_type("docs", docs_commits))

        # Вычисляем индивидуальный KPI с улучшенной функцией
        kpi_score = calculate_individual_kpi(commits[:commits_total])  # Берем только нужное количество

        # Рассчитываем среднюю сложность коммитов разработчика
        if commits_total > 0:
            total_complexity = sum(calculate_commit_complexity(commit) for commit in commits[:commits_total])
            avg_complexity = total_complexity / commits_total
        else:
            avg_complexity = 1.0

        data.append({
            'commits_total': commits_total,
            'feature_commits': feature_commits,
            'fix_commits': fix_commits,
            'refactor_commits': refactor_commits,
            'test_commits': test_commits,
            'docs_commits': docs_commits,
            'feature_ratio': feature_ratio,
            'fix_ratio': fix_ratio,
            'refactor_ratio': refactor_ratio,
            'test_ratio': test_ratio,
            'docs_ratio': docs_ratio,
            'active_days': active_days,
            'avg_complexity': avg_complexity,  # Добавляем новый признак
            'kpi_score': kpi_score
        })

    return pd.DataFrame(data)


def create_commits_with_type(commit_type: str, count: int) -> list:
    """Создает коммиты указанного типа с реалистичными сообщениями и ветками"""
    if count == 0:
        return []

    commits = []
    # Используем сообщения, которые будут правильно классифицироваться improved_classify_commit
    messages = {
        "feature": [
            "feat: add new user authentication system",
            "feature: implement payment gateway integration",
            "add: new dashboard analytics",
            "implement: real-time notifications",
        ],
        "fix": [
            "fix: resolve memory leak in cache",
            "bug: fix login issue on mobile",
            "fix: correct data validation error",
            "resolve: performance regression",
        ],
        "refactor": [
            "refactor: optimize database queries",
            "cleanup: remove deprecated code",
            "refactor: improve code structure",
            "optimize: reduce bundle size",
        ],
        "test": [
            "test: add unit tests for auth service",
            "spec: integration tests for API",
            "test: coverage for payment module",
            "test: e2e tests for user flow",
        ],
        "docs": [
            "docs: update API documentation",
            "readme: add installation guide",
            "docs: code comments cleanup",
            "document: architecture decisions",
        ]
    }

    # Ветки для расчета сложности
    release_branches = ["main", "master", "release/7.1", "release/7.2", "production"]
    feature_branches = ["feature/login", "feature/payment", "develop", "feature/api", "feature/ui"]

    for i in range(count):
        # Выбираем случайное сообщение
        message = random.choice(messages[commit_type])

        # Добавляем номер задачи с вероятностью (для увеличения сложности)
        has_task = random.random() < 0.4
        if has_task:
            message += f" (#{random.randint(1000, 9999)})"

        # Выбираем ветку (релизные ветки увеличивают сложность)
        is_release = random.random() < 0.25
        branches = [random.choice(release_branches if is_release else feature_branches)]

        # Создаем родителей для коммита (merge коммиты имеют 2+ родителей)
        parents = []
        if random.random() < 0.1:  # 10% коммитов - merge коммиты
            parents = ["parent1", "parent2"]
        else:
            parents = ["parent1"]

        commits.append({
            "message": message,
            "branches": branches,
            "parents": parents  # Добавляем родителей для расчета сложности
        })

    return commits


def analyze_dataset_quality(df: pd.DataFrame, dataset_type: str):
    """Анализирует качество сгенерированного датасета"""
    print(f"\n📈 Анализ {dataset_type} датасета:")
    print(f"   Размер: {len(df)} samples")
    print(f"   KPI диапазон: {df['kpi_score'].min():.2f} - {df['kpi_score'].max():.2f}")
    print(f"   Средний KPI: {df['kpi_score'].mean():.2f} ± {df['kpi_score'].std():.2f}")

    # Анализ сложности если есть такой признак
    if 'avg_complexity' in df.columns:
        print(f"   Средняя сложность: {df['avg_complexity'].mean():.2f} ± {df['avg_complexity'].std():.2f}")

    # Анализ распределения KPI
    kpi_ranges = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
    for low, high in kpi_ranges:
        count = len(df[(df['kpi_score'] >= low) & (df['kpi_score'] < high)])
        percentage = (count / len(df)) * 100
        print(f"   KPI {low}-{high}: {count} samples ({percentage:.1f}%)")

    return df


if __name__ == "__main__":
    print("🚀 Generating improved datasets for dual model training...")

    # Генерируем командный датасет
    print("\n📊 Generating team KPI dataset...")
    team_df = generate_team_metrics(50000)  # Уменьшим для скорости, можно вернуть 50000
    team_df = analyze_dataset_quality(team_df, "Team")
    team_df.to_csv("data/team_kpi_dataset.csv", index=False)

    # Генерируем индивидуальный датасет
    print("\n👤 Generating individual KPI dataset...")
    individual_df = generate_individual_metrics(50000)  # Уменьшим для скорости
    individual_df = analyze_dataset_quality(individual_df, "Individual")
    individual_df.to_csv("data/individual_kpi_dataset.csv", index=False)

    print("\n📋 Final dataset summary:")
    print(f"✅ Team dataset: {len(team_df)} samples")
    print(f"✅ Individual dataset: {len(individual_df)} samples")

    print("\n🔍 Sample team data:")
    print(team_df[['commits_total', 'team_size', 'feature_ratio', 'fix_ratio', 'avg_complexity', 'kpi_score']].head())

    print("\n🔍 Sample individual data:")
    print(individual_df[['commits_total', 'feature_ratio', 'fix_ratio', 'avg_complexity', 'kpi_score']].head())
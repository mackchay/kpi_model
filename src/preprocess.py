import pandas as pd
from datetime import datetime
from collections import defaultdict


def classify_commit(message: str) -> str:
    """Классификация коммитов по сообщению"""
    msg = message.lower()
    if any(k in msg for k in ["feat", "add", "implement", "create"]):
        return "feature"
    elif any(k in msg for k in ["fix", "bug", "error", "issue", "resolve"]):
        return "fix"
    elif any(k in msg for k in ["refactor", "cleanup", "remove", "optimize"]):
        return "refactor"
    elif any(k in msg for k in ["test", "unit", "integration"]):
        return "test"
    elif any(k in msg for k in ["doc", "readme"]):
        return "docs"
    return "other"


def extract_features_from_project(data: dict) -> pd.DataFrame:
    """
    Преобразует JSON из /app/sourcecode/api/api/v2/projects/.../commits
    в DataFrame с вычисляемыми признаками (только из данных коммитов).
    """

    project = data.get("project") or data  # API может возвращать с ключом project или без
    repos = project.get("repos", [])

    commits_total = 0
    merge_conflicts = 0
    refactor_commits = 0
    fix_commits = 0
    feature_commits = 0
    docs_commits = 0
    test_commits = 0
    authors = defaultdict(int)
    active_days = set()

    # Обходим все репозитории и коммиты
    for repo in repos:
        for commit in repo.get("commits", []):
            commits_total += 1

            msg = commit.get("message", "")
            ctype = classify_commit(msg)
            author_name = commit.get("author", {}).get("name", "Unknown")
            authors[author_name] += 1

            # типы коммитов
            if ctype == "refactor":
                refactor_commits += 1
            elif ctype == "fix":
                fix_commits += 1
            elif ctype == "feature":
                feature_commits += 1
            elif ctype == "docs":
                docs_commits += 1
            elif ctype == "test":
                test_commits += 1

            # Merge-коммит, если больше одного родителя
            if len(commit.get("parents", [])) > 1:
                merge_conflicts += 1

            # День активности
            created_at = commit.get("created_at")
            if created_at:
                try:
                    dt = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")
                    active_days.add(dt.date())
                except ValueError:
                    pass

    # --- производные метрики ---
    team_size = len(authors)
    avg_commits_per_dev = commits_total / team_size if team_size else 0
    active_days_count = len(active_days)
    refactor_ratio = refactor_commits / commits_total if commits_total else 0
    fix_ratio = fix_commits / commits_total if commits_total else 0
    feature_ratio = feature_commits / commits_total if commits_total else 0
    docs_ratio = docs_commits / commits_total if commits_total else 0
    test_ratio = test_commits / commits_total if commits_total else 0

    # Bus factor: грубая оценка — сколько людей делают >20% всех коммитов
    high_activity_threshold = 0.2 * commits_total
    bus_factor = sum(1 for c in authors.values() if c > high_activity_threshold)
    bus_factor = min(5, bus_factor)  # нормируем до 5 для удобства

    features = pd.DataFrame([{
        "commits_total": commits_total,
        "merge_conflicts": merge_conflicts,
        "bus_factor": bus_factor,
        "refactor_commits": refactor_commits,
        "fix_commits": fix_commits,
        "feature_commits": feature_commits,
        "docs_commits": docs_commits,
        "test_commits": test_commits,
        "refactor_ratio": refactor_ratio,
        "fix_ratio": fix_ratio,
        "feature_ratio": feature_ratio,
        "docs_ratio": docs_ratio,
        "test_ratio": test_ratio,
        "active_days": active_days_count,
        "team_size": team_size
    }])

    return features


# === Пример локального запуска ===
if __name__ == "__main__":
    import json
    with open("sample_commits.json", "r", encoding="utf-8") as f:
        sample = json.load(f)

    df = extract_features_from_project(sample)
    print(df)

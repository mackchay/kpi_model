import joblib
import json
from pathlib import Path


def load_team_model():
    """Загружает модель для командного KPI"""
    team_model_path = "models/team_kpi_regressor.pkl"
    team_info_path = "models/team_model_info.json"

    if not Path(team_model_path).exists():
        raise FileNotFoundError(f"Team model not found at {team_model_path}. Please train the model first.")

    if not Path(team_info_path).exists():
        raise FileNotFoundError(f"Team model info not found at {team_info_path}. Please train the model first.")

    try:
        # Загружаем модель
        team_model = joblib.load(team_model_path)

        # Загружаем информацию о модели
        with open(team_info_path, "r") as f:
            model_info = json.load(f)

        print(f"✅ Team model loaded successfully")
        print(f"   Features: {len(model_info['features'])}")
        print(f"   Performance: R² = {model_info['performance']['test_r2']:.4f}")

        return team_model, model_info

    except Exception as e:
        raise Exception(f"Error loading team model: {e}")


def load_individual_model():
    """Загружает модель для индивидуального KPI"""
    individual_model_path = "models/individual_kpi_regressor.pkl"
    individual_info_path = "models/individual_model_info.json"

    if not Path(individual_model_path).exists():
        raise FileNotFoundError(f"Individual model not found at {individual_model_path}. Please train the model first.")

    if not Path(individual_info_path).exists():
        raise FileNotFoundError(
            f"Individual model info not found at {individual_info_path}. Please train the model first.")

    try:
        # Загружаем модель
        individual_model = joblib.load(individual_model_path)

        # Загружаем информацию о модели
        with open(individual_info_path, "r") as f:
            model_info = json.load(f)

        print(f"✅ Individual model loaded successfully")
        print(f"   Features: {len(model_info['features'])}")
        print(f"   Performance: R² = {model_info['performance']['test_r2']:.4f}")

        return individual_model, model_info

    except Exception as e:
        raise Exception(f"Error loading individual model: {e}")


def get_team_features():
    """Возвращает список признаков для командной модели"""
    _, model_info = load_team_model()
    return model_info["features"]


def get_individual_features():
    """Возвращает список признаков для индивидуальной модели"""
    _, model_info = load_individual_model()
    return model_info["features"]


def check_model_compatibility():
    """Проверяет совместимость загруженных моделей с текущей системой"""
    try:
        team_model, team_info = load_team_model()
        individual_model, individual_info = load_individual_model()

        # Проверяем наличие ключевых признаков
        team_features = team_info["features"]
        individual_features = individual_info["features"]

        # Ключевые признаки, которые должны быть в моделях после обновления
        expected_team_features = ['avg_complexity', 'commits_total', 'team_size']
        expected_individual_features = ['avg_complexity', 'commits_total', 'active_days']

        missing_team_features = [f for f in expected_team_features if f not in team_features]
        missing_individual_features = [f for f in expected_individual_features if f not in individual_features]

        if missing_team_features:
            print(f"⚠️  В командной модели отсутствуют признаки: {missing_team_features}")
        else:
            print("✅ Командная модель совместима с новой системой")

        if missing_individual_features:
            print(f"⚠️  В индивидуальной модели отсутствуют признаки: {missing_individual_features}")
        else:
            print("✅ Индивидуальная модель совместима с новой системой")

        return len(missing_team_features) == 0 and len(missing_individual_features) == 0

    except Exception as e:
        print(f"❌ Ошибка проверки совместимости моделей: {e}")
        return False


# Функция для обратной совместимости
def load_model():
    """Legacy функция для обратной совместимости"""
    return load_team_model()


def load_config():
    """Загружает конфигурацию (для обратной совместимости)"""
    # Эта функция может быть реализована если нужна
    # Пока возвращаем пустой словарь для совместимости
    return {}
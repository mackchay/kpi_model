import joblib
import json
from pathlib import Path

def load_individual_model():
    """Загружает модель для индивидуального KPI"""
    individual_model_path = "models/individual_kpi_regressor.pkl"
    individual_info_path = "models/individual_model_info.json"
    
    if not Path(individual_model_path).exists():
        raise FileNotFoundError(f"Individual model not found at {individual_model_path}. Please train the model first.")
    
    if not Path(individual_info_path).exists():
        raise FileNotFoundError(f"Individual model info not found at {individual_info_path}. Please train the model first.")
    
    # Загружаем модель
    individual_model = joblib.load(individual_model_path)
    
    # Загружаем информацию о модели
    with open(individual_info_path, "r") as f:
        model_info = json.load(f)
    
    return individual_model, model_info

def get_individual_features():
    """Возвращает список признаков для индивидуальной модели"""
    _, model_info = load_individual_model()
    return model_info["features"]

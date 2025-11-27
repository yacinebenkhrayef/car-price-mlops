"""Script de training avec MLflow tracking"""

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import argparse
from datetime import datetime

from src.data.preprocessing import DataPreprocessor


class ModelTrainer:
    """Classe pour entraîner et évaluer les modèles"""
    
    def __init__(self, config_path: str = "configs/model_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup MLflow
        mlflow.set_tracking_uri(self.config.get('mlflow_uri', 'http://localhost:5000'))
        mlflow.set_experiment(self.config.get('experiment_name', 'car_price_prediction'))
    
    def load_and_prepare_data(self, data_path: str):
        """Charger et préparer les données"""
        print("📊 Chargement des données...")
        
        # Preprocessing
        preprocessor = DataPreprocessor()
        df = preprocessor.load_data(data_path)
        X, y = preprocessor.fit_transform(df)
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config['test_size'],
            random_state=self.config['random_state']
        )
        
        # Sauvegarder preprocessor
        preprocessor.save("models/preprocessor.pkl")
        
        return X_train, X_test, y_train, y_test, preprocessor
    
    def get_models(self):
        """Définir les modèles à tester"""
        models = {
            'linear_regression': LinearRegression(),
            
            'random_forest': RandomForestRegressor(
                n_estimators=self.config['random_forest']['n_estimators'],
                max_depth=self.config['random_forest']['max_depth'],
                random_state=self.config['random_state']
            ),
            
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=self.config['gradient_boosting']['n_estimators'],
                learning_rate=self.config['gradient_boosting']['learning_rate'],
                max_depth=self.config['gradient_boosting']['max_depth'],
                random_state=self.config['random_state']
            ),
            
            'xgboost': xgb.XGBRegressor(
                n_estimators=self.config['xgboost']['n_estimators'],
                learning_rate=self.config['xgboost']['learning_rate'],
                max_depth=self.config['xgboost']['max_depth'],
                random_state=self.config['random_state']
            )
        }
        
        return models
    
    def evaluate_model(self, model, X_test, y_test):
        """Évaluer le modèle"""
        y_pred = model.predict(X_test)
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        return metrics, y_pred
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Entraîner tous les modèles avec MLflow tracking"""
        models = self.get_models()
        results = {}
        
        for model_name, model in models.items():
            print(f"\n🚀 Training {model_name}...")
            
            with mlflow.start_run(run_name=model_name):
                # Log paramètres
                mlflow.log_params(model.get_params())
                
                # Train
                model.fit(X_train, y_train)
                
                # Evaluate
                metrics, y_pred = self.evaluate_model(model, X_test, y_test)
                
                # Log metrics
                mlflow.log_metrics(metrics)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, 
                                           cv=5, 
                                           scoring='neg_mean_squared_error')
                cv_rmse = np.sqrt(-cv_scores.mean())
                mlflow.log_metric("cv_rmse", cv_rmse)
                
                # Log model
                mlflow.sklearn.log_model(model, "model")
                
                # Sauvegarder localement le meilleur modèle
                results[model_name] = {
                    'model': model,
                    'metrics': metrics,
                    'cv_rmse': cv_rmse
                }
                
                print(f"✅ {model_name} - RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.3f}")
        
        return results
    
    def save_best_model(self, results):
        """Sauvegarder le meilleur modèle"""
        # Trouver le meilleur modèle (RMSE le plus bas)
        best_model_name = min(results, key=lambda x: results[x]['metrics']['rmse'])
        best_model = results[best_model_name]['model']
        
        # Sauvegarder
        model_path = Path("models") / "best_model.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        import joblib
        joblib.dump(best_model, model_path)
        
        # Sauvegarder métadonnées
        metadata = {
            'model_name': best_model_name,
            'metrics': results[best_model_name]['metrics'],
            'trained_at': datetime.now().isoformat()
        }
        
        import json
        with open("models/model_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n🏆 Meilleur modèle: {best_model_name}")
        print(f"📁 Sauvegardé dans: {model_path}")
        
        return best_model_name, best_model


def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/processed/cars_cleaned.csv')
    parser.add_argument('--config', default='configs/model_config.yaml')
    args = parser.parse_args()
    
    # Training
    trainer = ModelTrainer(args.config)
    X_train, X_test, y_train, y_test, preprocessor = trainer.load_and_prepare_data(args.data)
    results = trainer.train_models(X_train, X_test, y_train, y_test)
    best_name, best_model = trainer.save_best_model(results)
    
    print("\n✨ Training terminé!")
    print(f"🎯 Lancez MLflow UI: mlflow ui")


if __name__ == "__main__":
    main()

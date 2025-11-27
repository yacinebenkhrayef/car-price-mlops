"""FastAPI application pour prédiction de prix"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from typing import Optional
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Créer app
app = FastAPI(
    title="Car Price Prediction API",
    description="API MLOps pour prédire le prix des voitures d'occasion",
    version="1.0.0"
)


# Pydantic models
class CarFeatures(BaseModel):
    """Schéma pour les features d'entrée"""
    brand: str = Field(..., example="Toyota")
    model: str = Field(..., example="Corolla")
    year: int = Field(..., ge=1990, le=2024, example=2018)
    km_driven: int = Field(..., ge=0, example=45000)
    fuel: str = Field(..., example="Petrol")
    transmission: str = Field(..., example="Manual")
    owner: str = Field(..., example="First Owner")
    
    class Config:
        schema_extra = {
            "example": {
                "brand": "Toyota",
                "model": "Corolla",
                "year": 2018,
                "km_driven": 45000,
                "fuel": "Petrol",
                "transmission": "Manual",
                "owner": "First Owner"
            }
        }


class PredictionResponse(BaseModel):
    """Schéma pour la réponse"""
    predicted_price: float
    confidence_interval: Optional[dict] = None
    model_version: str
    prediction_time: str


# Charger modèle et preprocessor au démarrage
@app.on_event("startup")
async def load_models():
    """Charger les modèles au démarrage"""
    global model, preprocessor, model_metadata
    
    try:
        model = joblib.load("models/best_model.pkl")
        preprocessor = joblib.load("models/preprocessor.pkl")
        
        import json
        with open("models/model_metadata.json", 'r') as f:
            model_metadata = json.load(f)
        
        logger.info("✅ Modèles chargés avec succès")
    except Exception as e:
        logger.error(f"❌ Erreur chargement modèles: {e}")
        raise


@app.get("/")
async def root():
    """Endpoint racine"""
    return {
        "message": "Car Price Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "model_info": "/model-info"
        }
    }


@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/model-info")
async def model_info():
    """Informations sur le modèle"""
    return {
        "model_name": model_metadata.get('model_name'),
        "metrics": model_metadata.get('metrics'),
        "trained_at": model_metadata.get('trained_at')
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: CarFeatures):
    """
    Prédire le prix d'une voiture
    
    - **brand**: Marque de la voiture
    - **year**: Année de fabrication
    - **km_driven**: Kilométrage
    - **fuel**: Type de carburant (Petrol, Diesel, CNG, etc.)
    - **transmission**: Type de transmission (Manual, Automatic)
    - **owner**: Nombre de propriétaires précédents
    """
    try:
        # Convertir en DataFrame
        input_data = pd.DataFrame([features.dict()])
        
        # Preprocessing
        X = preprocessor.transform(input_data)
        
        # Prédiction
        prediction = model.predict(X)[0]
        
        # Log pour monitoring
        logger.info(f"Prédiction: {prediction:.2f} pour {features.brand} {features.model}")
        
        return PredictionResponse(
            predicted_price=round(float(prediction), 2),
            model_version=model_metadata.get('model_name', 'unknown'),
            prediction_time=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Erreur prédiction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
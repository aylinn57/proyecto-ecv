# app/main.py
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()

# Agregar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Puedes restringir a tu dominio si quieres más adelante
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo (RandomForest NO requiere scaler)
modelo = joblib.load(os.path.join(os.path.dirname(__file__), "modelo_ecv_rf.joblib"))

# Servir templates
templates = Jinja2Templates(directory="app/templates")

# Montar carpeta static (para logo, imágenes, etc.)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# MODELO DE DATOS
class Paciente(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float

# RUTA RAÍZ → muestra el HTML
@app.get("/", response_class=HTMLResponse)
async def formulario(request: Request):
    return templates.TemplateResponse("formulario_ecv.html", {"request": request})

# RUTA API POST
@app.post("/predecir_ecv")
async def predecir_ecv(paciente: Paciente):
    X_new = np.array([[paciente.age, paciente.sex, paciente.cp, paciente.trestbps,
                       paciente.chol, paciente.fbs, paciente.restecg, paciente.thalach,
                       paciente.exang, paciente.oldpeak, paciente.slope,
                       paciente.ca, paciente.thal]])
    
    # SIN SCALER
    proba = modelo.predict_proba(X_new)[0][1]
    prediccion = modelo.predict(X_new)[0]
    
    resultado = "Positivo: Alto riesgo de ECV" if prediccion == 1 else "Negativo: Sin ECV"
    
    return {
        "resultado": resultado,
        "probabilidad": round(proba, 4)
    }

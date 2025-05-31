# ==========================================
# CLIENTE PYTHON para consumir la API de ECV
# Fabián 🚀
# ==========================================

import requests
import json

# URL de la API local (si la corres con uvicorn en tu máquina)
url = "http://127.0.0.1:8000/predecir_ecv"

# Datos del paciente que queremos predecir
paciente = {
    "age": 60,
    "sex": 1,
    "cp": 2,
    "trestbps": 130,
    "chol": 230,
    "fbs": 0,
    "restecg": 0,
    "thalach": 140,
    "exang": 0,
    "oldpeak": 1.5,
    "slope": 1,
    "ca": 0,
    "thal": 3
}

# Mostrar qué se va a enviar
print("\n📤 Enviando paciente a la API:")
print(json.dumps(paciente, indent=2))

# Hacer la petición POST
response = requests.post(url, json=paciente)

# Revisar respuesta
if response.status_code == 200:
    resultado = response.json()
    print("\n✅ Respuesta de la API:")
    print(json.dumps(resultado, indent=2))
else:
    print("\n❌ Error al llamar la API:")
    print("Código de estado:", response.status_code)
    print("Contenido:", response.text)

# 🫀 Predicción de Enfermedad Cardiovascular (ECV) con FastAPI

Proyecto de inteligencia artificial para la **clasificación de riesgo de enfermedades cardiovasculares (ECV)** mediante un modelo de Machine Learning, desplegado online con FastAPI + Render.

## 🚀 Demo en producción:

🎓 **Proyecto académico - CEMP IA 2025**  
👩‍💻 _by [aylinn57](https://github.com/aylinn57)_

👉 [https://proyecto-ecv.onrender.com](https://proyecto-ecv.onrender.com)

## 🛠️ Tecnologías usadas:

- **Python 3.11**
- **FastAPI**
- **scikit-learn**
- **Joblib** (serialización del modelo)
- **HTML + Bootstrap** (formulario web)
- **Render** (deploy cloud gratuito)

## 🤖 Modelo:

- Algoritmo: **Random Forest Classifier**
- Entrenado sobre dataset *heart.csv* con las siguientes variables:

```txt
- age
- sex
- cp (tipo de dolor en pecho)
- trestbps (presión en reposo)
- chol (colesterol)
- fbs (>120 mg/dl)
- restecg
- thalach (máximo ritmo cardíaco)
- exang (angina inducida por ejercicio)
- oldpeak
- slope
- ca (número de vasos principales)
- thal

📝 Descripción:
Este proyecto permite que un usuario pueda:

1️⃣ Ingresar los valores clínicos de un paciente en un formulario web
2️⃣ Hacer clic en "Predecir ECV"
3️⃣ Recibir el resultado:
- Positivo: Alto riesgo de enfermedad cardiovascular
- Negativo: Sin ECV
- Además muestra la probabilidad de riesgo (score del modelo)

Todo esto en tiempo real, gracias a un endpoint /predecir_ecv en FastAPI que recibe el JSON del paciente y devuelve la predicción.

⚙️ Estructura del proyecto:

proyecto-ecv/
├── app/
│ ├── main.py
│ ├── templates/
│ │ └── formulario_ecv.html
│ ├── static/
│ │ └── logo.png
│ ├── modelo_ecv_rf.joblib
│ ├── scaler_ecv.joblib
├── requirements.txt
├── README.md

🚀 Cómo desplegar:
1️⃣ Clona el repo:
git clone https://github.com/aylinn57/proyecto-ecv.git
cd proyecto-ecv

2️⃣ Instala dependencias:
pip install -r requirements.txt

3️⃣ Corre localmente:
uvicorn app.main:app --reload

4️⃣ Accede a:

Formulario: http://127.0.0.1:8000/

Documentación Swagger: http://127.0.0.1:8000/docs

🌐 Deploy online:
Este proyecto ha sido desplegado en Render como Web Service de FastAPI.

💡 Autor:
👩‍💻 Proyecto desarrollado por aylinn57-MASA

Colaboración técnica: [MASA + FastAPI + Render].

📚 Inspiración:
Este tipo de proyectos demuestra cómo IA aplicada en medicina puede facilitar herramientas simples de prevención para uso clínico o educativo.

⭐ Si te gusta, ¡dale Star ⭐ al repo y sígueme! 🚀


---

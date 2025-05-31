# ==========================================
# PREDICCIÓN DE PACIENTE NUEVO CON MODELO DE ECV
# Fabián 🚀
# ==========================================

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# 1️⃣ Cargar modelo entrenado
modelo = joblib.load('modelo_ecv_rf.pkl')
print("✅ Modelo cargado correctamente.")

# 2️⃣ Definir los datos de un paciente nuevo
# ⚠️ Deben tener las MISMAS variables y en el MISMO orden que las usadas en el entrenamiento
# Aquí te pongo un ejemplo con variables "simuladas"

paciente_nuevo = pd.DataFrame({
    'age': [55],
    'sex': [1],            # 1=Hombre, 0=Mujer
    'cp': [3],             # Tipo de dolor en el pecho
    'trestbps': [140],     # Presión en reposo
    'chol': [250],         # Colesterol
    'fbs': [0],            # Glucosa en ayunas >120 mg/dl (1=True, 0=False)
    'restecg': [1],        # Resultados del ECG
    'thalach': [150],      # Máximo ritmo cardíaco alcanzado
    'exang': [0],          # Angina inducida por ejercicio
    'oldpeak': [1.0],      # Depresión ST
    'slope': [2],          # Pendiente ST
    'ca': [0],             # N° de vasos coloreados
    'thal': [3]            # Resultado del escáner de talio
})

print("\n📋 Datos del paciente nuevo:")
print(paciente_nuevo)

# 3️⃣ Aplicar mismo preprocesamiento (winsorización + escalado en las columnas necesarias)

# Primero winsorizamos las columnas correspondientes
def winsorize_series(series, lower=0.01, upper=0.99):
    q_low = series.quantile(lower)
    q_high = series.quantile(upper)
    return series.clip(q_low, q_high)

# Para el ejemplo, como solo es un paciente, no se calcula percentil real → se omite winsorización (opcional en producción).

# 4️⃣ Estándarizar las columnas numéricas como en el entrenamiento
columnas_a_escalar = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# ⚠️ Para que sea EXACTAMENTE igual, carga el scaler usado. Si no lo guardaste, aquí se puede reentrenar rápido (no es ideal pero funciona para test):
# Ideal: joblib.dump(scaler, 'scaler.pkl') → luego cargarlo.
# Por ahora vamos a reescalar con el mismo fit:
df_entrenamiento = pd.read_csv('heart.csv')
df_entrenamiento['target'] = df_entrenamiento['num'].apply(lambda x: 1 if x > 0 else 0)
df_entrenamiento.drop(columns=['num'], inplace=True)
X_train = df_entrenamiento.drop(columns=['target'])

#scaler = StandardScaler()
#scaler.fit(X_train[columnas_a_escalar])

# Cargar scaler entrenado
scaler = joblib.load('scaler.pkl')
print("✅ Scaler cargado correctamente.")

# Aplicar el escalado al paciente nuevo
paciente_nuevo[columnas_a_escalar] = scaler.transform(paciente_nuevo[columnas_a_escalar])

# 5️⃣ Hacer la predicción
prediccion = modelo.predict(paciente_nuevo)
probabilidad = modelo.predict_proba(paciente_nuevo)[0][1]  # Probabilidad de clase 1 (ECV)

# 6️⃣ Mostrar el resultado
print("\n🔮 Predicción:")
if prediccion[0] == 1:
    print(f"⚠️ El modelo predice que este paciente TIENE RIESGO DE ECV. Probabilidad: {probabilidad:.2%}")
else:
    print(f"✅ El modelo predice que este paciente NO tiene riesgo alto de ECV. Probabilidad: {probabilidad:.2%}")

# FIN

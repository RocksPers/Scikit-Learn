import joblib as jb
import numpy as np

model = jb.load('C:/ESPOCH/Mineria_Proyecto/scikit-learn/app/models/Modelo0.15.pkl')

# X_test = np.array([0, 25.6, 80.96, 21.54, 0.04, 0.63, 188.51, 2, 5, 0])  # Debe dar un valor 0
X_test = np.array([0, 23.08, 89.45, 21.07, 0.07, 0.87, 207.05, 4, 5, 10])  # Debe dar un valor 1
# X_test = np.array([0.01, 23.09, 89.45, 21.07, 0.08, 0.88, 207.05, 2, 4, 0])  # Debe dar un valor 1

# Obtener las probabilidades de las clases
probabilities = model.predict(X_test.reshape(1, -1))

# Definir un umbral para decidir si es 0 o 1
threshold = 0.5

# Redondear la probabilidad de la clase 1 a 0 o 1 según el umbral
prediction = (probabilities[0] > threshold).astype(int)

if prediction == 1:
    print('Su cultivo tiene la enfermedad:', prediction)
elif prediction == 0:
    print('Su cultivo no tiene la enfermedad', prediction)

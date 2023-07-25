import pandas as pd
import numpy as np
import sklearn
from utils import Utils
from models import Models

if __name__ == "__main__":
    utils = Utils()
    models = Models()
    data = utils.load_from_csv('./app/in/Sensor_.csv')

    # Limitar el tamaño de los datos para el entrenamiento
    sample_size = 10  # Establece el tamaño de la muestra deseada
    data_sample = data.sample(n=sample_size, random_state=42)  # Selecciona una muestra aleatoria de tamaño n

    X, y = utils.features_target(data_sample, ['INCIDENCIA'],['INCIDENCIA'])
    models.grid_training(X,y)
    print(data_sample)
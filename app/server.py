import joblib #importa las bibliotecas joblib para cargar el modelo y numpy y Flask para crear la aplicación web.
import numpy as np

from flask import Flask
from flask import jsonify

app = Flask(__name__)
#POSTMAN PARA PRUEBAS
@app.route('/predict', methods=['GET'])
def predict():
    X_test = np.array([0.009027778,23.08511458,89.454875,21.0713125,0.078125,0.876736111,207.0520833])
    prediction = model.predict(X_test.reshape(1,-1))
    
    return jsonify({'prediccion' : list(prediction)})
if __name__ == "__main__":
    
    model = joblib.load('./models/Modelo0.15.pkl')
    app.run(port=8080)
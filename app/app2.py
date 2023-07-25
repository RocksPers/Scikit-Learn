from flask import Flask, jsonify, request
from flask_pymongo import PyMongo
from flask_cors import CORS
from bson import json_util, ObjectId
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Configurar la conexión a la base de datos MongoDB
mongo_uri = os.environ.get('MONGODB_URI')
if not mongo_uri:
    raise Exception('La cadena de conexión de MongoDB no está configurada')

client = MongoClient(mongo_uri)
db = client['datos_manuales']
collection = db['Sensor']
collection_manuales = db['manuales']

@app.route('/api/datos-sensor')
def get_datos_sensor():
    datos = list(collection.find().limit(10))
    json_data = json_util.dumps(datos, default=json_util.default)
    return json_data

@app.route('/api/datos', methods=['GET'])
def obtener_datos():
    try:
        datos = list(collection.find())
        return jsonify(datos), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/datos-manuales/<id>', methods=['PUT'])
def editar_dato_manual(id):
    # ... (Lógica para actualizar un dato manual según su ID) ...
    pass

@app.route('/api/datos-manuales/<id>', methods=['DELETE'])
def delete_dato_manual(id):
    # ... (Lógica para eliminar un dato manual según su ID) ...
    pass

@app.route('/api/datos-manuales', methods=['GET'])
def get_datos_manuales():
    datos = list(collection_manuales.find())
    json_data = json_util.dumps(datos, default=json_util.default)
    return json_data

@app.route('/api/datos-manuales', methods=['POST'])
def guardar_datos_manuales():
    # ... (Lógica para guardar datos manuales en la base de datos) ...
    pass

@app.route('/api/predecir', methods=['POST'])
def dataarray():
    # ... (Lógica para realizar la predicción y guardar los datos en una nueva colección) ...
    pass


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)

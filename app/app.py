import os
import flask
import joblib #importa las bibliotecas joblib para cargar el modelo y numpy y Flask para crear la aplicación web.
import numpy as np
from flask import Flask, jsonify, request
from json import JSONEncoder
from flask_cors import CORS
from uuid import UUID
from pymongo import MongoClient
from bson import json_util
from dmanuales import Datos_manuales

app = Flask(__name__)
CORS(app)
mongo_uri = os.environ.get('MONGODB_URI')

if not mongo_uri:
    raise Exception('La cadena de conexión de MongoDB no está configurada')

client = MongoClient(mongo_uri)
db = client['datos_manuales']
collection = db['Sensor']
collection_manuales = db['manuales']

class CustomJSONEncoder(JSONEncoder):
    def default(self, o):
        if '_id' in o:
            o['id'] = str(o['_id'])
            del o['_id']
        return super().default(o)

app.json_encoder = CustomJSONEncoder




#
@app.route('/api/datos-sensor')
def get_datos_sensor():
    datos = list(collection.find().limit(10))
    json_data = json_util.dumps(datos, default=json_util.default)
    return json_data

# Ruta para obtener los datos de la base de datos
@app.route('/api/datos', methods=['GET'])
def obtener_datos():
    try:
        datos = list(collection.find())
        return jsonify(datos), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/datos-manuales/<id>', methods=['PUT'])
def editar_dato_manual(id):
    data = request.json
    temperatura = data['temperatura']
    humedad = data['humedad']
    luminosidad = data['luminosidad']
    ph_suelo = data['phSuelo']
    nivel_agua = data['nivelAgua']
    altitud = data['altitud']

    # Verificar si el ID es un UUID válido
    try:
        UUID(id)
    except ValueError:
        return jsonify(message='ID inválido'), 400

    # Buscar el dato manual por su ID
    dato_manual = collection_manuales.find_one({'id': id})

    if dato_manual is None:
        return jsonify(message='No se encontró ningún dato manual con el ID proporcionado'), 404

    # Actualizar los campos del dato manual
    dato_manual['temperatura'] = temperatura
    dato_manual['humedad'] = humedad
    dato_manual['luminosidad'] = luminosidad
    dato_manual['phSuelo'] = ph_suelo
    dato_manual['nivelAgua'] = nivel_agua
    dato_manual['altitud'] = altitud

    # Guardar los cambios en la base de datos
    collection_manuales.update_one({'id': id}, {'$set': dato_manual})

    return jsonify(message='Dato manual actualizado exitosamente'), 200

#
@app.route('/api/datos-manuales/<id>', methods=['DELETE'])
def delete_dato_manual(id):
    # Código para manejar la solicitud DELETE
    # Puedes implementar aquí la lógica para eliminar un dato manual específico según su ID

    # Ejemplo: eliminar un dato manual según su ID
    result = collection_manuales.delete_one({'_id': id})
    
    if result.deleted_count == 0:
        return jsonify(message='No se encontró ningún dato manual con el ID proporcionado'), 404

    return jsonify(message='Dato manual eliminado exitosamente'), 200

@app.route('/api/datos-manuales', methods=['GET'])
def get_datos_manuales():
    datos = list(collection_manuales.find_one())
    json_data = json_util.dumps(datos, default=json_util.default)
    return json_data

@app.route('/api/datos-manuales', methods=['POST'])
def guardar_datos_manuales():
    data = request.json
    id = data['id']
    temperatura = data['temperatura']
    humedad = data['humedad']
    luminosidad = data['luminosidad']
    ph_suelo = data['phSuelo']
    nivel_agua = data['nivelAgua']
    altitud = data['altitud']

    nuevo_dato = Datos_manuales(
        id=id,
        temperatura=temperatura,
        humedad=humedad,
        luminosidad=luminosidad,
        ph_suelo=ph_suelo,
        nivel_agua=nivel_agua,
        altitud=altitud,
    )

    collection_manuales.insert_one(nuevo_dato.ColleccionDatos())

    return jsonify(message='Datos manuales guardados exitosamente'), 200


if __name__ == '__main__':

    app.run(host='0.0.0.0', debug=True, port=5000)

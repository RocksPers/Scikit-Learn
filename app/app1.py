from flask import Flask, jsonify, request
from flask_pymongo import PyMongo 
from flask_cors import CORS
from flask_caching import Cache
from pymongo import MongoClient
import os
from bson import json_util, ObjectId

import joblib
import numpy as np

from entorno import FLASK_APP_BD #para una coneccion de forma local
from entorno import FLASK_APP_BDA #para una coneccion de en mongoAtlas

app = Flask(__name__)
app.config['MONGO_URI']= FLASK_APP_BDA
app.config['MONGO_URI']= FLASK_APP_BD 
mongo = PyMongo(app)

cache = Cache(app, config={'CACHE_TYPE': 'simple'})
CORS(app)

mongo_uri = os.environ.get('MONGODB_URI')

if not mongo_uri:
    raise Exception('La cadena de conexión de MongoDB no está configurada')

client = MongoClient(mongo_uri)
db = client['datos_manuales']
collection = db['Sensor']
collection_manuales = db['manuales']


@cache.cached(timeout=60)  # Almacena en caché durante 60 segundos
def get_datos_sensor():
    # ... Lógica para obtener datos ...
    return get_datos_sensor


@app.route('/api/offline', methods=['GET'])
def offline_data():
    return cache.get('/api/all')

@app.route('/api/all', methods=['GET'])#Usado para extraer datos al frontend, no extrae todos los datos
@cache.cached(timeout=60)  # Almacena en caché durante 60 segundos
def dataall():
    dataall = []
    doc = mongo.db.data.find_one(sort=[('_id', -1)])
    if doc:
        doc_dict = json_util.loads(json_util.dumps(doc))
        dataall.append({
            '_id': str(doc_dict['_id']),
            'Rain': doc_dict['Rain '],
            'Temperature': doc_dict['Temperature '],
            'RH': doc_dict['RH '],
            'Dew_Point': doc_dict['Dew Point'],
            'Wind_Speed': doc_dict['Wind Speed '],
            'Gust_Speed': doc_dict['Gust Speed '],
            'Wind_Direction': doc_dict['Wind Direction '],
            'Date': doc_dict['Date']
        })
    return jsonify(dataall)

@app.route('/api/predict', methods=['POST'])#Realiza la prediccion y guarda los datos en una nueva coleccion
def dataarray():
    data = request.get_json()
    planta = float(data['PLANTA'])
    fruto = float(data['FRUTO'])
    severidad = float(data['SEVERIDAD'])
    dataarray = []   
    doc = mongo.db.data.find_one(sort=[('_id', -1)])
    if doc:
        doc_dict = json_util.loads(json_util.dumps(doc))
        dataarray.append(
            [
                doc_dict["Rain "],
                doc_dict["Temperature "],
                doc_dict["RH "],
                doc_dict["Dew Point"],
                doc_dict["Wind Speed "],
                doc_dict["Gust Speed "],
                doc_dict["Wind Direction "],
                planta,
                fruto,
                severidad,
            ]
        )
        model = joblib.load("D:/prediccion_cacao/Backend/models/Abeldb.pkl")
        X_test = np.array(dataarray)
        prediction = model.predict(X_test).tolist()
        datainsert = dataarray[0] + [prediction[0]]
        mongo.db.Predictions.insert_one({
            'Rain': datainsert[0],
            'Temperature': datainsert[1],
            'RH': datainsert[2],
            'Dew_Point': datainsert[3],
            'Wind_Speed': datainsert[4],
            'Gust_Speed': datainsert[5],
            'Wind_Direction':datainsert[6],
            'planta': datainsert[7],
            'fruto': datainsert[8],
            'severidad': datainsert[9],
            'incidencia': datainsert[10],
            
        })
    #return jsonify(prediction)
    if (prediction[0] == 1):
        return jsonify('Cultivo infectado')
    else:
        return jsonify('Cultivo sano')
    
@app.route('/api/predecir', methods=['POST'])#Realiza la prediccion y guarda los datos en una nueva coleccion
def dataarray():
    data = request.get_json()
    planta = float(data['PLANTA'])
    fruto = float(data['FRUTO'])
    severidad = float(data['SEVERIDAD'])
    dataarray = []   
    doc = collection.db.Sensor.find_one(sort=[('_id', -1)])
    if doc:
        doc_dict = json_util.loads(json_util.dumps(doc))
        dataarray.append(
            [
                doc_dict["Rain"],
                doc_dict["Temperature"],
                doc_dict["RH"],
                doc_dict["DewPoint"],
                doc_dict["WindSpeed"],
                doc_dict["GustSpeed"],
                doc_dict["WindDirection"],
                planta,
                fruto,
                severidad,
            ]
        )
        model = joblib.load("C:\ESPOCH\Mineria_Proyecto\scikit-learn\app\models\Modelo0.15.pkl")
        X_test = np.array(dataarray)
        prediction = model.predict(X_test).tolist()
        
    #return jsonify(prediction)
    if (prediction[0] == 1):
        return jsonify('Cultivo infectado')
    else:
        return jsonify('Cultivo sano')


@app.route('/api/allp', methods=['GET'])#extrae todos los datos del sensor
def dataallp():
    dataall = []
    doc = mongo.db.data.find_one(sort=[('_id', -1)])
    if doc:
        doc_dict = json_util.loads(json_util.dumps(doc))
        dataall.append({
            '_id': str(doc_dict['_id']),
            'Rain': doc_dict['Rain '],
            'Temperature': doc_dict['Temperature '],
            'RH': doc_dict['RH '],
            'Dew_Point': doc_dict['Dew Point'],
            'Wind_Speed': doc_dict['Wind Speed '],
            'Gust_Speed': doc_dict['Gust Speed '],
            'Wind_Direction': doc_dict['Wind Direction '],
            'planta': doc_dict['PLANTA'],
            'fruto': doc_dict['FRUTO'],
            'incidencia': doc_dict['INCIDENCIA'],
            'severidad': doc_dict['SEVERIDAD'],
            'Date': doc_dict['Date']
        })
    return jsonify(dataall)

@app.route('/api/allpredict', methods=['GET']) #extrae todos los datos incluido la prediccion
def allpredict():
    predictions = mongo.db.Predictions.find()
    prediction_list = []
    
    for prediction in predictions:
        prediction_data = {
            'Rain': prediction['Rain'],
            'Temperature': prediction['Temperature'],
            'RH': prediction['RH'],
            'Dew_Point': prediction['Dew_Point'],
            'Wind_Speed': prediction['Wind_Speed'],
            'Gust_Speed': prediction['Gust_Speed'],
            'Wind_Direction': prediction['Wind_Direction'],
            'planta': prediction['planta'],
            'fruto': prediction['fruto'],
            'severidad': prediction['severidad'],
            'incidencia': prediction['incidencia']
        }
        prediction_list.append(prediction_data)
    
    return jsonify(prediction_list)

@app.route('/api/dat/<id>', methods=['PUT']) ##actualizado no necesario
def updatedata(id):
    data = request.get_json()
    planta = float(data['PLANTA'])
    fruto = float(data['FRUTO'])
    incidencia = float(data['INCIDENCIA'])
    severidad = float(data['SEVERIDAD'])
    mongo.db.data.update_one({'_id':ObjectId(id)}, {'$set':{
            'PLANTA':planta,
            'FRUTO':fruto,
            'SEVERIDAD':severidad,
            'INCIDENCIA':incidencia       
    }})
    return jsonify({'msg':'Datos actualizados'})

@app.route('/api/dat', methods=['POST']) #Inserta datos en la nueva tabla
def insertpredicts():
    data = request.get_json()
    planta = float(data['PLANTA'])
    fruto = float(data['FRUTO'])
    incidencia = float(data['INCIDENCIA'])
    severidad = float(data['SEVERIDAD'])
    mongo.db.Predictions.insert_one({
        'PLANTA': planta,
        'FRUTO': fruto,
        'SEVERIDAD': severidad,
        'INCIDENCIA': incidencia      
    })
    return jsonify({'msg':'Prediccion guardada'})



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000, debug=True)
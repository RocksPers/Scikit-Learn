from pymongo import MongoClient
import certifi
ca = certifi.where()

def dbConnection():
    try:
        client = MongoClient( tlsCAFile=ca)
        db = client["datos_manuales"]
    except ConnectionError:
        print('Error de conexión con la BD')
    return db

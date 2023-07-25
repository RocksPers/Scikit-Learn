class Datos_manuales:
    def __init__(self, id, temperatura, humedad, luminosidad, ph_suelo, nivel_agua, altitud):
        self.id = id
        self.temperatura = temperatura
        self.humedad = humedad
        self.luminosidad = luminosidad
        self.ph_suelo = ph_suelo
        self.nivel_agua = nivel_agua
        self.altitud = altitud
    
    def ColleccionDatos(self):
        return {
            'id': self.id,
            'Temperatura': self.temperatura,
            'Humedad': self.humedad,
            'Luminosidad': self.luminosidad,
            'Ph del suelo': self.ph_suelo,
            'Nivel del agua': self.nivel_agua,
            'Altitud': self.altitud,
        }

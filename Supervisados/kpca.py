import pandas as pd #importamos pandas
import sklearn #biblioteca de aprendizaje automático
import matplotlib.pyplot as plt #Librería especializada en la creación de gráficos
from sklearn.decomposition import KernelPCA #importamos algorimo PCA
from sklearn.decomposition import IncrementalPCA #importamos algorimo PCA
from sklearn.linear_model import LogisticRegression #clasificación y análisis predictivo 
from sklearn.preprocessing import StandardScaler #Normalizar los datos
from sklearn.model_selection import train_test_split #permite hacer una división de un conjunto de datos en dos bloques de entrenamiento y prueba de un modelo

if __name__ == "__main__":
    # Cargamos los datos del dataframe de pandas
    dt_heart=pd.read_csv('./data/Sensor.csv')
    # Imprimimos un encabezado con los primeros 5 registros
    #print(dt_heart.head(5))
    # Guardamos nuestro dataset sin la columna de target
    dt_features = dt_heart.drop(['INCIDENCIA'], axis=1)
    # Este será nuestro dataset, pero sin la columna
    dt_target = dt_heart['INCIDENCIA']
    # Normalizamos los datos
    dt_features = StandardScaler().fit_transform(dt_features)
    # Partimos el conjunto de entrenamiento y para añadir replicabilidad usamos el random state
    X_train, X_test, y_train, y_test = train_test_split(dt_features, dt_target, test_size=0.3, random_state=42)
    kernel = ['linear','poly','rbf']
    #Aplicamos la función de kernel de tipo polinomial
    for k in kernel:
        kpca = KernelPCA(n_components=3, kernel = k)
        #kpca = KernelPCA(n_components=4, kernel='poly' )
        #Vamos a ajustar los datos
        kpca.fit(X_train)
        #Aplicamos el algoritmo a nuestros datos de prueba y de entrenamiento
        dt_train = kpca.transform(X_train)
        dt_test = kpca.transform(X_test)
        #Aplicamos la regresión logística un vez que reducimos su dimensionalidad
        logistic = LogisticRegression(solver='lbfgs')
        #Entrenamos los datos
        logistic.fit(dt_train, y_train)
        #Imprimimos los resultados
        print("SCORE KPCA " + k + " : ", logistic.score(dt_test, y_test))
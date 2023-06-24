import pandas as pd #importamos pandas
import sklearn #biblioteca de aprendizaje automático
import matplotlib.pyplot as plt #Librería especializada en la creación de gráficos
from sklearn.decomposition import KernelPCA #importamos algorimo PCA
from sklearn.decomposition import PCA #importamos algorimo PCA
from sklearn.decomposition import IncrementalPCA #importamos algorimo PCA
from sklearn.linear_model import LogisticRegression #clasificación y análisis predictivo 
from sklearn.preprocessing import StandardScaler #Normalizar los datos
from sklearn.model_selection import train_test_split #permite hacer una división de un conjunto de datos en dos 
from sklearn.preprocessing import KBinsDiscretizer

#bloques de entrenamiento y prueba de un modelo
if __name__ == '__main__':
    dt_heart=pd.read_csv('./data/gibhutdatos.csv')
    print(dt_heart.head(10)) #imprimimos los 5 primeros datos
    dt_features=dt_heart.drop(['PorcentajeToxicos','Toxicos'], axis=1) #las featurus sin el target
    dt_target = dt_heart['PorcentajeToxicos'] #obtenemos el target
    dt_target = dt_heart['Toxicos'] 

    #Normalizados
    #dt_features = StandardScaler().fit_transform(dt_features) #Normalizamnos los datos
    
    #Discretizados
    #discretizer = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="uniform")
    #dt_features = discretizer.fit_transform(dt_features)

    
    X_train,X_test,y_train,y_test =train_test_split(dt_features,dt_target,test_size=0.30,random_state=42)
    print(X_train.shape) #consultar la forma de la tabla con pandas
    print(y_train.shape)
    #print('DATOS NORMALIZADOS')
    #print('DATOS DISCRETIZADOS')
    print('DATOS SIN NORMALIZAR Y DISCRETIZAR')
    kernel = ['linear', 'poly', 'rbf']
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
        logistic = LogisticRegression(solver='lbfgs', max_iter=1000)  # Incorporar max_iter=1000
        logistic.fit(dt_train, y_train)
        #Entrenamos los datos
        logistic.fit(dt_train, y_train)
        #Imprimimos los resultados
        print("SCORE KPCA " + k + " : ", logistic.score(dt_test, y_test))
    '''EL número de componentes es opcional, ya que por defecto si no le pasamos el número de componentes lo asignará 
    de esta forma:
    a: n_components = min(n_muestras, n_features)'''
    pca=PCA(n_components=3)
    # Esto para que nuestro PCA se ajuste a los datos de entrenamiento que tenemos como tal
    pca.fit(X_train)
    #Como haremos una comparación con incremental PCA, haremos lo mismo para el IPCA
    '''EL parámetro batch se usa para crear pequeños bloques, de esta forma podemos ir entrenandolos
    poco a poco y combinarlos en el resultado final'''
    ipca=IncrementalPCA(n_components=3,batch_size=10) #tamaño de bloques, no manda a entrear todos los datos
    #Esto para que nuestro PCA se ajuste a los datos de entrenamiento que tenemos como tal
    ipca.fit(X_train)
    ''' Aquí graficamos los números de 0 hasta la longitud de los componentes que me sugirió el PCA o que
    me generó automáticamente el pca en el eje x, contra en el eje y, el valor de la importancia
    en cada uno de estos componentes, así podremos identificar cuáles son realmente importantes
    para nuestro modelo '''
    plt.plot(range(len(pca.explained_variance_)),pca.explained_variance_ratio_) #gneera desde 0 hasta los componentes
    plt.show()
    #Ahora vamos a configurar nuestra regresión logística
    logistic=LogisticRegression(solver='lbfgs')
    # Configuramos los datos de entrenamiento
    dt_train = pca.transform(X_train)#conjunto de entrenamiento
    dt_test = pca.transform(X_test)#conjunto de prueba
    # Mandamos los data frames la la regresión logística
    logistic.fit(dt_train, y_train) #mandasmos a regresion logistica los dos datasets
    #Calculamos nuestra exactitud de nuestra predicción
    print("SCORE PCA: ", logistic.score(dt_test, y_test))
    #Configuramos los datos de entrenamiento
    dt_train = ipca.transform(X_train)
    dt_test = ipca.transform(X_test)
    # Mandamos los data frames la la regresión logística
    logistic.fit(dt_train, y_train)
    #Calculamos nuestra exactitud de nuestra predicción
    print("SCORE IPCA: ", logistic.score(dt_test, y_test))
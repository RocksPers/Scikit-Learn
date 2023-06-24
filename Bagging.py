import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")
if __name__ == '__main__':
    dt_heart = pd.read_csv('./data/Sensor.csv')
    #print(dt_heart['INCIDENCIA'].describe())

    x = dt_heart.drop(['INCIDENCIA'], axis=1)
    y = dt_heart['INCIDENCIA']

    print("No Normalizados")
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=1)
    
    '''''
    knn_class = KNeighborsClassifier().fit(X_train, y_train)
    knn_prediction = knn_class.predict(X_test)
    print('='*64)
    print('SCORE con KNN: ', accuracy_score(knn_prediction, y_test))'''
    bag_class = BaggingClassifier(base_estimator=KNeighborsClassifier(), n_estimators=50).fit(X_train, y_train) 
    # base_estimator pide el 
    #estimador en el que va a estar basado nuestro metodo || n_estimators 
    #nos pide cuantos de estos modelos vamos a utilizar
    bag_pred = bag_class.predict(X_test)
    print('='*64)
    print(accuracy_score(bag_pred, y_test))
    print(X_train.shape) #consultar la forma de la tabla con pandas
    print(y_train.shape)

    print("Normalización")
    dt_features = StandardScaler().fit_transform(x) #Normalizamnos los datos

    X_train,X_test,y_train,y_test =train_test_split(dt_features,y,test_size=0.30,random_state=42)
    bag_class = BaggingClassifier(base_estimator=KNeighborsClassifier(), n_estimators=50).fit(X_train, y_train)
    bag_pred = bag_class.predict(X_test)
    print('='*64)
    print(accuracy_score(bag_pred, y_test))
    print(X_train.shape) #consultar la forma de la tabla con pandas
    print(y_train.shape)

    

    
    '''''
    estimators = {
        'LogisticRegression' : LogisticRegression(),
        'SVC' : SVC(),
        'LinearSVC' : LinearSVC(),
        'SGD' : SGDClassifier(loss="hinge", penalty="l2", 
        max_iter=5),
        'KNN' : KNeighborsClassifier(),
        'DecisionTreeClf' : DecisionTreeClassifier(),
        'RandomTreeForest' : RandomForestClassifier(random_state=0)
    }
    for name, estimator in estimators.items():
        bag_class = BaggingClassifier(base_estimator=estimator, 
        n_estimators=50).fit(X_train, y_train)
        bag_predict = bag_class.predict(X_test)
        print('='*64)
        print('SCORE Bagging with {} : {}'.format(name, 
        accuracy_score(bag_predict, y_test)))'''''
    
    


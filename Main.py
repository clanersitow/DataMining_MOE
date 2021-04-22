#### Librerias a utilizar #####
import numpy as np 
import pandas as pd
from sklearn import datasets,linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split #separa data
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge

# Import libraries for graphs
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

filename="polynomial.txt"




def main():
    ##PRIMERO

    print("######### REGRESION MULTISALIDA #########")

    #Read data
    X=np.loadtxt('fileEnd_X.pos')
    y=np.loadtxt("fileEnd_Y.pof")


    #Split on training set and test set
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


    #Training model
    clf=MultiOutputRegressor(Ridge(random_state=123)).fit(X_train, y_train)

    #Does prediction
    Y_pred = clf.predict(X_test) #Values of test set
    print(clf.score(X_train,y_train))

    # Creating dataset predition test
    z = Y_pred[:,0]
    x = Y_pred[:,1]
    y = Y_pred[:,2]

    # Creating dataset test test
    z1 = y_test[:,0]
    x1 = y_test[:,1]
    y1 = y_test[:,2]

 
    # Creating figure
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
 
    # Creating plot
    ax.scatter3D(x, y, z, color = "blue", alpha=0.1)

    ax.scatter3D(x1, y1, z1, color = "red")
    plt.title("Results predict Y and Test set")

 
    # show plot
    plt.show()

    ##SEGUNDO
    

    print("######### REGRESION POLINOMIAL #########")
    ##Preparar la data
    data = pd.read_csv(filename,sep="\t",header=0)

    # Entendimiento de la data
    #print('Informacion del data set')
    #print(data.shape)
    #print(data.head(78))
    #print(data.columns)


    #### PREPARAR DATA PARA REGRESION POLINOMIAL ###

    #Defino entradas X Solamente la columna 6
    X_p = data['age']

    #Defino Y
    y_p = data['length']


    
    #Defino el algoritmo a usar
    pr = linear_model.LinearRegression()
    
    #Definir grado del polinomio
    print("Ingrese el valor para degree")
    input_degree = input()
    deg = int(input_degree)
    poli_reg = PolynomialFeatures(degree = deg)
    
    precision = 0
    data = data.values
    k_iterations = 100
    n_size = len(data)
    print("n_size", n_size)
    

    #Comienza bootstraping
    for i in range(k_iterations):
        train = resample(data,n_samples = n_size)
        test = np.array([x for x in data if x.tolist() not in train.tolist()])
        
        X_train = train[:,0].reshape(-1,1) #0 seria age 1 seria length
        y_train = train[:,1].reshape(-1,1)
        X_test = test[:,0].reshape(-1,1)
        y_test = test[:,1].reshape(-1,1)
         
        X_train_poli = poli_reg.fit_transform(X_train)
        X_test_poli = poli_reg.fit_transform(X_test)
        
        #entrenar
        pr.fit(X_train_poli,y_train)

        #precision
        Y_pred_pr = pr.predict(X_test_poli)
    
        #print("Datos reales")
        #print(y_test)

        #print("Datos obtenidos")
        #print(Y_pred_pr)

        
        #Calculo precision cada iteracion de bootstraping
        precision += pr.score(X_train_poli, y_train)
        
    plt.scatter(X_p,y_p) #Data set completo
    plt.scatter(X_test,Y_pred_pr,color="red",linewidth=3)
    plt.show()


    print("Precision")
    print(precision/k_iterations)

    mse = mean_squared_error(y_test,Y_pred_pr)
    print("MSE ", mse)
      
    




main()
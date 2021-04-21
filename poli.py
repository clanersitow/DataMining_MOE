#### Librerias a utilizar #####
import numpy as np 
import pandas as pd
from sklearn import datasets,linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split #separa data
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import resample

filename="polynomial.txt"

def main():
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
    print("Input degree")
    input_degree = input()
    deg = int(input_degree)
    poli_reg = PolynomialFeatures(degree = deg)
    
    precision = 0
    data = data.values
    k_iterations = 10
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
      
    




main()
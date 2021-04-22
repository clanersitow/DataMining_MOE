import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


# Import libraries for graphs
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

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

#MSE
mse = mean_squared_error(y_test,Y_pred)
print("Error cuadratico medio",mse)

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
ax.set_xlabel("y1")
ax.set_ylabel("y2")
ax.set_zlabel("y3")
plt.title("Results predict and Test set")


 
# show plot
plt.show()
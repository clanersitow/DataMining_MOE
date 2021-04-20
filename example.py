import numpy as np
import pandas as pd
from sklearn.datasets import load_linnerud
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge


X,y = load_linnerud(return_X_y=True)
clf=MultiOutputRegressor(Ridge(random_state=123)).fit(X, y)
y_predic = clf.predict(X[[0]])

x=np.loadtxt('fileEnd_X.pos')
y=np.loadtxt("fileEnd_Y.pof")


clf=MultiOutputRegressor(Ridge(random_state=123)).fit(x, y)


print(y.shape)
print(x.shape)

y_predic2 = clf.predict(x)
print(y_predic2)

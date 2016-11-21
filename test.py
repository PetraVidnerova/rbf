import numpy as np
import matplotlib.pyplot as plot 
from rbf import HiddenLayer, RBFNet 


X = np.linspace(0, 7)
X = X.reshape(len(X), 1)
Y = np.sin(X) 

rbf = RBFNet(10) 

rbf.fit(X,Y) 

Yp = rbf.predict(X) 

plot.plot(Y) 
plot.plot(Yp)
plot.show()


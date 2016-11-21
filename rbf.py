import numpy as np
import scipy.linalg as linalg 
import theano
import theano.tensor as T 

def _minus(x, c):
    x = x.dimshuffle(0, 1, 'x') # add dimension
    x = T.addbroadcast(x, 2)    # make the new dimension broadcastable 
    return (x - c.T).T  

def _gaussian_kernel(x, c, gamma=0.1):
    K = _minus(x, c)
    return T.exp( -gamma * (K*K).sum(axis=1).T)


class HiddenLayer():

    def __init__(self, h):
        self.h = h 

        x = T.dmatrix() 
        c = T.dmatrix() 
        self.design_matrix = theano.function([x, c], _gaussian_kernel(x, c))

    def fit(self, X):
        self.c = self._random_centers(X)
    
    def transform(self, X):
        return self.design_matrix(X, self.c)
    
    def _random_centers(self, X):
        random_indices = np.random.choice(range(len(X)), self.h, replace=False)
        return X[random_indices]



class RBFNet():

    def __init__(self, h):
        self.h = h
    
    def fit(self, X, Y):
        return _fit_3step(X, Y) 


    def _fit_gradient(self, X, Y):
        pass

    def _fit_3step(self, X, Y):
        self.hl = HiddenLayer(self.h) 
        self.hl.fit(X) 

        D = self.hl.transform(X) 
        
        # compute w 
        print(D.shape)
        print(Y.shape)
        self.w_ = np.dot(linalg.pinv2(D), Y) 
        return self

    def predict(self, X):
        D = self.hl.transform(X) 
        return np.dot(D, self.w_)

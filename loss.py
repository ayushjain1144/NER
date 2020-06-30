import numpy as np


#stable softmax(X) = exp(Xi + logc)/(sigma(Xij + logC))
def softmax(X):
    exps = np.exp(X - np.max(X, axis=0))
    return exps / np.sum(exps, axis=0) 

# loss = sigma(y * log (softmax(X)))
def cross_entropy_loss(X, y, lamda = 0):
    #y_hat = softmax(X)
    m = y.shape[1]
    return - (1 / m) * np.sum(np.multiply(y, np.log(X)) + np.multiply(1 - y, np.log(1 - X)))

def cross_entropy_loss_grad(X, y):
    pass

    

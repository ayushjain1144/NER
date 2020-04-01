import numpy as np


#stable softmax(X) = exp(Xi + logc)/(sigma(Xij + logC))
def softmax(X):
    exps = np.exp(X - np.max(X))
    return exps / np.sum(exps) 

# loss = sigma(y * log (softmax(X)))
def cross_entropy_loss(X, y, lamda = 0):
    y_hat = softmax(X)
    return np.sum(np.multiply(y, np.log(y_hat)))

def cross_entropy_loss_grad(X, y):
    pass

    
import numpy as np

#sigmoid activation

#sigmoid(x) = 1 / (1 + e^-x)
def sigmoid(X):
    return 1.0 / (1.0 + np.exp(-X))

# sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
def sigmoid_backward(dA, X):
    s = sigmoid(X)
    dS = s * (1.0 - s)
    return dA * dS

#tanh activation

# tanh(x) = e^x - e^-x / e^x + e^-x
def tanh(X):
    X_exp = np.exp(X)
    X_neg_exp = np.exp(-X)
    return (X_exp - X_neg_exp) / (X_exp + X_neg_exp)

# tanh'(x) = 1 - tanh(x)^2
def tanh_backward(dA, X):
    tan = tanh(X)
    dS = 1.0 - (tan * tan)
    return dS * dA

# Relu

# relu(X) = max(0, X)

def relu(X):
    return np.maximum(0, X)

# relu'(X) = 0 for X <= 0, 1 for X > 0

def relu_backward(dA, X):
    dX = np.zeros_like(X)
    dX[X>0] = 1
    return dA * dX


#leaky Relu

# leaky_relu(X) = 1 for X > 0, else 0.1
def leaky_relu(X, alpha = 0.01):
    lrelu = np.copy(X)
    lrelu[X<=0] *= alpha
    return lrelu

#leaky_relu'(X) = 1 for X > 0, else alpha
def leaky_relu_backward(dA, X, alpha = 0.01):
    dX = np.ones_like(X)
    dX[X<=0] = alpha
    return dX * dA

# def main():
#     A = -np.random.rand(5, 1)
#     print(f"Array: {A}")
#     print(f"Sigmoid: {sigmoid(A)}, Sigmoid_grad = {sigmoid_grad(A)}")
#     print(f"Tanh: {tanh(A)}, tanh_grad = {tanh_grad(A)}")
#     print(f"Relu: {relu(A)}, relu_grad = {relu_grad(A)}")
#     print(f"Leaky relu: {leaky_relu(A)}, leaky_relu_grad = {leaky_relu_grad(A)}")
    



# if __name__ == "__main__":
#     main()

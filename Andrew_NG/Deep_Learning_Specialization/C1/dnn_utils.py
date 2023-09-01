import numpy as np


def initialize_params(N:list):
    np.random.seed(1)
    parameters = {}
    for i in range(1, len(N)):
        parameters[f'W{i}'] = np.random.randn(N[i], N[i-1]) / np.sqrt(N[i-1])
        parameters[f'b{i}'] = np.zeros((N[i], 1))
    return parameters


### ------- Forward Propagation ------- ###
def linear_forward(A, W, b):
    return W.dot(A) + b, (A, W, b)


def linear_activation_forward(A_prev, W, b, activation):
    Z, linear_cache = linear_forward(A_prev, W, b)
    if activation == 'sigmoid':
        return 1/(1+np.exp(-Z)), (linear_cache, Z)
    elif activation == 'relu':
        return np.maximum(0, Z), (linear_cache, Z)
    elif activation == 'lrelu':
        return np.maximum(0.01, Z), (linear_cache, Z)    
    elif activation == 'tanh':
        return np.exp(Z) - np.exp(-Z) / np.exp(Z) + np.exp(-Z), (linear_cache, Z)    
    else:  # if not specified it would be linear activation function
        return Z, (linear_cache, Z)


def L_model_forward(X, parameters):
    A = X
    size = len(parameters)//2
    caches = []
    for layer in range(1, size):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters[f'W{layer}'], parameters[f'b{layer}'], 'relu')
        caches.append(cache)
        
    A_Last, cache = linear_activation_forward(A, parameters[f'W{size}'], parameters[f'b{size}'], 'sigmoid')
    caches.append(cache)
    return A_Last, caches


### ------- Compute Cost ------- ###
def compute_cost(A_Last, Y):
    return (-1/Y.shape[0]) * (np.dot(Y, np.log(A_Last.T)) + np.dot((1 - Y), np.log(1-A_Last.T))).squeeze()


### ------- Backward Propagation ------- ###
def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = (1/m) * np.dot(dZ, A_prev.T)
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    if activation == 'sigmoid':
        dZ = dA * (1/(1+np.exp(-cache[1]))) * (1 - 1/(1+np.exp(-cache[1])))
    elif activation == 'relu':
        dZ = np.where(cache[1] >= 0, dA, 0)
    elif activation == 'lrelu':
        dZ = np.where(cache[1] >= 0, dA, 0.01)
    elif activation == 'tanh':
        dZ = dA * 1 - np.power(((np.exp(cache[1]) - np.exp(-cache[1])) / (np.exp(cache[1]) + np.exp(-cache[1]))), 2)
    dA_prev, dW, db = linear_backward(dZ, cache[0])
    return dA_prev, dW, db


def L_model_backward(A_Last, Y, caches):
    grads = {}             # keep the gradients here
    L = len(caches)        # number of layers
    m = A_Last.shape[1]    # number of examples
    Y = Y.reshape(A_Last.shape)  # Y has same shape as A_Last
    grads[f'dA{L-1}'], grads[f'dW{L}'], grads[f'db{L}'] = linear_activation_backward(
        -(np.divide(Y, A_Last) - np.divide(1 - Y, 1 - A_Last)),
        caches[L-1], # last layer
        'sigmoid'
    )
    for l in reversed(range(L-1)):
        grads[f'dA{l}'], grads[f'dW{l+1}'], grads[f'db{l+1}'] = linear_activation_backward(
            grads[f'dA{l+1}'],
            caches[l], # last layer
            'relu'
        )
    return grads


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters[f'W{l+1}'] = parameters[f'W{l+1}'] - learning_rate * grads[f'dW{l+1}']
        parameters[f'b{l+1}'] = parameters[f'b{l+1}'] - learning_rate * grads[f'db{l+1}']
    return parameters
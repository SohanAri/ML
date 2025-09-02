# stochastic hmm, lets go.

import numpy as np


def initialize_parameters(n_features):
    """
    Initialize parameters (weights and bias) for the model.
    """
    # TODO: Initialize weights and bias
    weights = np.zeros(n_features)
    bias = 0
    return weights, bias


def compute_gradient(X, y, weights, bias):
    """
    Compute the gradient of the loss with respect to weights and bias.
    """
    # TODO: Compute the gradients for weights and bias

    gradient_w = (weights*X+bias - y)*(X)
    gradient_b = (weights*X+bias -y)
    return gradient_w, gradient_b

def loss(X,y,weights,bias):
    y_pred = weights*X + bias
    cost = (1/y.size)*(((y_pred-y)**2).sum()) 
    return cost


def stochastic_gradient_descent(X, y, learning_rate=0.01, epochs=200):
    """
    Perform stochastic gradient descent optimization.
    """
    n_samples, n_features = X.shape
    weights, bias = initialize_parameters(n_features)
    print("Initial weights and bias: ", weights, bias)

    for epoch in range(epochs):
        for i in range(n_samples):
            xi = X[i:i+1]  # Taking one sample
            yi = y[i]
            grad_w, grad_b = compute_gradient(xi, yi, weights, bias)

            # Update weights and bias
            if grad_w is not None and grad_b is not None:
                weights = weights - learning_rate * grad_w
                bias = bias - learning_rate * grad_b
            # else:
            #     # TODO: Define update rules using computed gradients
                
            #     pass
            #loss = 
        # Optionally log the progress of training
        test_x = 2* np.random.rand(50,1)
        test_y = 3 + 4*test_x + np.random.rand(50,1)

        cost1 = loss(X,y,weights,bias)
        cost2 = loss(test_x,test_y,weights,bias)
        print(f"Epoch {epoch+1}/{epochs}: Loss on train, test = ",cost1,cost2)

    return weights, bias


if __name__ == "__main__":
    # Example usage:
    # TODO: Load your data into X and y
    X = 2 * np.random.rand(250, 1)
    y = 3 + 4*X + np.random.rand(250, 1)
    xmean = np.mean(X)
    xstd = np.std(X)
    xnor = (X-xmean)/xstd

    #X = np.array([])  # Replace with your feature data
    #y = np.array([])  # Replace with your target data

    weights, bias = stochastic_gradient_descent(xnor, y)
    # TODO: Use the optimized parameters as needed
    print("Optimized weights:", weights)
    print("Optimized bias:", bias)

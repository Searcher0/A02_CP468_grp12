import numpy as np

class Perceptron(object):
    """ 
    Need 3 attributes for a single layer Perceptron, Learning rate (l_rate);
     number of iterations (n_iter); Weights (weight)
     As per instructions Learning rate is set to 0.1 & iterations are 1000.
    """
    def __init__(self, l_rate=0.1, n_iter=1000):
        self.l_rate = l_rate
        self.n_iter = n_iter

    """Following the formula, Using numpy to calculate the dot product of the Input & weight vectors"""
    def weightedSum(self, X):
        return np.dot(X, self.weight[1:]) + self.weight[0]
    
    def stepFunction(self, X):
        return np.where(self.weighted_sum(X) >= 0.0, 1, -1)

    def fit(self, X, y):
        # Initialize the weights randomly between -0.5 and 0.5
        self.weight = np.random.uniform(-0.5, 0.5, 1 + X.shape[1])
        self.errors_ = []
        print("Weights:", self.weight)

        # training the model n_iter times
        for _ in range(self.n_iter):
            error = 0

            # loop through each input
            for xi, y in zip(X, y):

                # 1. calculate ŷ (the predicted value)
                y_pred = self.predict(xi)

                # 2. calculate Update
                # update = η * (y - ŷ)
                update = self.eta * (y - y_pred)

                # 3. Update the weights
                # Wi = Wi + Δ(Wi)   where  Δ(Wi) = η * (y - ŷ) = update * Xi
                self.weight[1:] = self.weight[1:] + update * xi
                print("Updated Weights:", self.weight[1:])

                # Update the bias (Xo = 1)
                self.weight[0] = self.weight[0] + update

                # if update != 0, it means that ŷ != y
                error += int(update != 0.0)

            self.errors_.append(error)

        return self

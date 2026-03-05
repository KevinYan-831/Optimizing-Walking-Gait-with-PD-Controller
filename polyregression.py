import numpy as np



class Polynomial_Regression:
    # initialize the regression model with custom degrees and weights
    def __init__(self, degree, alpha, iterations):
        self.degree = degree
        self.weights = None
        self.alpha = alpha
        self.iterations = iterations
        self.mean = None
        self.std = None
    # input matrix M contain the input parameters for our testing trials, num of rows is number of trials, and columns are the input parameters
    # The input matrix should look like [[rot, lif, dur, kp, kd], ... ]

    def init_features_matrix(self, M, fit=False):
        M = np.asarray(M)
        if M.ndim == 1:
            M = M.reshape(1, -1)

        n_trial, n_params = M.shape
        features = [np.ones(n_trial)]
        # Use train-set normalization stats for all future transforms.
        if fit:
            self.mean = np.mean(M, axis=0)
            self.std = np.std(M, axis=0)
            self.std = np.where(self.std == 0, 1, self.std)
        elif self.mean is None or self.std is None:
            raise Exception("Model normalization stats not initialized. Train first.")

        M_norm = (M - self.mean) / self.std

        for i in range(1, self.degree + 1):
            for j in range(n_params):
                features.append(M_norm[:, j] ** i)
        
        return np.column_stack(features)
    
    # y is the measurement of trials, for instance, the distance traveled, or the change of heading
    def gradient_descent(self, M, y):
        y = np.asarray(y)
        M_norm = self.init_features_matrix(M, fit=True)
        # initialize coefficients array
        self.weights = np.zeros(M_norm.shape[1])
        # begin the training loop
        print(f"Start Training: Learning Rate = {self.alpha} and Iterations = {self.iterations}\n")
        for i in range(self.iterations):
            print(f"== Iteration {i} ==")
            # prediction based on the given weights and normalized parameters
            y_pred = np.dot(M_norm, self.weights)
            error = y_pred - y
            # Calculate the gradient
            gradient = (1 / M_norm.shape[0]) * np.dot(M_norm.T, error)
            # update the weight of the model
            self.weights = self.weights - (self.alpha * gradient)

            # print the error every 100 iterations
            if i % 100 == 0:
                print(f"Iteration {i}, Cost = {self.cost_function(M_norm, error)}")
            
    # Use updated weights to make prediction based on the input parameters
    def predict(self, params):
        if self.weights is None:
            raise Exception("Model not trained yet.")
        M_norm = self.init_features_matrix(params)
        return np.dot(M_norm, self.weights)

    # Evaluate a validation/test split using common regression metrics.
    def evaluate(self, M, y):
        y = np.asarray(y)
        y_pred = self.predict(M)
        error = y_pred - y
        mse = np.mean(error ** 2)
        mae = np.mean(np.abs(error))
        ss_res = np.sum(error ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        return {"mse": mse, "mae": mae, "r2": r2}
    
    
    # calculate the MSE cost function, gradient descent is to minimize this value
    def cost_function(self, M, error):
        return (1 / (2 * M.shape[0])) * np.sum(error ** 2)

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
        self.loss_history = []
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
        self.loss_history = []
        # begin the training loop
        print(f"Start Training: Learning Rate = {self.alpha} and Iterations = {self.iterations}\n")
        for i in range(self.iterations):
            print(f"== Iteration {i} ==")
            # prediction based on the given weights and normalized parameters
            y_pred = np.dot(M_norm, self.weights)
            error = y_pred - y
            cost = self.cost_function(M_norm, error)
            self.loss_history.append(cost)
            # Calculate the gradient
            gradient = (1 / M_norm.shape[0]) * np.dot(M_norm.T, error)
            # update the weight of the model
            self.weights = self.weights - (self.alpha * gradient)

            # print the error every 100 iterations
            if i % 100 == 0:
                print(f"Iteration {i}, Cost = {cost}")
            
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


def choose_best_degree(M_train, y_train, M_validate, y_validate, degree_candidates, alpha, iterations):
    M_train = np.asarray(M_train)
    y_train = np.asarray(y_train)
    M_validate = np.asarray(M_validate)
    y_validate = np.asarray(y_validate)

    if M_train.shape[0] != y_train.shape[0]:
        raise ValueError(f"TRAIN: Distance and heading data collection has different length ({M_train.shape[0]} vs {y_train.shape[0]}).")
    if M_validate.shape[0] != y_validate.shape[0]:
        raise ValueError("VALIDATION: Distance and heading data collection has different length.")
    if len(degree_candidates) == 0:
        raise ValueError("degree_candidates cannot be empty.")

    best_degree = None
    best_mse = float("inf")
    best_model = None
    degree_results = []

    for degree in degree_candidates:
        model = Polynomial_Regression(degree=degree, alpha=alpha, iterations=iterations)
        model.gradient_descent(M_train, y_train)
        metrics = model.evaluate(M_validate, y_validate)
        mse = metrics["mse"]
        degree_results.append({"degree": degree, "mse": mse, "mae": metrics["mae"], "r2": metrics["r2"]})

        if mse < best_mse:
            best_mse = mse
            best_degree = degree
            best_model = model

    return best_degree, best_model, degree_results

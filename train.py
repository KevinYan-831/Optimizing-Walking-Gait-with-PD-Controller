# create data for model input
import numpy as np
import math
import polyregression as pr

# Discrete candidate values for parameter generation.
ROT_VALUES = np.arange(20, 150, 1)              
LIF_VALUES = np.arange(20, 150, 1)              
DUR_VALUES = np.round(np.arange(0.1, 1.5, 0.1), 1) 
KP_VALUES = np.round(np.arange(0.00, 2, 0.01), 2)  
KD_VALUES = np.round(np.arange(0.00, 2, 0.01), 2)   

# Keep bounds for optimization/search helpers.
ROT_BOUNDS = (float(np.min(ROT_VALUES)), float(np.max(ROT_VALUES)))
LIF_BOUNDS = (float(np.min(LIF_VALUES)), float(np.max(LIF_VALUES)))
DUR_BOUNDS = (float(np.min(DUR_VALUES)), float(np.max(DUR_VALUES)))
KP_BOUNDS = (float(np.min(KP_VALUES)), float(np.max(KP_VALUES)))
KD_BOUNDS = (float(np.min(KD_VALUES)), float(np.max(KD_VALUES)))

# After we have the model trained to predict the distance and heading given input
# we create a reward function and then find the best model parameter that has the highest reward
def reward_function(params, pred_distance, pred_heading):
    # For simplicity, we define the reward as a weighted sum of distance and heading
    distance_weight = 0.8
    heading_weight = 1

    pred_d = float(pred_distance.predict(params).reshape(-1)[0])
    pred_h = float(pred_heading.predict(params).reshape(-1)[0])
    reward = (distance_weight * pred_d) - (heading_weight * abs(pred_h))
    return reward

# Generate random parameters within the bound, then find the best combindation of parameters that has the highest reward
def find_best_params(pred_distance, pred_heading, bounds, n_candidates=5000):

    # Generate random candidates within bounds
    candidates = []
    for _ in range(n_candidates):
        candidate = [np.random.uniform(b[0], b[1]) for b in bounds]
        candidates.append(candidate)

    # Calculate reward for every candidate
    rewards = []
    for candidate in candidates:
        r = reward_function(candidate, pred_distance, pred_heading)
        rewards.append(r)
        print(f"Candidate: {candidate}, Reward: {r}")
        
    # Find the one with the highest reward
    best_idx = np.argmax(rewards)
    return candidates[best_idx]

if __name__ == '__main__':
    M_train = np.array([
        [116.00, 147.00, 0.10, 0.70, 0.11],
        [71.00, 82.00, 1.30, 0.07, 0.83],
        [61.00, 53.00, 0.90, 0.74, 0.93],
        [136.00, 105.00, 0.10, 1.59, 1.75],
        [26.00, 122.00, 1.20, 0.16, 0.80],
        [69.00, 60.00, 0.10, 1.78, 1.40],
        [37.00, 149.00, 0.30, 0.75, 0.70],
        [112.00, 44.00, 0.30, 1.76, 0.25],
        [82.00, 27.00, 1.20, 1.29, 0.84],
        [25.00, 22.00, 0.20, 0.90, 0.12],
        [58.00, 51.00, 1.20, 0.38, 0.21],
        [50.00, 60.00, 1.30, 0.55, 1.80],
        [111.00, 63.00, 1.40, 1.83, 1.00],
        [39.00, 97.00, 0.40, 0.85, 1.34],
        [38.00, 54.00, 1.20, 0.01, 0.83],
        [111.00, 113.00, 1.40, 0.03, 1.60],
        [31.00, 69.00, 0.20, 1.77, 1.29],
        [112.00, 31.00, 0.80, 1.02, 0.43],
        [40.00, 66.00, 0.40, 0.82, 1.74],
        [23.00, 121.00, 1.20, 1.88, 0.25],
        [82.00, 71.00, 0.80, 1.41, 1.02],
        [136.00, 53.00, 1.10, 1.91, 0.70],
        [38.00, 140.00, 0.50, 0.03, 1.23],
        [89.00, 96.00, 1.00, 0.48, 1.26],
        [42.00, 144.00, 0.10, 1.74, 0.51],
        [22.00, 54.00, 0.20, 0.03, 0.12],
        [20.00, 110.00, 0.70, 0.15, 1.81],
        [40.00, 68.00, 0.60, 1.13, 0.73],
        [104.00, 124.00, 1.30, 0.02, 1.23],
        [140.00, 138.00, 0.10, 1.61, 1.61],
        [84.00, 65.00, 1.10, 0.86, 0.13],
        # [83.00, 131.00, 0.70, 0.56, 1.57], #start here (continue)
        # [143.00, 43.00, 0.80, 0.26, 1.69],
        # [93.00, 118.00, 0.60, 0.32, 0.95],
        # [45.00, 80.00, 0.50, 0.06, 1.74],
        # [56.00, 94.00, 1.40, 0.11, 0.36],
        # [22.00, 37.00, 0.80, 1.53, 1.60],
        # [118.00, 23.00, 1.40, 0.03, 1.26],
        # [55.00, 44.00, 0.50, 1.89, 1.09],
        # [138.00, 149.00, 0.40, 1.12, 0.59],
        # [99.00, 112.00, 0.10, 0.06, 1.48],
        # [47.00, 120.00, 0.60, 1.29, 1.29],
    ], dtype=float)

    M_validate = np.array([
        [36.00, 133.00, 1.20, 1.87, 1.62],
        [112.00, 99.00, 1.30, 0.37, 1.45],
        [120.00, 127.00, 1.00, 1.37, 0.11],
        [114.00, 25.00, 0.40, 0.99, 1.81],
        [122.00, 78.00, 0.80, 1.62, 1.72],
        [77.00, 21.00, 0.70, 1.74, 0.75],
        [132.00, 65.00, 0.80, 0.27, 1.50],
        [138.00, 47.00, 0.80, 0.25, 0.13],
        [129.00, 49.00, 1.20, 1.60, 0.20],
        [67.00, 92.00, 1.00, 0.34, 1.42],
        [58.00, 23.00, 0.60, 0.56, 0.57],
        [135.00, 61.00, 1.10, 0.61, 1.00],
    ], dtype=float)

    M_test = np.array([
        [131.00, 125.00, 1.00, 0.92, 1.69],
        [26.00, 30.00, 0.60, 0.63, 1.46],
        [133.00, 30.00, 0.80, 1.15, 0.01],
        [26.00, 66.00, 0.80, 1.49, 1.19],
        [125.00, 149.00, 0.40, 1.26, 1.36],
        [121.00, 50.00, 1.30, 0.15, 1.47],
    ], dtype=float)
    # validate successful generation of params
    print("==Training Params==")
    print(M_train)
    print("==Validating Params==")
    print(M_validate)
    print("==Testing Params==")
    print(M_test)

    #15 seconds for gait running
    y_distance_train = [3277, 363, 306, 3510, 100, 2877, 605, 1357, 154, 209, 204, 202, 421, 501, 143, 468, 773, 342, 502, 80, 516, 549, 391, 519, 1696, 410, 112, 325, 405, 2714, 399]
    #negative is clockwisde rotation
    y_heading_train = [3.21, 2.72, 2.30, 5.73, 2.39, 1.13, -0.08, -1.71, -0.86, -1.86, 1.36, 1.16, -0.76, 0.70, 1.72, 0.11, 0.08, -6.70, -0.73, 3.19, 0.69, -3.70, 3.12, 0.47, -1.49, 1.56, 5.22, 2.56, 1.93, 1.10, 0.10]
    y_distance_validate = []
    y_heading_validate = []
    y_distance_test = []
    y_heading_test = []


    # Create the model for both distance and heading
    model_distance = pr.Polynomial_Regression(degree=5, alpha=0.01, iterations=1000)
    model_heading = pr.Polynomial_Regression(degree=5, alpha=0.01, iterations=1000)

    #Training two models
    print("Training model for distance...")
    model_distance.gradient_descent(M_train, y_distance_train)
    print("Training model for heading...")
    model_heading.gradient_descent(M_train, y_heading_train)

    #Find the best parameters that has the highest reward
    bounds = np.array([ROT_BOUNDS, LIF_BOUNDS, DUR_BOUNDS, KP_BOUNDS, KD_BOUNDS])
    best_params = find_best_params(model_distance, model_heading, bounds)
    print(f"Best Parameters: {best_params}")
import sys
import time
import signal
import threading  # use threading to run PID control and tripod simultaneously
import ros_robot_controller_sdk as rrc
import sonar
import matplotlib.pyplot as plt
import numpy as np
import math
import polyregression as pr

# Display numpy floats in regular decimal form
np.set_printoptions(
    suppress=True,
    formatter={"float_kind": lambda x: f"{x:.2f}"},
    linewidth=180
)






print('''
**********************************************************
********CS/ME 301 Assignment Template*******
**********************************************************
----------------------------------------------------------
Usage:
    sudo python3 asn_template.py
----------------------------------------------------------
Tips:
    * Press Ctrl+C to close the program. If it fails,
      please try multiple times！
----------------------------------------------------------
''')

board = rrc.Board()
start = True


def Stop(signum, frame):
    global start
    start = False


signal.signal(signal.SIGINT, Stop)
s = sonar.Sonar()

# Right Front Leg
RF_INNER_ID = 7
RF_MIDDLE_ID = 8
RF_OUTER_ID = 9
RF_INNER_DEFAULT = 500
RF_MIDDLE_DEFAULT = 200
RF_OUTER_DEFAULT = 100

# Left Front Leg
LF_INNER_ID = 16
LF_MIDDLE_ID = 17
LF_OUTER_ID = 18
LF_INNER_DEFAULT = 500
LF_MIDDLE_DEFAULT = 800
LF_OUTER_DEFAULT = 900

# Right Middle Leg
RM_INNER_ID = 4
RM_MIDDLE_ID = 5
RM_OUTER_ID = 6
RM_INNER_DEFAULT = 500
RM_MIDDLE_DEFAULT = 200
RM_OUTER_DEFAULT = 100

# Left Middle Leg
LM_INNER_ID = 13
LM_MIDDLE_ID = 14
LM_OUTER_ID = 15
LM_INNER_DEFAULT = 500
LM_MIDDLE_DEFAULT = 800
LM_OUTER_DEFAULT = 900

# Right Back Leg
RB_INNER_ID = 1
RB_MIDDLE_ID = 2
RB_OUTER_ID = 3
RB_INNER_DEFAULT = 500
RB_MIDDLE_DEFAULT = 200
RB_OUTER_DEFAULT = 100

# Left Back Leg
LB_INNER_ID = 10
LB_MIDDLE_ID = 11
LB_OUTER_ID = 12
LB_INNER_DEFAULT = 500
LB_MIDDLE_DEFAULT = 800
LB_OUTER_DEFAULT = 900

# Platform's value
P_ID = 21
P_DEFAULT = 500
P_RIGHT = 130
P_LEFT = 870


#Reactive control, distance threshold, used to avoid hitting the wall. If exceed threshold, just immediately skip the current step and move on
DISTANCE_PLAN = 340
DISTANCE_BLOCK = 400

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




def set_all_default():
    board.bus_servo_set_position(1, [
        # Right Front
        [RF_INNER_ID, RF_INNER_DEFAULT],
        [RF_MIDDLE_ID, RF_MIDDLE_DEFAULT],
        [RF_OUTER_ID, RF_OUTER_DEFAULT],
        # Right Middle
        [RM_INNER_ID, RM_INNER_DEFAULT],
        [RM_MIDDLE_ID, RM_MIDDLE_DEFAULT],
        [RM_OUTER_ID, RM_OUTER_DEFAULT],
        # Right Back
        [RB_INNER_ID, RB_INNER_DEFAULT],
        [RB_MIDDLE_ID, RB_MIDDLE_DEFAULT],
        [RB_OUTER_ID, RB_OUTER_DEFAULT],
        # Left Front
        [LF_INNER_ID, LF_INNER_DEFAULT],
        [LF_MIDDLE_ID, LF_MIDDLE_DEFAULT],
        [LF_OUTER_ID, LF_OUTER_DEFAULT],
        # Left Middle
        [LM_INNER_ID, LM_INNER_DEFAULT],
        [LM_MIDDLE_ID, LM_MIDDLE_DEFAULT],
        [LM_OUTER_ID, LM_OUTER_DEFAULT],
        # Left Back
        [LB_INNER_ID, LB_INNER_DEFAULT],
        [LB_MIDDLE_ID, LB_MIDDLE_DEFAULT],
        [LB_OUTER_ID, LB_OUTER_DEFAULT],
        # Platform
        [P_ID, P_DEFAULT]
    ])
    time.sleep(1)
#helper function that return boolean to determin whether the robot is blocked. Return True if the robot is blocked
def is_blocked(distance):
    return distance <= DISTANCE_BLOCK
def platform_left(dur):
    duration = dur
    board.bus_servo_set_position(duration, [[P_ID, P_LEFT]])
def platform_right(dur):
    duration = dur
    board.bus_servo_set_position(duration, [[P_ID, P_RIGHT]])
def platform_default(dur):
    duration = dur
    board.bus_servo_set_position(duration, [[P_ID, P_DEFAULT]])
#for one cycle, it takes up 4 times (dur + pu)
def tripod(dur=0.3, pu=0.3, lif=100, rot=105):
    duration = dur
    pause = pu
    lift = lif
    rotation = rot

    t0 = time.perf_counter()
    print(f"Start of the Cycle Gait: {s.getdistance()}")

    board.bus_servo_set_position(duration, [
        [RB_MIDDLE_ID, RB_MIDDLE_DEFAULT - lift],
        [RF_MIDDLE_ID, RF_MIDDLE_DEFAULT - lift],
        [LM_MIDDLE_ID, LM_MIDDLE_DEFAULT + lift],
        [RB_INNER_ID, RB_INNER_DEFAULT - rotation],
        [RF_INNER_ID, RF_INNER_DEFAULT - rotation],
        [LM_INNER_ID, LM_INNER_DEFAULT + rotation ],
        [RM_INNER_ID, RM_INNER_DEFAULT + rotation],
        [LB_INNER_ID, LB_INNER_DEFAULT - rotation],
        [LF_INNER_ID, LF_INNER_DEFAULT - rotation]
    ])  # Initial lift of legs and rotation
    time.sleep(pause)


    board.bus_servo_set_position(duration, [
        [RB_MIDDLE_ID, RB_MIDDLE_DEFAULT],
        [RF_MIDDLE_ID, RF_MIDDLE_DEFAULT],
        [LM_MIDDLE_ID, LM_MIDDLE_DEFAULT]
    ])  # Putting legs back down
    time.sleep(pause)


    board.bus_servo_set_position(duration, [
        [RM_MIDDLE_ID, RM_MIDDLE_DEFAULT - lift],
        [LB_MIDDLE_ID, LB_MIDDLE_DEFAULT + lift],
        [LF_MIDDLE_ID, LF_MIDDLE_DEFAULT + lift],
        [RM_INNER_ID, RM_INNER_DEFAULT - rotation],
        [LB_INNER_ID, LB_INNER_DEFAULT + rotation],
        [LF_INNER_ID, LF_INNER_DEFAULT + rotation],
        [RB_INNER_ID, RB_INNER_DEFAULT + rotation],
        [RF_INNER_ID, RF_INNER_DEFAULT + rotation],
        [LM_INNER_ID, LM_INNER_DEFAULT - rotation]
    ])  # Lifting second set of legs and rotation
    time.sleep(pause)


    board.bus_servo_set_position(duration, [
        [RM_MIDDLE_ID, RM_MIDDLE_DEFAULT],
        [LB_MIDDLE_ID, LB_MIDDLE_DEFAULT],
        [LF_MIDDLE_ID, LF_MIDDLE_DEFAULT]
    ])  # Putting down
    time.sleep(pause)
    print(f"End of the Cycle Gait: {s.getdistance()}")

def turn_left(dur, pu, rot, lif):
    duration = dur
    pause = pu
    rotation = rot
    lift = lif
    board.bus_servo_set_position(duration, [
        [2, RB_MIDDLE_DEFAULT - lift],
        [8, RF_MIDDLE_DEFAULT - lift],
        [14, LM_MIDDLE_DEFAULT + lift],
        [1, RB_INNER_DEFAULT + rotation],
        [7, RF_INNER_DEFAULT + rotation],
        [13, LM_INNER_DEFAULT + rotation]
    ])
    time.sleep(pause)
    board.bus_servo_set_position(duration, [
        [2, RB_MIDDLE_DEFAULT],
        [8, RF_MIDDLE_DEFAULT],
        [14, LM_MIDDLE_DEFAULT]
    ])
    time.sleep(pause)
    board.bus_servo_set_position(duration, [
        [5, RM_MIDDLE_DEFAULT - lift],
        [11, LB_MIDDLE_DEFAULT + lift],
        [17, LF_MIDDLE_DEFAULT + lift],
        [1, RB_INNER_DEFAULT],
        [7, RF_INNER_DEFAULT],
        [13, LM_INNER_DEFAULT]
    ])
    time.sleep(pause)
    board.bus_servo_set_position(duration, [
        [5, RM_MIDDLE_DEFAULT],
        [11, LB_MIDDLE_DEFAULT],
        [17, LF_MIDDLE_DEFAULT]
    ])
def turn_right(dur, pu, rot, lif):
    duration = dur
    pause = pu
    rotation = rot
    lift = lif

    board.bus_servo_set_position(duration, [
        [5, RM_MIDDLE_DEFAULT - lift],
        [11, LB_MIDDLE_DEFAULT + lift],
        [17, LF_MIDDLE_DEFAULT + lift],
        [4, RM_INNER_DEFAULT - rotation],
        [10, LB_INNER_DEFAULT - rotation],
        [16, LF_INNER_DEFAULT - rotation]
    ])
    time.sleep(pause)

    board.bus_servo_set_position(duration, [
        [5, RM_MIDDLE_DEFAULT],
        [11, LB_MIDDLE_DEFAULT],
        [17, LF_MIDDLE_DEFAULT]
    ])
    time.sleep(pause)

    board.bus_servo_set_position(duration, [
        [2, RB_MIDDLE_DEFAULT - lift],
        [8, RF_MIDDLE_DEFAULT - lift],
        [14, LM_MIDDLE_DEFAULT + lift],
        [4, RM_INNER_DEFAULT],
        [10, LB_INNER_DEFAULT],
        [16, LF_INNER_DEFAULT]
    ])
    time.sleep(pause)

    board.bus_servo_set_position(duration, [
        [2, RB_MIDDLE_DEFAULT],
        [8, RF_MIDDLE_DEFAULT],
        [14, LM_MIDDLE_DEFAULT]
    ])
def turn_left_90():
    for i in range(4):
        turn_left(0.3, 0.3, 197, 100)
        time.sleep(0.3)
def turn_right_90():
    for i in range(4):
        turn_right(0.3, 0.3, 197, 100)
        time.sleep(0.3)
def turn_around_180():
    for i in range(8):
        turn_left(0.3, 0.3, 197, 100)
        time.sleep(0.3)




# Generate experiment params and split into train/validate/test sets.
def gen_params(n_trials, train_ratio=0.7, validate_ratio=0.2, test_ratio=0.1):
    if not np.isclose(train_ratio + validate_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + validate_ratio + test_ratio must equal 1.0")

    params_list = []
    for _ in range(n_trials):
        rot = np.random.choice(ROT_VALUES)
        lif = np.random.choice(LIF_VALUES)
        dur = np.random.choice(DUR_VALUES)
        kp = np.random.choice(KP_VALUES)
        kd = np.random.choice(KD_VALUES)
        params_list.append([rot, lif, dur, kp, kd])

    params = np.array(params_list)
    np.random.shuffle(params)

    train_size = int(n_trials * train_ratio)
    validate_size = int(n_trials * validate_ratio)
    # Keep remainder in test split so all trials are used.
    test_size = n_trials - train_size - validate_size

    train_params = params[:train_size]
    validate_params = params[train_size:train_size + validate_size]
    test_params = params[train_size + validate_size:train_size + validate_size + test_size]
    return train_params, validate_params, test_params

    


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
      


    


if __name__ == "__main__":
    set_all_default()
    
    start_time = time.time()

    # Generate params for train/validate/test split, total 60 trials.
    train_params, validate_params, test_params = gen_params(60)

    # create data for model input
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

    # # Validate models
    # distance_val_metrics = model_distance.evaluate(M_validate, y_distance_validate)
    # heading_val_metrics = model_heading.evaluate(M_validate, y_heading_validate)
    # print(f"Distance validation : {distance_val_metrics}")
    # print(f"Heading validation : {heading_val_metrics}")

    # # Final test metrics
    # distance_test_metrics = model_distance.evaluate(M_test, y_distance_test)
    # heading_test_metrics = model_heading.evaluate(M_test, y_heading_test)
    # print(f"Distance test : {distance_test_metrics}")
    # print(f"Heading test : {heading_test_metrics}")

    #Find the best parameters that has the highest reward
    bounds = np.array([ROT_BOUNDS, LIF_BOUNDS, DUR_BOUNDS, KP_BOUNDS, KD_BOUNDS])
    best_params = find_best_params(model_distance, model_heading, bounds)
    print(f"Best Parameters: {best_params}")



    




    

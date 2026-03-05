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

#Bounds for the parameters
ROT_BOUNDS = (0, 200)
LIF_BOUNDS = (0, 200)
DUR_BOUNDS = (0.1, 0.5)
KP_BOUNDS = (0, 1)
KD_BOUNDS = (0, 1)




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




# Generate experiment params and split them into test/validate sets.
# test/validate ratio = 0.8
def gen_params(n_trials):
    params_list = []
    for _ in range(n_trials):
        rot = np.random.uniform(ROT_BOUNDS[0], ROT_BOUNDS[1])
        lif = np.random.uniform(LIF_BOUNDS[0], LIF_BOUNDS[1])
        dur = np.random.uniform(DUR_BOUNDS[0], DUR_BOUNDS[1])
        kp = np.random.uniform(KP_BOUNDS[0], KP_BOUNDS[1])
        kd = np.random.uniform(KD_BOUNDS[0], KD_BOUNDS[1])
        params_list.append([rot, lif, dur, kp, kd])

    params = np.array(params_list)
    np.random.shuffle(params)

    # If test/validate = 0.8, test fraction is 0.8 / (1 + 0.8).
    test_size = int(n_trials * (0.8 / 1.8))
    test_params = params[:test_size]
    validate_params = params[test_size:]
    return test_params, validate_params

    


# After we have the model trained to predict the distance and heading given input
# we create a reward function and then find the best model parameter that has the highest reward
def reward_function(params, pred_distance, pred_heading):
    # For simplicity, we define the reward as a weighted sum of distance and heading
    distance_weight = 1.0
    heading_weight = 0.5

    reward = (distance_weight * pred_distance.predict(params)) - (heading_weight * abs(pred_heading.predict(params)))
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

    # Generate params for test/validate split, with test/validate = 0.8
    test_params, validate_params = gen_params(50)

    # create data for model input
    # M =
    # y_distance = 
    # y_heading = 


    # Create the model for both distance and heading
    # model_distance = pr.Polynomial_Regression(M, y_distance, degree=5, alpha=0.01, iterations=1000)
    # model_heading = pr.Polynomial_Regression(M, y_heading, degree=5, alpha=0.01, iterations=1000)

    # #Training two models
    # print("Training model for distance...")
    # model_distance.gradient_descent(M, y_distance)
    # print("Training model for heading...")
    # model_heading.gradient_descent(M, y_heading)

    # #Find the best parameters that has the highest reward
    # bounds = np.array([ROT_BOUNDS, LIF_BOUNDS, DUR_BOUNDS, KP_BOUNDS, KD_BOUNDS])
    # best_params = find_best_params(model_distance, model_heading, bounds)
    # print(f"Best Parameters: {best_params}")



    




    

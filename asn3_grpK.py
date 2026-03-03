import sys
import time
import signal
import threading  # use threading to run PID control and tripod simultaneously
import ros_robot_controller_sdk as rrc
import sonar
import matplotlib.pyplot as plt
import numpy as np
import math







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


class Polynomial_Regression:
    #initialize the regression model with custom degrees and weights
    def __init__(self, M, y, degree, alpha, iterations):
        self.M = M
        self.y = y
        self.degree = degree
        self.weights = None
        self.alpha = alpha
        self.iterations = iterations
    #input matrix M contain the input parameters for our testing trials, num of rows is number of trials, and columns are the input parameters
    #The input matrix should look like [[rot, lif, dur, kp, kd], ... ]
    def init_features_matrix(self):
        n_trial, n_params = self.M.shape
        features = [np.ones(n_trial)]
        #we need to normalize the input parameters, since they all use different scale
        mean = np.mean(self.M, axis=0)
        std = np.std(self.M, axis=0)
        M_norm = (self.M - mean) / std

        for i in range(1, self.degree + 1):
            for j in range(n_params):
                features.append(M_norm[:, j] ** i)
        
        return np.column_stack(features)
    #y is the measurement of trials, for instance, the distance traveled, or the change of heading
    def gradient_descent(self):
        M_norm = self.init_features_matrix(self.M)
        #initialize coefficients array
        self.weights = np.zeros(M_norm.shape[1])
        #begin the training loop
        print(f"Start Training: Learning Rate = {self.alpha} and Iterations = {self.iterations}\n")
        for i in range(self.iterations):
            print(f"== Iteration {i} ==\n")
            #prediction based on the given weights and normalized parameters
            y_pred = np.dot(M_norm, self.weights)
            error = y_pred - self.y
            #Calculate the gradient
            gradient = (1 / M_norm.shape[0]) * np.dot(M_norm.T, error)
            #update the weight of the model
            self.weights = self.weights - (self.alpha * gradient)

            #print the error every 100 iterations
            if i % 100 == 0:
                print(f"Iteration {i}, Cost = {self.cost_function(error)}")
            

    
    #calculate the MSE cost function, gradient descent is to minimize this value
    def cost_function(self, error):
        return (1 / (2 * self.M.shape[0])) * np.sum(error ** 2)








    
    


    



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




#for one trial, let the robot walks for 5 cycles
def run_one_trial(dur, pu, lif, rot, kp, kd):
    for i in range(5):
        tripod(dur, pu, lif, rot)




        


    


if __name__ == "__main__":
    set_all_default()
    
    start_time = time.time()


    




    


import sys
import time
import signal
import threading
import ros_robot_controller_sdk as rrc

board = rrc.Board()
start = True

# 关闭前处理(process before closing)
def Stop(signum, frame):
    global start
    start = False
    print('关闭中...')

signal.signal(signal.SIGINT, Stop)

# Right Front Leg
RF_INNER_ID = 7     
RF_MIDDLE_ID = 8     
RF_OUTER_ID = 9      
RF_INNER_DEFAULT = 500
RF_MIDDLE_DEFAULT = 400
RF_OUTER_DEFAULT = 300

# Left Front Leg
LF_INNER_ID = 16
LF_MIDDLE_ID = 17
LF_OUTER_ID = 18
LF_INNER_DEFAULT = 500
LF_MIDDLE_DEFAULT = 600
LF_OUTER_DEFAULT = 700

# Right Middle Leg 
RM_INNER_ID = 4
RM_MIDDLE_ID = 5
RM_OUTER_ID = 6
RM_INNER_DEFAULT = 500
RM_MIDDLE_DEFAULT = 400
RM_OUTER_DEFAULT = 300

# Left Middle Leg
LM_INNER_ID = 13
LM_MIDDLE_ID = 14
LM_OUTER_ID = 15
LM_INNER_DEFAULT = 500
LM_MIDDLE_DEFAULT = 600
LM_OUTER_DEFAULT = 700

# Right Back Leg
RB_INNER_ID = 1
RB_MIDDLE_ID = 2
RB_OUTER_ID = 3
RB_INNER_DEFAULT = 500
RB_MIDDLE_DEFAULT = 400
RB_OUTER_DEFAULT = 300

# Left Back Leg
LB_INNER_ID = 10
LB_MIDDLE_ID = 11
LB_OUTER_ID = 12
LB_INNER_DEFAULT = 500
LB_MIDDLE_DEFAULT = 600
LB_OUTER_DEFAULT = 700

#Set every servo to default position
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
    ])
    time.sleep(1)


#Front pair leg helper functions
def front_pair_swing_forward(duration, amount):
    board.bus_servo_set_position(duration, [
        [RF_INNER_ID, RF_INNER_DEFAULT - amount],  # RF forward (-)
        [LF_INNER_ID, LF_INNER_DEFAULT + amount]   # LF forward (+)
    ])


def front_pair_swing_backward(duration, amount):
    board.bus_servo_set_position(duration, [
        [RF_INNER_ID, RF_INNER_DEFAULT + amount],  # RF backward (+)
        [LF_INNER_ID, LF_INNER_DEFAULT - amount]   # LF backward (-)
    ])


def front_pair_swing_default(duration):
    board.bus_servo_set_position(duration, [
        [RF_INNER_ID, RF_INNER_DEFAULT],
        [LF_INNER_ID, LF_INNER_DEFAULT]
    ])


def front_pair_push_down(duration, amount):
    board.bus_servo_set_position(duration, [
        [RF_MIDDLE_ID, RF_MIDDLE_DEFAULT + amount],  # RF push down (+)
        [LF_MIDDLE_ID, LF_MIDDLE_DEFAULT - amount]   # LF push down (-)
    ])


def front_pair_lift_leg(duration, amount):
    board.bus_servo_set_position(duration, [
        [RF_MIDDLE_ID, RF_MIDDLE_DEFAULT - amount],  # RF lift (-)
        [LF_MIDDLE_ID, LF_MIDDLE_DEFAULT + amount]   # LF lift (+)
    ])


def front_pair_middle_default(duration):
    board.bus_servo_set_position(duration, [
        [RF_MIDDLE_ID, RF_MIDDLE_DEFAULT],
        [LF_MIDDLE_ID, LF_MIDDLE_DEFAULT]
    ])


def front_pair_extend_outward(duration, amount):
    board.bus_servo_set_position(duration, [
        [RF_OUTER_ID, RF_OUTER_DEFAULT + amount],  # RF extend (+)
        [LF_OUTER_ID, LF_OUTER_DEFAULT - amount]   # LF extend (-)
    ])


def front_pair_retract_inward(duration, amount):
    board.bus_servo_set_position(duration, [
        [RF_OUTER_ID, RF_OUTER_DEFAULT - amount],  # RF retract (-)
        [LF_OUTER_ID, LF_OUTER_DEFAULT + amount]   # LF retract (+)
    ])


def front_pair_outer_default(duration):
    board.bus_servo_set_position(duration, [
        [RF_OUTER_ID, RF_OUTER_DEFAULT],
        [LF_OUTER_ID, LF_OUTER_DEFAULT]
    ])

# Middle pair leg helper functions
def middle_pair_swing_forward(duration, amount):
    board.bus_servo_set_position(duration, [
        [RM_INNER_ID, RM_INNER_DEFAULT - amount],  # RM forward (-)
        [LM_INNER_ID, LM_INNER_DEFAULT + amount]   # LM forward (+)
    ])


def middle_pair_swing_backward(duration, amount):
    board.bus_servo_set_position(duration, [
        [RM_INNER_ID, RM_INNER_DEFAULT + amount],  # RM backward (+)
        [LM_INNER_ID, LM_INNER_DEFAULT - amount]   # LM backward (-)
    ])


def middle_pair_swing_default(duration):
    board.bus_servo_set_position(duration, [
        [RM_INNER_ID, RM_INNER_DEFAULT],
        [LM_INNER_ID, LM_INNER_DEFAULT]
    ])


def middle_pair_push_down(duration, amount):
    board.bus_servo_set_position(duration, [
        [RM_MIDDLE_ID, RM_MIDDLE_DEFAULT + amount],  # RM push down (+)
        [LM_MIDDLE_ID, LM_MIDDLE_DEFAULT - amount]   # LM push down (-)
    ])


def middle_pair_lift_leg(duration, amount):
    board.bus_servo_set_position(duration, [
        [RM_MIDDLE_ID, RM_MIDDLE_DEFAULT - amount],  # RM lift (-)
        [LM_MIDDLE_ID, LM_MIDDLE_DEFAULT + amount]   # LM lift (+)
    ])


def middle_pair_middle_default(duration):
    board.bus_servo_set_position(duration, [
        [RM_MIDDLE_ID, RM_MIDDLE_DEFAULT],
        [LM_MIDDLE_ID, LM_MIDDLE_DEFAULT]
    ])


def middle_pair_extend_outward(duration, amount):
    board.bus_servo_set_position(duration, [
        [RM_OUTER_ID, RM_OUTER_DEFAULT + amount],  # RM extend (+)
        [LM_OUTER_ID, LM_OUTER_DEFAULT - amount]   # LM extend (-)
    ])


def middle_pair_retract_inward(duration, amount):
    board.bus_servo_set_position(duration, [
        [RM_OUTER_ID, RM_OUTER_DEFAULT - amount],  # RM retract (-)
        [LM_OUTER_ID, LM_OUTER_DEFAULT + amount]   # LM retract (+)
    ])


def middle_pair_outer_default(duration):
    board.bus_servo_set_position(duration, [
        [RM_OUTER_ID, RM_OUTER_DEFAULT],
        [LM_OUTER_ID, LM_OUTER_DEFAULT]
    ])

#Back pair leg helper functions
def back_pair_swing_forward(duration, amount):
    board.bus_servo_set_position(duration, [
        [RB_INNER_ID, RB_INNER_DEFAULT - amount],  # RB forward (-)
        [LB_INNER_ID, LB_INNER_DEFAULT + amount]   # LB forward (+)
    ])


def back_pair_swing_backward(duration, amount):
    board.bus_servo_set_position(duration, [
        [RB_INNER_ID, RB_INNER_DEFAULT + amount],  # RB backward (+)
        [LB_INNER_ID, LB_INNER_DEFAULT - amount]   # LB backward (-)
    ])


def back_pair_swing_default(duration):
    board.bus_servo_set_position(duration, [
        [RB_INNER_ID, RB_INNER_DEFAULT],
        [LB_INNER_ID, LB_INNER_DEFAULT]
    ])


def back_pair_push_down(duration, amount):
    board.bus_servo_set_position(duration, [
        [RB_MIDDLE_ID, RB_MIDDLE_DEFAULT + amount],  # RB push down (+)
        [LB_MIDDLE_ID, LB_MIDDLE_DEFAULT - amount]   # LB push down (-)
    ])


def back_pair_lift_leg(duration, amount):
    board.bus_servo_set_position(duration, [
        [RB_MIDDLE_ID, RB_MIDDLE_DEFAULT - amount],  # RB lift (-)
        [LB_MIDDLE_ID, LB_MIDDLE_DEFAULT + amount]   # LB lift (+)
    ])


def back_pair_middle_default(duration):
    board.bus_servo_set_position(duration, [
        [RB_MIDDLE_ID, RB_MIDDLE_DEFAULT],
        [LB_MIDDLE_ID, LB_MIDDLE_DEFAULT]
    ])


def back_pair_extend_outward(duration, amount):
    board.bus_servo_set_position(duration, [
        [RB_OUTER_ID, RB_OUTER_DEFAULT + amount],  # RB extend (+)
        [LB_OUTER_ID, LB_OUTER_DEFAULT - amount]   # LB extend (-)
    ])


def back_pair_retract_inward(duration, amount):
    board.bus_servo_set_position(duration, [
        [RB_OUTER_ID, RB_OUTER_DEFAULT - amount],  # RB retract (-)
        [LB_OUTER_ID, LB_OUTER_DEFAULT + amount]   # LB retract (+)
    ])


def back_pair_outer_default(duration):
    board.bus_servo_set_position(duration, [
        [RB_OUTER_ID, RB_OUTER_DEFAULT],
        [LB_OUTER_ID, LB_OUTER_DEFAULT]
    ])
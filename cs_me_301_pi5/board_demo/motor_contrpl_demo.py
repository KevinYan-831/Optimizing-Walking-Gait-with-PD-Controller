#!/usr/bin/python3
# coding=utf8
import sys
import time
import signal
import ros_robot_controller_sdk as rrc

if sys.version_info.major == 2:
    print('Please run this program with python3!')
    sys.exit(0)
    
print('''
**********************************************************
********功能:幻尔科技树莓派扩展板，控制直流带电机(Function:Hiwonder Raspberry Pi expansion board, drive DC motor)**********
**********************************************************
----------------------------------------------------------
Official website:https://www.hiwonder.com
Online mall:https://hiwonder.tmall.com
----------------------------------------------------------
Tips:
 * 按下Ctrl+C可关闭此次程序运行，若失败请多次尝试！(Press "Ctrl+C" to exit the program. If it fails, please try again multiple times!)
----------------------------------------------------------
''')
board = rrc.Board()

start = True
#关闭前处理(process before closing)
def Stop(signum, frame):
    global start

    start = False
    print('关闭中...')
    board.set_motor_duty([[1, 0], [2, 0], [3, 0], [4, 0]])  # 关闭所有电机(close all motors)

signal.signal(signal.SIGINT, Stop)

if __name__ == '__main__':
    
    while True:
        board.set_motor_duty([[1, 35]])  #设置1号电机速度35(set the speed of motor 1 to 35)
        time.sleep(2)
        board.set_motor_duty([[1, 90]])  #设置1号电机速度90(set the speed of motor 1 to 90)
        time.sleep(1) 
        board.set_motor_duty([[1, 0]])   
        time.sleep(6)
        if not start:
            board.set_motor_duty([[1, 0], [2, 0], [3, 0], [4, 0]])  # 关闭所有电机(close all motors)
            print('已关闭')
            break
    
    
        

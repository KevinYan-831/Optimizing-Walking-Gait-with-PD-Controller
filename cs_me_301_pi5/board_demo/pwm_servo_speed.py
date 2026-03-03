import sys
import time
import signal
import threading
import ros_robot_controller_sdk as rrc

print('''
**********************************************************
********功能:幻尔科技树莓派扩展板，控制PWM舵机速度(function:Hiwonder Raspberry Pi expansion board, PWM servo speed control)**********
**********************************************************
----------------------------------------------------------
Official website:https://www.hiwonder.com
Online mall:https://hiwonder.tmall.com
----------------------------------------------------------
Tips:
 * 按下Ctrl+C可关闭此次程序运行，若失败请多次尝试！(Press 'Ctrl+C' to exit the program. If it fails, please try again multiple times!)
----------------------------------------------------------
''')
board = rrc.Board()
start = True

# 关闭前处理(process before closing)
def Stop(signum, frame):
    global start
    start = False
    print('关闭中...')

signal.signal(signal.SIGINT, Stop)

if __name__ == '__main__':
    while True:
        board.pwm_servo_set_position(0.5, [[1, 1500]]) # 设置1号舵机脉宽为1500(set the pulse width of servo 1 to 1500)
        time.sleep(0.5)
        board.pwm_servo_set_position(0.5, [[1, 1000]]) # 设置1号舵机脉宽为1000(set the pulse width of servo 1 to 1000)
        time.sleep(0.5)
        board.pwm_servo_set_position(0.5, [[1, 500]]) # 设置1号舵机脉宽为500(set the pulse width of servo 1 to 500)
        time.sleep(0.5)
        board.pwm_servo_set_position(1, [[1, 1000]]) # 设置1号舵机脉宽为1000(set the pulse width of servo 1 to 1000)
        time.sleep(1)
        board.pwm_servo_set_position(1, [[1, 1500]]) # 设置1号舵机脉宽为1500(set the pulse width of servo 1 to 1500)
        time.sleep(1)
        board.pwm_servo_set_position(1, [[1, 2000]]) # 设置1号舵机脉宽为2000(set the pulse width of servo 1 to 2000)
        time.sleep(1)
        board.pwm_servo_set_position(1, [[1, 2500]]) # 设置1号舵机脉宽为2500(set the pulse width of servo 1 to 2500)
        time.sleep(1)
        if not start:
            board.pwm_servo_set_position(1, [[1, 1500]]) # 设置1号舵机脉宽为1500(set the pulse width of servo 1 to 1500)
            time.sleep(1)
            print('已关闭')
            break
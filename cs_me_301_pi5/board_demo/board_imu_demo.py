import time
import common.ros_robot_controller_sdk as rrc

'''
    程序功能：IMU例程(MPU6050)(Program function: IMU(MPU6050))

    运行效果：程序运行后，在屏幕不断输出获取到的IMU的数据(Program outcome: After running the program, the obtained IMU data will be continuously output on the screen)

    对应教程文档路径：  SpiderPi Pro智能六足机器人\第10章 树莓派主板及扩展板拓展课程\2.拓展课程之树莓派扩展板课程\第4课 加速度计的使用(The path of corresponding tutorial： SpiderPi Pro 5\10.Raspberry Pi Controller & Expansion Board\2.Raspberry Pi Expansion Board Lesson\Lesson 4 Use of Accelerometer)
'''
board = rrc.Board()
board.enable_reception()

while True:
    try:
        res = board.get_imu()          # 获取IMU数据(obtain IMU data)
        if res is not None:
            print(res)           # 输出获取到的IMU数据(output the obtained IMU data)
        
        time.sleep(0.01)
    except KeyboardInterrupt:
        break

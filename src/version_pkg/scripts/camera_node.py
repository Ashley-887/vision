#!/usr/bin/env python3

import rospy
import cv2
from sensor_msgs.msg import Image, CameraInfo, Imu
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from cv_bridge import CvBridge
import random  # 用于生成示例IMU数据

class CameraNode:
    def __init__(self):
        # 初始化 ROS 节点
        rospy.init_node('camera_node', anonymous=True)
        rospy.loginfo("Camera Node initialized.")

        # 摄像头打开（0 为默认摄像头）
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            rospy.logerr("Failed to open camera.")
            return
        
        # 发布器：发布图像数据
        self.color_image_pub = rospy.Publisher('/camera/color/image_raw', Image, queue_size=10)
        self.depth_image_pub = rospy.Publisher('/camera/depth/image_rect_raw', Image, queue_size=10)  # 假设有深度摄像头

        # 发布器：发布相机内参数据
        self.color_camera_info_pub = rospy.Publisher('/camera/color/camera_info', CameraInfo, queue_size=10)
        self.depth_camera_info_pub = rospy.Publisher('/camera/depth/camera_info', CameraInfo, queue_size=10)

        # 发布器：发布 IMU 数据
        self.gyro_pub = rospy.Publisher('/camera/gyro/sample', Imu, queue_size=10)
        self.accel_pub = rospy.Publisher('/camera/accel/sample', Imu, queue_size=10)

        # 发布器：发布诊断消息
        self.diagnostics_pub = rospy.Publisher('/diagnostics', DiagnosticArray, queue_size=10)

        # 创建 CvBridge 实例以转换 OpenCV 图像到 ROS 图像消息
        self.bridge = CvBridge()

        # 设置发布频率（10Hz）
        self.rate = rospy.Rate(10)
        rospy.loginfo("Camera Node setup complete. Starting to capture images.")

    def start(self):
        while not rospy.is_shutdown():
            # 从摄像头读取图像
            ret, frame = self.cap.read()
            if ret:
                rospy.loginfo("Captured image from camera.")

                # 将 OpenCV 图像转换为 ROS 图像消息并发布
                color_image_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                self.color_image_pub.publish(color_image_msg)
                rospy.loginfo("Published color image.")

                # 模拟发布深度图像（假设有深度摄像头）
                #depth_image_msg = self.bridge.cv2_to_imgmsg(depth_frame, encoding="passthrough")
                #self.depth_image_pub.publish(depth_image_msg)
                #rospy.loginfo("Published depth image.")

                # 发布相机内参（假设使用静态参数）
                self.publish_camera_info()

                # 发布 IMU 数据
                self.publish_imu_data()

                # 发布诊断消息
                self.publish_diagnostics()

                # 显示图像窗口
                # cv2.imshow('Camera Image', frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     rospy.loginfo("Exiting on user request.")
                #     break

            else:
                rospy.logwarn("Failed to capture image from camera.")

            self.rate.sleep()

    def publish_camera_info(self):
        # 创建并发布 color 相机内参消息
        color_camera_info = CameraInfo()
        color_camera_info.width = 640  # 假设图像宽度
        color_camera_info.height = 480  # 假设图像高度
        # 相机内参示例，真实应用中应根据实际相机校准信息填写
        color_camera_info.K = [1000, 0, 320, 0, 1000, 240, 0, 0, 1]  # 焦距和主点
        self.color_camera_info_pub.publish(color_camera_info)
        rospy.loginfo("Published color camera info.")

        # 创建并发布 depth 相机内参消息（假设存在深度相机）
        depth_camera_info = CameraInfo()
        depth_camera_info.width = 640
        depth_camera_info.height = 480
        depth_camera_info.K = [1000, 0, 320, 0, 1000, 240, 0, 0, 1]
        self.depth_camera_info_pub.publish(depth_camera_info)
        rospy.loginfo("Published depth camera info.")

    def publish_imu_data(self):
        # 创建并发布 IMU 数据消息
        imu_msg = Imu()

        # 模拟陀螺仪数据
        imu_msg.angular_velocity.x = random.uniform(-1.0, 1.0)
        imu_msg.angular_velocity.y = random.uniform(-1.0, 1.0)
        imu_msg.angular_velocity.z = random.uniform(-1.0, 1.0)

        # 模拟加速度数据
        imu_msg.linear_acceleration.x = random.uniform(-9.8, 9.8)
        imu_msg.linear_acceleration.y = random.uniform(-9.8, 9.8)
        imu_msg.linear_acceleration.z = random.uniform(-9.8, 9.8)

        # 发布陀螺仪和加速度数据
        self.gyro_pub.publish(imu_msg)
        self.accel_pub.publish(imu_msg)
        rospy.loginfo("Published IMU data.")

    def publish_diagnostics(self):
        # 创建并发布诊断消息
        diag_msg = DiagnosticArray()
        status = DiagnosticStatus()
        status.name = "Camera Diagnostics"
        status.level = DiagnosticStatus.OK
        status.message = "Camera is running normally"
        diag_msg.status.append(status)

        # 发布诊断消息
        self.diagnostics_pub.publish(diag_msg)
        rospy.loginfo("Published diagnostics.")

    def __del__(self):
        try:
            # 释放摄像头资源
            self.cap.release()
            # 关闭所有 OpenCV 窗口
            cv2.destroyAllWindows()
            # 记录日志
            rospy.loginfo("Camera Node shutting down.")
        except Exception as e:
            print(f"Error during shutdown: {e}")

if __name__ == '__main__':
    try:
        # 创建 CameraNode 实例并启动
        node = CameraNode()
        node.start()
    except rospy.ROSInterruptException:
        pass

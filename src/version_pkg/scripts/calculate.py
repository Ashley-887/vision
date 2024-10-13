#!/usr/bin/env python3

import rospy
from version_pkg.msg import Point3D, Object
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import yaml

class SolverOnceNode:
    def __init__(self):
        # 初始化节点
        rospy.init_node('solveronce_node')

        # 加载相机配置文件
        self.camera_matrix, self.dist_coeffs = self.load_camera_config('/home/hushaoting/ros_wk/src/version_pkg/config/camera_calibration.yaml')

        # 创建发布者，发布3D点消息
        self.points_pub = rospy.Publisher('/solveronce/result_point', Point3D, queue_size=10)

        # 创建订阅者，订阅匹配对象消息
        self.target_sub = rospy.Subscriber('/target', Object, self.listener_callback)

        # 创建订阅者，订阅深度图像消息
        self.depth_sub = rospy.Subscriber('/camera/depth/image_rect_raw', Image, self.depth_callback)

        self.bridge = CvBridge()  # 用于图像转换的桥接
        self.depth_image = None   # 初始化深度图像为空

        # 初始化深度图像的角点深度值
        self.corner_depths = {'top_left': 0, 'top_right': 0, 'bottom_left': 0, 'bottom_right': 0}

    def depth_callback(self, depth_msg):
        # 将ROS图像消息转换为OpenCV格式
        self.depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        # 获取深度图像的高度和宽度
        img_height, img_width = self.depth_image.shape

        # 提取深度图像四个角点的深度值
        self.corner_depths['top_left'] = self.get_depth_at(0, 0)  # 左上角
        self.corner_depths['top_right'] = self.get_depth_at(0, img_width - 1)  # 右上角
        self.corner_depths['bottom_left'] = self.get_depth_at(img_height - 1, 0)  # 左下角
        self.corner_depths['bottom_right'] = self.get_depth_at(img_height - 1, img_width - 1)  # 右下角

        rospy.loginfo(f'角点深度值: {self.corner_depths}')

    def listener_callback(self, obj):
        # 检查是否已经接收到深度图像
        if self.depth_image is None:
            rospy.logwarn("尚未收到深度图像。")
            return

        # 计算目标中心点的像素坐标
        u = (obj.box[0] + obj.box[2]) / 2  
        v = (obj.box[1] + obj.box[3]) / 2 
        
        # 获取该像素的深度值
        depth = self.get_depth_at(v, u)

        # 检查深度值是否有效
        if depth <= 0:
            rospy.logwarn(f"深度值无效: {depth}")
            return

        # 将像素坐标转换为3D坐标
        point_3d = self.pixel_to_3d(u, v, depth, self.camera_matrix)

        # 根据四个角点确定平面方程
        plane_normal, d = self.fit_plane()

        # 计算点到平面的距离
        distance_to_plane = self.distance_point_to_plane(point_3d, plane_normal, d)

        # 创建3D点消息
        point_msg = Point3D(
            label=obj.label,
            confidence=obj.confidence,
            x=point_3d[0],
            y=point_3d[1],
            z=point_3d[2]
        )

        # 发布3D点消息并输出距离
        self.points_pub.publish(point_msg)
        rospy.loginfo(f'发布3D点: {point_3d}, 到平面的距离: {distance_to_plane} 米')

    def get_depth_at(self, u, v):
        # 获取指定像素位置的深度值，并检查有效性
        if 0 <= int(v) < self.depth_image.shape[0] and 0 <= int(u) < self.depth_image.shape[1]:
            return self.depth_image[int(v), int(u)]
        else:
            rospy.logwarn("像素坐标超出图像范围。")
            return -1

    def pixel_to_3d(self, u, v, depth, camera_matrix):
        # 从相机内参矩阵中获取参数
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]

        # 使用深度值计算3D坐标
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth

        return np.array([x, y, z])

    def fit_plane(self):
        # 使用四个角点的3D坐标拟合平面方程
        corners_3d = []

        for (corner, depth) in self.corner_depths.items():
            u, v = self.get_corner_pixel_coords(corner)
            point_3d = self.pixel_to_3d(u, v, depth, self.camera_matrix)
            corners_3d.append(point_3d)

        # 转换为 NumPy 数组
        corners_3d = np.array(corners_3d)

        # 计算平面法向量
        p1, p2, p3 = corners_3d[0], corners_3d[1], corners_3d[2]
        plane_normal = np.cross(p2 - p1, p3 - p1)  # 法向量
        plane_normal = plane_normal / np.linalg.norm(plane_normal)  # 单位化

        # 平面方程: n*x + n*y + n*z + d = 0, 求解 d
        d = -np.dot(plane_normal, p1)

        return plane_normal, d

    def get_corner_pixel_coords(self, corner_name):
        """根据角点名称返回像素坐标"""
        img_height, img_width = self.depth_image.shape
        if corner_name == 'top_left':
            return 0, 0
        elif corner_name == 'top_right':
            return img_width - 1, 0
        elif corner_name == 'bottom_left':
            return 0, img_height - 1
        elif corner_name == 'bottom_right':
            return img_width - 1, img_height - 1

    def distance_point_to_plane(self, point, normal, d):
        # 计算点到平面的距离
        return np.abs(np.dot(normal, point) + d) / np.linalg.norm(normal)

    def load_camera_config(self, config_file):
        # 加载相机配置文件，解析相机矩阵和畸变系数
        try:
            with open(config_file, 'r') as file:
                config = yaml.safe_load(file)

            camera_matrix = np.array(config['camera_matrix']['data']).reshape(config['camera_matrix']['rows'], config['camera_matrix']['cols'])
            dist_coeffs = np.array(config['distortion_coefficients']['data'])
            rospy.loginfo('相机配置加载成功。')
            return camera_matrix, dist_coeffs
        except Exception as e:
            rospy.logerr(f'加载相机配置失败: {e}')
            return None, None

if __name__ == '__main__':
    node = SolverOnceNode()  # 创建节点实例
    rospy.spin()  # 保持节点运行

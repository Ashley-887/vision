#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from version_pkg.msg import Object, Objects  # 替换为你的包名
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

# YOLO 模型初始化 (例如YOLOv8, 你也可以根据需求替换为自己的模型)
from ultralytics import YOLO
model = YOLO("yolov8m.pt")  # 加载模型，替换为实际模型路径

# OpenCV-ROS桥
bridge = CvBridge()

def image_callback(image_msg):
    try:
        # 将ROS Image消息转换为OpenCV图像
        cv_image = bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))
        return

    # 执行推理
    results = model.predict(source=cv_image)  # YOLO 推理

    # 创建 Detections 消息
    detections_msg = Objects()
    
    # 遍历 YOLO 检测结果
    for result in results:
        for box, conf, label in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            
            obj = Object()
            obj.box = [box[0], box[1], box[2], box[3]]  # [x_min, y_min, x_max, y_max]
            obj.label = result.names[int(label)]
            obj.confidence = conf
            detections_msg.detections.append(obj)

            x_min, y_min, x_max, y_max = [int(v) for v in box]
            cv2.rectangle(cv_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(cv_image, f"{label} {conf:.2f}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 发布检测结果
    detection_pub.publish(detections_msg)

    try:
        image_with_boxes_msg = bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
        image_pub.publish(image_with_boxes_msg)
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))

if __name__ == '__main__':
    rospy.init_node('object_detection_node', anonymous=True)

    # 创建一个订阅者，订阅相机图像话题
    rospy.Subscriber('/camera/color/image_raw', Image, image_callback)

    # 创建一个发布者，发布检测结果到 /detections 话题
    detection_pub = rospy.Publisher('/detections', Objects, queue_size=10)

    image_pub = rospy.Publisher('/camera/color/image_with_boxes', Image, queue_size=10)

    rospy.loginfo("Object detection node started. Waiting for images...")

    # 循环等待回调
    rospy.spin()

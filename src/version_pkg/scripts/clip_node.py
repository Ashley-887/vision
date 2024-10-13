#!/usr/bin/env python3

import rospy
import torch
import cn_clip.clip as clip
from cn_clip.clip import load_from_name
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import String
from version_pkg.msg import Objects, Object
from PIL import Image as PILImage
import cv2
import numpy as np

class CLIPNode:
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node('clip_node', anonymous=True)

        # 加载CLIP模型
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = load_from_name("ViT-B-16", device=self.device)

        # OpenCV和ROS消息转换工具
        self.bridge = CvBridge()

        # 订阅来自摄像头的图像
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)

        # 订阅YOLO推理结果
        self.yolo_sub = rospy.Subscriber("/detections", Objects, self.yolo_callback)

        # 订阅用户的意图信息
        self.string_sub = rospy.Subscriber("/man/say", String, self.man_callback)

        self.target_pub = rospy.Publisher('/target', Object, queue_size=10)

        # 存储当前帧和推理结果
        self.current_frame = None
        self.current_objects = None

        # 要匹配的文本描述
        self.text = None

    def image_callback(self, img_msg):
        # 将ROS图像消息转换为OpenCV格式
        try:
            self.current_frame = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"Failed to convert image: {e}")

    def yolo_callback(self, yolo_msg):
        # 从YOLO推理结果中获取bounding boxes
        self.current_objects = yolo_msg.detections

    def man_callback(self, man_msg):
        # 接收文本信息并进行CLIP推理

        self.text = clip.tokenize([man_msg.data]).to(self.device)
        if self.current_frame is not None and self.current_objects is not None:
            self.perform_clip_inference()

    def perform_clip_inference(self):
        # 先裁剪YOLO框中的对象
        boxes = []
        now_objs = self.current_objects
        for obj in now_objs:
            boxes.append([obj.box[0], obj.box[1], obj.box[2], obj.box[3]])
        cropped_images = self.crop_objects(self.current_frame, boxes)
        # 对裁剪的图像与文本进行匹配
        index = self.match_images_with_text(cropped_images, self.text)
        self.target_pub.publish(now_objs[index])
        

    def crop_objects(self, image, bounding_boxes):
        # 根据YOLO的bounding boxes裁剪图像中的物体
        cropped_images = []
        for box in bounding_boxes:
            x1, y1, x2, y2 = map(int, box)
            cropped_img = image[y1:y2, x1:x2]  # 裁剪物体区域
            cropped_images.append(cropped_img)
        return cropped_images

    def match_images_with_text(self, images, text):
        # 使用CLIP模型进行图像与文本匹配
        image_tensors = []
        for img in images:
            img_pil = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            img_preprocessed = self.preprocess(img_pil).unsqueeze(0).to(self.device)
            image_tensors.append(img_preprocessed)

        image_batch = torch.cat(image_tensors, dim=0)
        with torch.no_grad():

            _, logits_per_text = self.model.get_similarity(image_batch, text)
            probs = logits_per_text.softmax(dim=-1).cpu().numpy().squeeze()

        return np.argmax(probs)

if __name__ == "__main__":
    try:
        clip_node = CLIPNode()
        rospy.spin()  # 保持节点运行
    except rospy.ROSInterruptException:
        pass
import os
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraImageSaver(Node):
    """
    极简相机图片保存接口
    功能：订阅相机话题 → 等待获取图像 → 保存到指定路径（无窗口显示）
    """
    def __init__(self):
        super().__init__("camera_image_saver_node")
        
        # 1. 核心配置（可根据需要调整）
        self.camera_topic = "/camera/color/image_raw"  # 相机图像话题
        self.save_dir = "/home/jetson/chapt7_ws/src/autopatrol_robot/tempphoto/"  # 指定保存路径
        self.image_filename = "camera_capture.png"  # 默认保存的文件名
        self.bridge = CvBridge()  # ROS图像 ↔ OpenCV图像转换工具
        self.latest_image = None  # 缓存最新的相机图像
        
        # 2. 自动创建保存目录（避免路径不存在报错）
        self._create_save_dir()
        
        # 3. 订阅相机话题，实时获取图像
        self.image_sub = self.create_subscription(
            Image,
            self.camera_topic,
            self.image_callback,  # 图像回调函数
            1  # 【修复】：将队列大小设为1，丢弃移动过程中的旧图像
        )
        self.get_logger().info(f"已订阅相机话题: {self.camera_topic}")
        self.get_logger().info(f"图片将保存到: {self.save_dir}")

    def _create_save_dir(self):
        """自动创建保存目录（如果不存在）"""
        if not os.path.exists(self.save_dir):
            try:
                os.makedirs(self.save_dir)
                self.get_logger().info(f"创建保存目录成功: {self.save_dir}")
            except Exception as e:
                self.get_logger().error(f"创建保存目录失败: {str(e)}")
                raise  # 目录创建失败则终止节点

    def image_callback(self, msg: Image):
        """相机图像回调函数：缓存最新图像"""
        try:
            # 将ROS的Image消息转换为OpenCV格式（BGR8是RGB相机的标准格式）
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"图像格式转换失败: {str(e)}")

    def wait_for_image(self, timeout=10.0):
        """
        等待获取相机图像（超时退出）
        :param timeout: 最长等待时间（秒）
        :return: bool - 是否获取到图像
        """
        # 【修复】：强制清空历史图像，确保每次调用都去拉取最新的一帧
        self.latest_image = None  

        start_time = time.time()
        self.get_logger().info("等待获取最新相机图像...")
        
        while self.latest_image is None:
            # 检查是否超时
            if time.time() - start_time > timeout:
                self.get_logger().error(f"等待图像超时（{timeout}秒），请检查相机！")
                return False
            # 让ROS处理回调（关键：必须调用spin_once，否则回调不会执行）
            rclpy.spin_once(self, timeout_sec=0.1)
        
        self.get_logger().info("成功获取最新相机图像！")
        return True

    def save_image(self, filename=None):
        """
        对外暴露的核心接口：保存当前相机画面到指定路径
        :param filename: 自定义文件名（可选，默认用self.image_filename）
        :return: bool - 保存成功/失败
        """
        if filename is None:
            filename = self.image_filename
        
        save_path = os.path.join(self.save_dir, filename)
        
        # 检查是否有缓存的图像
        if self.latest_image is None:
            self.get_logger().error("未获取到相机图像！请检查相机是否正常/话题是否正确")
            return False
        
        # 保存图像到指定路径
        try:
            cv2.imwrite(save_path, self.latest_image)
            self.get_logger().info(f"图片保存成功: {save_path}")
            return True
        except Exception as e:
            self.get_logger().error(f"图片保存失败: {str(e)}")
            return False

# ------------------- 调用示例 -------------------
def main(args=None):
    # 初始化ROS2
    rclpy.init(args=args)
    
    # 创建相机保存节点
    camera_saver = CameraImageSaver()
    
    # 关键：先等待获取图像，再保存
    if camera_saver.wait_for_image(timeout=10.0):
        # 方式1：保存默认文件名
        camera_saver.save_image()
        
        # 方式2：自定义文件名保存（可选）
        # timestamp = time.strftime("%Y%m%d_%H%M%S")
        # camera_saver.save_image(f"capture_{timestamp}.png")
    
    # 销毁节点、关闭ROS2
    camera_saver.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
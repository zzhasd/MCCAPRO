import rclpy
from geometry_msgs.msg import PoseStamped, Pose
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from tf2_ros import TransformListener, Buffer
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from rclpy.duration import Duration
# 添加服务接口
from autopatrol_interfaces.srv import SpeachText
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os  # 新增：用于确保保存目录存在

class PatrolNode(BasicNavigator):
    def __init__(self, node_name='patrol_node'):
        super().__init__(node_name)
        # 导航相关定义
        self.declare_parameter('initial_point', [0.0, 0.0, 0.0])
        self.declare_parameter('target_points', [0.0, 0.0, 0.0, 1.0, 1.0, 1.57])
        self.initial_point_ = self.get_parameter('initial_point').value
        self.target_points_ = self.get_parameter('target_points').value
        # 实时位置获取 TF 相关定义
        self.buffer_ = Buffer()
        self.listener_ = TransformListener(self.buffer_, self)
        self.speach_client_ = self.create_client(SpeachText, 'speech_text')

        # -------------------------- 关键修改1：固定图片保存路径 --------------------------
        # 直接指定固定路径，覆盖参数读取逻辑
        self.image_save_path = '/home/zzh/ros2bookcode-master/chapt7/chapt7_ws/src/autopatrol_robot/tempphoto/'
        # 确保目录存在，避免保存图片时出错
        os.makedirs(self.image_save_path, exist_ok=True)
        
        self.bridge = CvBridge()
        self.latest_image = None
        self.subscription_image = self.create_subscription(
            Image, '/camera_sensor/image_raw', self.image_callback, 10)

    def image_callback(self, msg):
        """
        将最新的消息放到 latest_image 中
        """
        self.latest_image = msg

    def record_image(self):
        """
        记录图像（使用固定路径）
        """
        if self.latest_image is not None:
          try:
              pose = self.get_current_pose()
              # 拼接完整保存路径，确保路径末尾有/避免文件名错误
              save_path = f'{self.image_save_path}image_{pose.translation.x:.2f}_{pose.translation.y:.2f}.png'
              cv_image = self.bridge.imgmsg_to_cv2(self.latest_image)
              cv2.imwrite(save_path, cv_image)
              self.get_logger().info(f'图片已保存至: {save_path}')
          except Exception as e:
              self.get_logger().error(f'保存图片失败: {str(e)}')
        else:
            self.get_logger().warn('无可用图像数据，无法保存图片')

    def speach_text(self, text):
        """
        调用服务播放语音
        """
        while not self.speach_client_.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('语合成服务未上线，等待中。。。')

        request = SpeachText.Request()
        request.text = text
        future = self.speach_client_.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            result = future.result().result
            if result:
                self.get_logger().info(f'语音合成成功：{text}')
            else:
                self.get_logger().warn(f'语音合成失败：{text}')
        else:
            self.get_logger().warn('语音合成服务请求失败')

    def get_pose_by_xyyaw(self, x, y, yaw):
        """
        通过 x,y,yaw 合成 PoseStamped
        """
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()  # 新增：添加时间戳，避免警告
        pose.pose.position.x = x
        pose.pose.position.y = y
        rotation_quat = quaternion_from_euler(0, 0, yaw)
        pose.pose.orientation.x = rotation_quat[0]
        pose.pose.orientation.y = rotation_quat[1]
        pose.pose.orientation.z = rotation_quat[2]
        pose.pose.orientation.w = rotation_quat[3]
        return pose

    def init_robot_pose(self):
        """
        初始化机器人位姿
        """
        # 从参数获取初始化点
        self.initial_point_ = self.get_parameter('initial_point').value
        # 合成位姿并进行初始化
        self.setInitialPose(self.get_pose_by_xyyaw(
            self.initial_point_[0], self.initial_point_[1], self.initial_point_[2]))
        # 等待直到导航激活
        self.waitUntilNav2Active()

    def get_target_points(self):
        from .generate_target_points import generate_target_points
        """
        通过 generate_target_points 获取随机目标点
        """
        self.get_logger().info('正在生成随机目标点并保存图片...')
        # -------------------------- 关键修改2：目标点生成也使用固定路径 --------------------------
        target_dir = '/home/zzh/ros2bookcode-master/chapt7/chapt7_ws/src/autopatrol_robot/tempphoto/'
        points = generate_target_points(image_dir=target_dir)
        for index, point in enumerate(points):
            self.get_logger().info(f'获取到目标点: {index}->({point[0]:.2f},{point[1]:.2f},{point[2]:.2f})')
        return points

    def nav_to_pose(self, target_pose):
        """
        导航到指定位姿
        """
        self.waitUntilNav2Active()
        result = self.goToPose(target_pose)
        while not self.isTaskComplete():
            feedback = self.getFeedback()
            if feedback:
                self.get_logger().info(f'预计: {Duration.from_msg(feedback.estimated_time_remaining).nanoseconds / 1e9} s 后到达')
        # 最终结果判断
        result = self.getResult()
        if result == TaskResult.SUCCEEDED:
            self.get_logger().info('导航结果：成功')
        elif result == TaskResult.CANCELED:
            self.get_logger().warn('导航结果：被取消')
        elif result == TaskResult.FAILED:
            self.get_logger().error('导航结果：失败')
        else:
            self.get_logger().error('导航结果：返回状态无效')

    def get_current_pose(self):
        """
        通过TF获取当前位姿
        """
        while rclpy.ok():
            try:
                tf = self.buffer_.lookup_transform(
                    'map', 'base_footprint', rclpy.time.Time(seconds=0), rclpy.time.Duration(seconds=1))
                transform = tf.transform
                rotation_euler = euler_from_quaternion([
                    transform.rotation.x,
                    transform.rotation.y,
                    transform.rotation.z,
                    transform.rotation.w
                ])
                self.get_logger().info(
                    f'平移:{transform.translation},旋转四元数:{transform.rotation}:旋转欧拉角:{rotation_euler}')
                return transform
            
            except Exception as e:
                self.get_logger().warn(f'不能够获取坐标变换，原因: {str(e)}')
    
def main():
    rclpy.init()
    patrol = PatrolNode()
    patrol.speach_text(text='正在初始化位置')
    patrol.init_robot_pose()
    patrol.speach_text(text='位置初始化完成')

    # 只生成一次目标点列表，后续循环复用
    target_points_list = patrol.get_target_points()  
    if not target_points_list:  # 增加空值判断，避免空列表导致循环无意义
        patrol.get_logger().error('未生成任何目标点，程序退出')
        rclpy.shutdown()
        return

    while rclpy.ok():
        for point in target_points_list:  # 遍历初始化生成的固定目标点列表
            x, y, yaw = point[0], point[1], point[2]
            # 导航到目标点
            target_pose = patrol.get_pose_by_xyyaw(x, y, yaw)
            patrol.speach_text(text=f'准备前往目标点{x:.2f},{y:.2f}')
            patrol.nav_to_pose(target_pose)
            patrol.speach_text(text=f"已到达目标点{x:.2f},{y:.2f},准备记录图像")
            patrol.record_image()
            patrol.speach_text(text=f"图像记录完成")
        # 可选：遍历完一轮目标点后，添加语音提示
        patrol.speach_text(text="已完成所有目标点遍历，将再次循环")
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()
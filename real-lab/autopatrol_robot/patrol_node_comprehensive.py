import rclpy
import time  # 用于生成时间戳
from geometry_msgs.msg import PoseStamped, Pose
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from tf2_ros import TransformListener, Buffer
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from rclpy.duration import Duration
# 导入TTS工具模块
from .tts_utils import synthesize_and_play
# 导入相机拍照工具类
from .camera_utils import CameraImageSaver
# 导入传感器工具类
from .sensor_utils import SensorReader  # 新增

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
        # 初始化相机保存节点（提前订阅相机话题，避免拍照时等待）
        self.camera_saver = CameraImageSaver()
        
        # 初始化传感器读取器（新增）
        self.sensor_reader = SensorReader(logger=self.get_logger())  # 传入ROS2日志器
        self.sensor_reader.init_sensors()  # 初始化传感器

    def get_pose_by_xyyaw(self, x, y, yaw):
        """通过 x,y,yaw 合成 PoseStamped"""
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        rotation_quat = quaternion_from_euler(0, 0, yaw)
        pose.pose.orientation.x = rotation_quat[0]
        pose.pose.orientation.y = rotation_quat[1]
        pose.pose.orientation.z = rotation_quat[2]
        pose.pose.orientation.w = rotation_quat[3]
        return pose

    def init_robot_pose(self):
        """初始化机器人位姿"""
        self.initial_point_ = self.get_parameter('initial_point').value
        self.setInitialPose(self.get_pose_by_xyyaw(
            self.initial_point_[0], self.initial_point_[1], self.initial_point_[2]))
        self.waitUntilNav2Active()

    def get_target_points(self):
        """通过 generate_target_points 获取随机目标点"""
        from .generate_target_points_PCA import generate_target_points
        self.get_logger().info('正在生成随机目标点...')
        target_dir = '/home/jetson/chapt7_ws/src/autopatrol_robot/tempphoto/'
        points = generate_target_points(image_dir=target_dir)
        for index, point in enumerate(points):
            self.get_logger().info(f'获取到目标点: {index}->({point[0]:.2f},{point[1]:.2f},{point[2]:.2f})')
        return points

    def nav_to_pose(self, target_pose):
        """导航到指定位姿并返回结果"""
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
        return result

    def get_current_pose(self):
        """通过TF获取当前位姿"""
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

    # ---------------------- 封装通用拍照函数（核心重构） ----------------------
    def take_photo(self, photo_prefix: str, x: float, y: float) -> bool:
        """
        通用拍照函数：封装所有拍照逻辑，消除代码重复
        :param photo_prefix: 照片文件名前缀（如 init/patrol），用于区分照片类型
        :param x: 拍照位置的x坐标（用于文件名）
        :param y: 拍照位置的y坐标（用于文件名）
        :return: 拍照是否成功
        """
        # 1. 语音提示正在拍照
        taking_photo_text = "正在拍照"
        self.get_logger().info(taking_photo_text)
        synthesize_and_play(taking_photo_text)

        # 2. 生成带时间戳+坐标的文件名
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        photo_filename = f"{photo_prefix}_{timestamp}_x{x:.2f}_y{y:.2f}.png"

        # 3. 等待相机图像并保存
        if self.camera_saver.wait_for_image(timeout=10.0):
            save_success = self.camera_saver.save_image(filename=photo_filename)
            if save_success:
                # 拍照成功提示
                photo_done_text = "拍照完成"
                self.get_logger().info(photo_done_text)
                synthesize_and_play(photo_done_text)
                return True
            else:
                # 拍照失败提示
                photo_fail_text = "拍照失败"
                self.get_logger().error(photo_fail_text)
                synthesize_and_play(photo_fail_text)
                return False
        else:
            # 图像超时提示
            photo_timeout_text = "获取相机图像超时，拍照失败"
            self.get_logger().error(photo_timeout_text)
            synthesize_and_play(photo_timeout_text)
            return False
    
    # ---------------------- 新增：传感器数据播报函数 ----------------------
    def broadcast_sensor_data(self, location_desc: str):
        """
        读取并播报传感器数据
        :param location_desc: 位置描述（如"初始位置"、"目标点(1.0,2.0)"）
        """
        if not self.sensor_reader.is_initialized:
            self.get_logger().warn("传感器未初始化，无法播报温湿度数据")
            return
        
        # 读取传感器数据
        sensor_data = self.sensor_reader.read_all_sensors()
        
        # 构造播报文本
        if sensor_data['temperature'] is not None and sensor_data['humidity'] is not None:
            broadcast_text = f"{location_desc}当前温度{sensor_data['temperature']}，当前湿度{sensor_data['humidity']}"
        else:
            broadcast_text = f"{location_desc}温湿度数据读取失败"
        
        # 日志输出 + 语音播报
        self.get_logger().info(f"传感器数据播报: {broadcast_text}")
        synthesize_and_play(broadcast_text)
        
        # 可选：打印完整传感器数据（包含CO2和TVOC）到日志
        self.get_logger().info(
            f"完整传感器数据 - 温度: {sensor_data['temperature']}°C, "
            f"湿度: {sensor_data['humidity']}%, "
            f"eCO2: {sensor_data['eco2']}ppm, "
            f"TVOC: {sensor_data['tvoc']}ppb"
        )
    # -------------------------------------------------------------------------

def main():
    rclpy.init()
    patrol = PatrolNode()
    
    # 初始化位置：日志+语音
    init_text = "正在初始化位置"
    patrol.get_logger().info(init_text)
    synthesize_and_play(init_text)
    
    patrol.init_robot_pose()
    
    # 初始化完成：日志+语音
    init_complete_text = "位置初始化完成"
    patrol.get_logger().info(init_complete_text)
    synthesize_and_play(init_complete_text)

    # ---------------------- 初始化完成后播报初始位置传感器数据（新增） ----------------------
    patrol.broadcast_sensor_data("初始位置")
    # -------------------------------------------------------------------------------------

    # 初始化完成后拍照（调用通用函数）
    patrol.take_photo(
        photo_prefix="init",
        x=patrol.initial_point_[0],
        y=patrol.initial_point_[1]
    )

    # 只生成一次目标点列表，后续循环复用
    target_points_text = "正在生成巡检目标"
    patrol.get_logger().info(target_points_text)
    synthesize_and_play(target_points_text)
    target_points_list = patrol.get_target_points()  
    if not target_points_list:
        error_text = '未生成任何目标点，程序退出'
        patrol.get_logger().error(error_text)
        synthesize_and_play(error_text)
        # 关闭传感器（新增）
        patrol.sensor_reader.close()
        patrol.camera_saver.destroy_node()
        rclpy.shutdown()
        return

    while rclpy.ok():
        for idx, point in enumerate(target_points_list):
            x, y, yaw = point[0], point[1], point[2]
            
            # 前往目标点：日志+语音
            go_text = f"前往下一个目标点，坐标为{x:.2f}，{y:.2f}"
            patrol.get_logger().info(go_text)
            synthesize_and_play(go_text)
            
            # 导航到目标点
            target_pose = patrol.get_pose_by_xyyaw(x, y, yaw)
            result = patrol.nav_to_pose(target_pose)
            
            # 到达目标点后处理
            if result == TaskResult.SUCCEEDED:
                arrive_text = f"已到达目标点，坐标为{x:.2f}，{y:.2f}"
                patrol.get_logger().info(arrive_text)
                synthesize_and_play(arrive_text)
                
                # ---------------------- 播报目标点传感器数据（新增） ----------------------
                patrol.broadcast_sensor_data(f"目标点{x:.2f}，{y:.2f}")
                # -------------------------------------------------------------------------
                
                # 目标点拍照（调用通用函数）
                patrol.take_photo(
                    photo_prefix="patrol",
                    x=x,
                    y=y
                )
        
        # 完成一轮遍历：日志+语音
        loop_text = "已完成所有目标点遍历，将再次循环"
        patrol.get_logger().info(loop_text)
        synthesize_and_play(loop_text)
    
    # 优雅退出（新增传感器关闭）
    patrol.sensor_reader.close()  # 关闭传感器
    patrol.camera_saver.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
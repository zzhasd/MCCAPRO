import rclpy
import time
import threading
import requests
import yaml
import os
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from tf2_ros import TransformListener, Buffer
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from rclpy.duration import Duration

try:
    from nav2_msgs.msg import SpeedLimit
    HAS_NAV2_MSGS = True
except ImportError:
    HAS_NAV2_MSGS = False

# 优先使用源码目录绝对路径，保证修改 yaml 后即时生效
CONFIG_FILE_PATH = '/home/jetson/chapt7_ws/src/autopatrol_robot/autopatrol_robot/config.yaml'

class PatrolNode(BasicNavigator):
    def __init__(self, node_name='patrol_node'):
        super().__init__(node_name)
        
        # ----------------- 加载 YAML 配置 -----------------
        target_path = CONFIG_FILE_PATH
        if not os.path.exists(target_path):
            # 备用路径：如果绝对路径不存在，取本脚本同级目录的 config.yaml
            target_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')

        try:
            with open(target_path, 'r', encoding='utf-8') as f:
                self.global_config = yaml.safe_load(f)
                self.get_logger().info(f"读取yaml成功: {target_path}")
        except Exception as e:
            self.get_logger().error(f"读取yaml失败: {e}，将使用系统默认参数")
            self.global_config = {}

        # ----------------- 导航相关定义 -----------------
        self.declare_parameter('initial_point', [0.0, 0.0, 0.0])
        
        # 将 robot_id 的默认值交由 yaml 提供
        yaml_robot_id = self.global_config.get('robot', {}).get('robot_id', 0)
        self.declare_parameter('robot_id', yaml_robot_id)
        
        self.initial_point_ = self.get_parameter('initial_point').value
        self.robot_id_ = self.get_parameter('robot_id').value 
        
        self.current_x = self.initial_point_[0]
        self.current_y = self.initial_point_[1]
        
        self.buffer_ = Buffer()
        self.listener_ = TransformListener(self.buffer_, self)

        self.current_weights = [0.5, 0.5]
        self.weights_changed_flag = False 

        # ----------------- 速度与状态管理 -----------------
        self.current_speed_gear = 4 
        self.returning_home = False  # 标记是否正在执行低电量返航
        
        if HAS_NAV2_MSGS:
            self.speed_pub = self.create_publisher(SpeedLimit, '/speed_limit', 10)
        else:
            self.speed_pub = None
            self.get_logger().warn("未找到 nav2_msgs 包，调速功能将无法生效！")

        # 将 server_url 的值交由 yaml 提供
        self.server_url = self.global_config.get('network', {}).get('server_url', "http://10.201.126.178:9999/report")
        
        self.report_thread = threading.Thread(target=self.report_position_loop, daemon=True)
        self.report_thread.start()

    def set_speed_limit(self, percentage):
        if self.speed_pub:
            msg = SpeedLimit()
            msg.percentage = True
            msg.speed_limit = float(percentage)
            self.speed_pub.publish(msg)

    def report_position_loop(self):
        while rclpy.ok():
            time.sleep(1.0)
            try:
                transform = self.get_current_pose()
                if transform:
                    self.current_x = transform.translation.x
                    self.current_y = transform.translation.y
                    
                    payload = {"id": self.robot_id_, "x": self.current_x, "y": self.current_y}
                    response = requests.post(self.server_url, json=payload, timeout=2.0)
                    
                    if response.status_code == 200:
                        data = response.json()
                        new_weights = data.get("weights")
                        new_speed = data.get("speed")
                        
                        # ----- 1. 速度调控 (优先级最高) -----
                        if new_speed is not None and new_speed != self.current_speed_gear:
                            self.get_logger().warn(f"检测到速度挡位变化: {self.current_speed_gear} -> {new_speed}")
                            self.current_speed_gear = new_speed
                            
                            if new_speed == -1:
                                self.get_logger().error("电量不足10%，触发强制下线！中断当前任务！")
                                self.cancelTask()
                            elif new_speed == 0:
                                self.get_logger().error("收到速度挡位 0，立即中断当前任务，原地待命！")
                                self.cancelTask()
                            elif new_speed == 2:
                                self.get_logger().info("收到速度挡位 2，调整最高速度上限为 50%。")
                                self.set_speed_limit(50.0)
                            elif new_speed == 4:
                                self.get_logger().info("收到速度挡位 4，恢复默认全速 100%。")
                                self.set_speed_limit(100.0)

                        # ----- 2. 权重/路径调控 -----
                        # 如果已经由于没电下线，则不再理会权重变更
                        if self.current_speed_gear != -1:
                            if new_weights and len(new_weights) == len(self.current_weights):
                                diff = sum(abs(a - b) for a, b in zip(self.current_weights, new_weights))
                                if diff > 1e-4:  
                                    self.get_logger().warn(f"权重变化: {self.current_weights} -> {new_weights}，重规路径！")
                                    self.current_weights = new_weights
                                    self.weights_changed_flag = True
                                    if self.current_speed_gear != 0:
                                        self.cancelTask()

            except requests.exceptions.RequestException:
                pass
            except Exception as e:
                self.get_logger().warn(f'位置上报线程异常: {str(e)}')

    def get_pose_by_xyyaw(self, x, y, yaw):
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
        self.initial_point_ = self.get_parameter('initial_point').value
        self.setInitialPose(self.get_pose_by_xyyaw(
            self.initial_point_[0], self.initial_point_[1], self.initial_point_[2]))
        self.waitUntilNav2Active()

    def get_target_points(self):
        from .generate_target_points import generate_target_points
        current_pos = (self.current_x, self.current_y)
        self.get_logger().info(f'获取机器人 {self.robot_id_} 的目标点，参考起点: {current_pos}')
        points = generate_target_points(self.robot_id_, self.current_weights, current_pos)
        return points

    def nav_to_pose(self, target_pose):
        self.waitUntilNav2Active()
        result = self.goToPose(target_pose)
        while not self.isTaskComplete():
            feedback = self.getFeedback()
        
        result = self.getResult()
        if result == TaskResult.SUCCEEDED:
            self.get_logger().info('导航结果：成功')
        elif result == TaskResult.CANCELED:
            self.get_logger().warn('导航结果：被取消 (可能是由于调速/下线/权重变更)')
        else:
            self.get_logger().error('导航结果：失败')
        return result

    def get_current_pose(self):
        while rclpy.ok():
            try:
                tf = self.buffer_.lookup_transform('map', 'base_footprint', rclpy.time.Time(seconds=0), rclpy.time.Duration(seconds=1))
                return tf.transform
            except Exception as e:
                time.sleep(0.5) 

def main():
    rclpy.init()
    patrol = PatrolNode()

    patrol.get_logger().info("正在初始化位置")
    patrol.init_robot_pose()
    patrol.set_speed_limit(100.0)

    while rclpy.ok():
        # 【下线返航逻辑】
        if patrol.current_speed_gear == -1:
            if not patrol.returning_home:
                patrol.get_logger().error(">>> 启动自动返航程序，目标：地图起点 (0, 0) <<<")
                patrol.set_speed_limit(100.0)  # 返航使用默认全速
                target_pose = patrol.get_pose_by_xyyaw(0.0, 0.0, 0.0)
                patrol.nav_to_pose(target_pose)
                patrol.returning_home = True
                patrol.get_logger().info("已到达起点，机器人进入下线休眠状态。")
            
            # 到达起点后，无限阻塞在此处，不再索要巡检任务
            time.sleep(1.0)
            continue

        # 【0挡位原地待命阻塞】
        if patrol.current_speed_gear == 0:
            patrol.get_logger().info("当前速度挡位为 0，任务已中断，原地待命...", throttle_duration_sec=3.0)
            time.sleep(1.0)
            continue
            
        patrol.returning_home = False # 重置状态（万一被手动满血复活）
        patrol.weights_changed_flag = False 
        target_points_list = patrol.get_target_points()  
        
        if not target_points_list:
            break

        for idx, point in enumerate(target_points_list):
            if patrol.weights_changed_flag or patrol.current_speed_gear in [0, -1]:
                break
                
            x, y, yaw = point[0], point[1], point[2]
            patrol.get_logger().info(f"前往下一个目标点，坐标为{x:.2f}，{y:.2f}")
            
            target_pose = patrol.get_pose_by_xyyaw(x, y, yaw)
            result = patrol.nav_to_pose(target_pose)
            
            if patrol.current_speed_gear == 0:
                patrol.get_logger().info("巡检由于收到 0 挡位命令被强制打断！")
                break
            if patrol.current_speed_gear == -1:
                break
            if patrol.weights_changed_flag:
                break
            
            if result == TaskResult.SUCCEEDED:
                patrol.get_logger().info(f"已到达目标点，坐标为{x:.2f}，{y:.2f}")
        
        if patrol.weights_changed_flag:
            patrol.get_logger().info("旧的巡检由于权重变化被打断，重新开始巡检...")
            continue
            
        if patrol.current_speed_gear not in [0, -1]:
            patrol.get_logger().info("已完成所有目标点遍历，将再次循环")
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()
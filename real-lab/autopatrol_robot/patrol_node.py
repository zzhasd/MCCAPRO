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

try:
    from .runtime_config import config_file_path
except ImportError:
    from runtime_config import config_file_path

CONFIG_FILE_PATH = str(config_file_path())

class PatrolNode(BasicNavigator):
    def __init__(self, node_name='patrol_node'):
        super().__init__(node_name)
        
        # ----------------- Load YAML configuration -----------------
        target_path = CONFIG_FILE_PATH
        if not os.path.exists(target_path):
            # Fallback: if the absolute path is missing, use the adjacent config.yaml
            target_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')

        try:
            with open(target_path, 'r', encoding='utf-8') as f:
                self.global_config = yaml.safe_load(f)
                self.get_logger().info(f"YAML loaded successfully: {target_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load YAML: {e}; using system defaults")
            self.global_config = {}

        # ----------------- Navigation settings -----------------
        self.declare_parameter('initial_point', [0.0, 0.0, 0.0])
        
        # Read the default robot_id from YAML
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

        # ----------------- Speed and state management -----------------
        self.current_speed_gear = 4 
        self.returning_home = False  # Track whether a low-battery return is in progress
        
        if HAS_NAV2_MSGS:
            self.speed_pub = self.create_publisher(SpeedLimit, '/speed_limit', 10)
        else:
            self.speed_pub = None
            self.get_logger().warn("nav2_msgs package not found; speed control will be unavailable!")

        # Read server_url from YAML
        self.server_url = self.global_config.get('network', {}).get('server_url', "http://127.0.0.1:9999/report")
        
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
                        
                        # ----- 1. Speed control (highest priority) -----
                        if new_speed is not None and new_speed != self.current_speed_gear:
                            self.get_logger().warn(f"Speed setting changed: {self.current_speed_gear} -> {new_speed}")
                            self.current_speed_gear = new_speed
                            
                            if new_speed == -1:
                                self.get_logger().error("Battery below 10%; forcing offline and interrupting the current task!")
                                self.cancelTask()
                            elif new_speed == 0:
                                self.get_logger().error("Received speed setting 0; interrupting the task and holding position!")
                                self.cancelTask()
                            elif new_speed == 2:
                                self.get_logger().info("Received speed setting 2; setting the maximum speed to 50%.")
                                self.set_speed_limit(50.0)
                            elif new_speed == 4:
                                self.get_logger().info("Received speed setting 4; restoring full speed to 100%.")
                                self.set_speed_limit(100.0)

                        # ----- 2. Weight/path control -----
                        # Ignore weight changes after a battery-triggered offline transition
                        if self.current_speed_gear != -1:
                            if new_weights and len(new_weights) == len(self.current_weights):
                                diff = sum(abs(a - b) for a, b in zip(self.current_weights, new_weights))
                                if diff > 1e-4:  
                                    self.get_logger().warn(f"Weights changed: {self.current_weights} -> {new_weights}; replanning!")
                                    self.current_weights = new_weights
                                    self.weights_changed_flag = True
                                    if self.current_speed_gear != 0:
                                        self.cancelTask()

            except requests.exceptions.RequestException:
                pass
            except Exception as e:
                self.get_logger().warn(f'Position reporting thread error: {str(e)}')

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
        self.get_logger().info(f'Getting targets for robot {self.robot_id_}; reference start: {current_pos}')
        points = generate_target_points(self.robot_id_, self.current_weights, current_pos)
        return points

    def nav_to_pose(self, target_pose):
        self.waitUntilNav2Active()
        result = self.goToPose(target_pose)
        while not self.isTaskComplete():
            feedback = self.getFeedback()
        
        result = self.getResult()
        if result == TaskResult.SUCCEEDED:
            self.get_logger().info('Navigation result: succeeded')
        elif result == TaskResult.CANCELED:
            self.get_logger().warn('Navigation result: canceled (possibly due to speed, offline status, or weight changes)')
        else:
            self.get_logger().error('Navigation result: failed')
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

    patrol.get_logger().info("Initializing position")
    patrol.init_robot_pose()
    patrol.set_speed_limit(100.0)

    while rclpy.ok():
        # [Offline return-to-start logic]
        if patrol.current_speed_gear == -1:
            if not patrol.returning_home:
                patrol.get_logger().error(">>> Starting automatic return; target: map origin (0, 0) <<<")
                patrol.set_speed_limit(100.0)  # Return at the default full speed
                target_pose = patrol.get_pose_by_xyyaw(0.0, 0.0, 0.0)
                patrol.nav_to_pose(target_pose)
                patrol.returning_home = True
                patrol.get_logger().info("Reached the start; robot entering offline sleep mode.")
            
            # Block indefinitely after reaching the start; do not request further patrol tasks
            time.sleep(1.0)
            continue

        # [0 Block and hold position at this speed setting]
        if patrol.current_speed_gear == 0:
            patrol.get_logger().info("Current speed setting is 0; task interrupted, holding position...", throttle_duration_sec=3.0)
            time.sleep(1.0)
            continue
            
        patrol.returning_home = False # Reset state (in case of manual reactivation)
        patrol.weights_changed_flag = False 
        target_points_list = patrol.get_target_points()  
        
        if not target_points_list:
            break

        for idx, point in enumerate(target_points_list):
            if patrol.weights_changed_flag or patrol.current_speed_gear in [0, -1]:
                break
                
            x, y, yaw = point[0], point[1], point[2]
            patrol.get_logger().info(f"Moving to the next target at{x:.2f}, {y:.2f}")
            
            target_pose = patrol.get_pose_by_xyyaw(x, y, yaw)
            result = patrol.nav_to_pose(target_pose)
            
            if patrol.current_speed_gear == 0:
                patrol.get_logger().info("Patrol interrupted by speed setting 0!")
                break
            if patrol.current_speed_gear == -1:
                break
            if patrol.weights_changed_flag:
                break
            
            if result == TaskResult.SUCCEEDED:
                patrol.get_logger().info(f"Reached the target at{x:.2f}, {y:.2f}")
        
        if patrol.weights_changed_flag:
            patrol.get_logger().info("Previous patrol interrupted by a weight change; restarting patrol...")
            continue
            
        if patrol.current_speed_gear not in [0, -1]:
            patrol.get_logger().info("All targets visited; starting another cycle")
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()

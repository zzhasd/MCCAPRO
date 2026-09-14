import rclpy
import time  # Used to generate timestamps
from geometry_msgs.msg import PoseStamped, Pose
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from tf2_ros import TransformListener, Buffer
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from rclpy.duration import Duration
# Import TTS utilities
from .tts_utils import synthesize_and_play
# Import the camera capture utility
from .camera_utils import CameraImageSaver
# Import sensor utilities
from .sensor_utils import SensorReader  # New
from .runtime_config import image_directory

class PatrolNode(BasicNavigator):
    def __init__(self, node_name='patrol_node'):
        super().__init__(node_name)
        # Navigation settings
        self.declare_parameter('initial_point', [0.0, 0.0, 0.0])
        self.declare_parameter('target_points', [0.0, 0.0, 0.0, 1.0, 1.0, 1.57])
        self.initial_point_ = self.get_parameter('initial_point').value
        self.target_points_ = self.get_parameter('target_points').value
        # TF settings for live position tracking
        self.buffer_ = Buffer()
        self.listener_ = TransformListener(self.buffer_, self)
        # Initialize the camera-saving node (subscribe early to avoid waiting during capture)
        self.camera_saver = CameraImageSaver()
        
        # Initialize the sensor reader (new)
        self.sensor_reader = SensorReader(logger=self.get_logger())  # Pass the ROS2 logger
        self.sensor_reader.init_sensors()  # Initialize sensors

    def get_pose_by_xyyaw(self, x, y, yaw):
        """Convert x,y,yaw to PoseStamped"""
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
        """Initialize the robot pose"""
        self.initial_point_ = self.get_parameter('initial_point').value
        self.setInitialPose(self.get_pose_by_xyyaw(
            self.initial_point_[0], self.initial_point_[1], self.initial_point_[2]))
        self.waitUntilNav2Active()

    def get_target_points(self):
        """Get random targets through generate_target_points"""
        from .generate_target_points_PCA import generate_target_points
        self.get_logger().info('Generating random targets...')
        target_dir = image_directory()
        points = generate_target_points(image_dir=target_dir)
        for index, point in enumerate(points):
            self.get_logger().info(f'Received target: {index}->({point[0]:.2f},{point[1]:.2f},{point[2]:.2f})')
        return points

    def nav_to_pose(self, target_pose):
        """Navigate to the specified pose and return the result"""
        self.waitUntilNav2Active()
        result = self.goToPose(target_pose)
        while not self.isTaskComplete():
            feedback = self.getFeedback()
            if feedback:
                self.get_logger().info(f'Estimated arrival in {Duration.from_msg(feedback.estimated_time_remaining).nanoseconds / 1e9} s')
        # Check the final result
        result = self.getResult()
        if result == TaskResult.SUCCEEDED:
            self.get_logger().info('Navigation result: succeeded')
        elif result == TaskResult.CANCELED:
            self.get_logger().warn('Navigation result: canceled')
        elif result == TaskResult.FAILED:
            self.get_logger().error('Navigation result: failed')
        else:
            self.get_logger().error('Navigation result: invalid return status')
        return result

    def get_current_pose(self):
        """Get the current pose through TF"""
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
                    f'Translation:{transform.translation}, rotation quaternion:{transform.rotation}: Euler angles:{rotation_euler}')
                return transform
            except Exception as e:
                self.get_logger().warn(f'Unable to obtain the coordinate transform; reason: {str(e)}')

    # ---------------------- Encapsulate the shared photo function (core refactor) ----------------------
    def take_photo(self, photo_prefix: str, x: float, y: float) -> bool:
        """
        Shared photo function: encapsulate capture logic and eliminate duplication
        :param photo_prefix: Photo filename prefix (such as init/patrol) to identify the photo type
        :param x: Capture x coordinate (used in the filename)
        :param y: Capture y coordinate (used in the filename)
        :return: Whether capture succeeded
        """
        # 1. Announce that a photo is being taken
        taking_photo_text = "Taking a photo"
        self.get_logger().info(taking_photo_text)
        synthesize_and_play(taking_photo_text)

        # 2. Generate a filename with a timestamp and coordinates
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        photo_filename = f"{photo_prefix}_{timestamp}_x{x:.2f}_y{y:.2f}.png"

        # 3. Wait for a camera image and save it
        if self.camera_saver.wait_for_image(timeout=10.0):
            save_success = self.camera_saver.save_image(filename=photo_filename)
            if save_success:
                # Announce successful capture
                photo_done_text = "Photo taken"
                self.get_logger().info(photo_done_text)
                synthesize_and_play(photo_done_text)
                return True
            else:
                # Announce capture failure
                photo_fail_text = "Photo capture failed"
                self.get_logger().error(photo_fail_text)
                synthesize_and_play(photo_fail_text)
                return False
        else:
            # Announce image timeout
            photo_timeout_text = "Timed out receiving a camera image; capture failed"
            self.get_logger().error(photo_timeout_text)
            synthesize_and_play(photo_timeout_text)
            return False
    
    # ---------------------- New: sensor announcement function ----------------------
    def broadcast_sensor_data(self, location_desc: str):
        """
        Read and announce sensor data
        :param location_desc: Location description (such as "initial position" or "target(1.0,2.0)")
        """
        if not self.sensor_reader.is_initialized:
            self.get_logger().warn("Sensors are not initialized; cannot announce temperature and humidity")
            return
        
        # Read sensor data
        sensor_data = self.sensor_reader.read_all_sensors()
        
        # Construct announcement text
        if sensor_data['temperature'] is not None and sensor_data['humidity'] is not None:
            broadcast_text = f"{location_desc}Current temperature {sensor_data['temperature']}, current humidity{sensor_data['humidity']}"
        else:
            broadcast_text = f"{location_desc}Failed to read temperature and humidity"
        
        # Log output and speech announcement
        self.get_logger().info(f"Sensor announcement: {broadcast_text}")
        synthesize_and_play(broadcast_text)
        
        # Optional: log complete sensor data (including CO2 and TVOC)
        self.get_logger().info(
            f"Full sensor data - temperature: {sensor_data['temperature']}°C, "
            f"humidity: {sensor_data['humidity']}%, "
            f"eCO2: {sensor_data['eco2']}ppm, "
            f"TVOC: {sensor_data['tvoc']}ppb"
        )
    # -------------------------------------------------------------------------

def main():
    rclpy.init()
    patrol = PatrolNode()
    
    # Position initialization: log and speech
    init_text = "Initializing position"
    patrol.get_logger().info(init_text)
    synthesize_and_play(init_text)
    
    patrol.init_robot_pose()
    
    # Initialization complete: log and speech
    init_complete_text = "Position initialized"
    patrol.get_logger().info(init_complete_text)
    synthesize_and_play(init_complete_text)

    # ---------------------- Announce sensor data at the initial position after initialization (new) ----------------------
    patrol.broadcast_sensor_data("Initial position")
    # -------------------------------------------------------------------------------------

    # Take a photo after initialization (call the shared function)
    patrol.take_photo(
        photo_prefix="init",
        x=patrol.initial_point_[0],
        y=patrol.initial_point_[1]
    )

    # Generate the target list once and reuse it in subsequent cycles
    target_points_text = "Generating patrol targets"
    patrol.get_logger().info(target_points_text)
    synthesize_and_play(target_points_text)
    target_points_list = patrol.get_target_points()  
    if not target_points_list:
        error_text = 'No targets generated; exiting'
        patrol.get_logger().error(error_text)
        synthesize_and_play(error_text)
        # Close sensors (new)
        patrol.sensor_reader.close()
        patrol.camera_saver.destroy_node()
        rclpy.shutdown()
        return

    while rclpy.ok():
        for idx, point in enumerate(target_points_list):
            x, y, yaw = point[0], point[1], point[2]
            
            # Moving to a target: log and speech
            go_text = f"Moving to the next target at{x:.2f}, {y:.2f}"
            patrol.get_logger().info(go_text)
            synthesize_and_play(go_text)
            
            # Navigate to the target
            target_pose = patrol.get_pose_by_xyyaw(x, y, yaw)
            result = patrol.nav_to_pose(target_pose)
            
            # Handle arrival at the target
            if result == TaskResult.SUCCEEDED:
                arrive_text = f"Reached the target at{x:.2f}, {y:.2f}"
                patrol.get_logger().info(arrive_text)
                synthesize_and_play(arrive_text)
                
                # ---------------------- Announce sensor data at the target (new) ----------------------
                patrol.broadcast_sensor_data(f"Target{x:.2f}, {y:.2f}")
                # -------------------------------------------------------------------------
                
                # Take a photo at the target (call the shared function)
                patrol.take_photo(
                    photo_prefix="patrol",
                    x=x,
                    y=y
                )
        
        # Complete one traversal: log and speech
        loop_text = "All targets visited; starting another cycle"
        patrol.get_logger().info(loop_text)
        synthesize_and_play(loop_text)
    
    # Graceful exit (including sensor shutdown)
    patrol.sensor_reader.close()  # Close sensors
    patrol.camera_saver.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

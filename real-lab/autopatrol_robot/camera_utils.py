import os
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

try:
    from .runtime_config import image_directory
except ImportError:
    from runtime_config import image_directory

class CameraImageSaver(Node):
    """
    Minimal camera-image saving interface
    Workflow: subscribe to the camera topic -> wait for an image -> save to the specified path (no display window)
    """
    def __init__(self):
        super().__init__("camera_image_saver_node")
        
        # 1. Core settings (adjust as needed)
        self.camera_topic = "/camera/color/image_raw"  # Camera image topic
        self.save_dir = image_directory()
        self.image_filename = "camera_capture.png"  # Default output filename
        self.bridge = CvBridge()  # ROS image <-> OpenCV image converter
        self.latest_image = None  # Cache the latest camera image
        
        # 2. Create the output directory automatically (avoid missing-path errors)
        self._create_save_dir()
        
        # 3. Subscribe to the camera topic for live images
        self.image_sub = self.create_subscription(
            Image,
            self.camera_topic,
            self.image_callback,  # Image callback
            1  # [Fix]: Set queue size to 1 to discard old images captured during movement
        )
        self.get_logger().info(f"Subscribed to camera topic: {self.camera_topic}")
        self.get_logger().info(f"Images will be saved to: {self.save_dir}")

    def _create_save_dir(self):
        """Create the output directory automatically (if it does not exist)"""
        if not os.path.exists(self.save_dir):
            try:
                os.makedirs(self.save_dir)
                self.get_logger().info(f"Output directory created: {self.save_dir}")
            except Exception as e:
                self.get_logger().error(f"Failed to create output directory: {str(e)}")
                raise  # Stop the node if directory creation fails

    def image_callback(self, msg: Image):
        """Camera callback: cache the latest image"""
        try:
            # Convert the ROS Image message to OpenCV format (BGR8 is the standard RGB-camera format)
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Image format conversion failed: {str(e)}")

    def wait_for_image(self, timeout=10.0):
        """
        Wait for a camera image (exit on timeout)
        :param timeout: Maximum wait time (seconds)
        :return: bool - Whether an image was received
        """
        # [Fix]: Clear the previous image so every call requests the latest frame
        self.latest_image = None  

        start_time = time.time()
        self.get_logger().info("Waiting for the latest camera image...")
        
        while self.latest_image is None:
            # Check for timeout
            if time.time() - start_time > timeout:
                self.get_logger().error(f"Timed out waiting for an image ({timeout} s); check the camera!")
                return False
            # Let ROS process callbacks (spin_once is required for callbacks to run)
            rclpy.spin_once(self, timeout_sec=0.1)
        
        self.get_logger().info("Latest camera image received!")
        return True

    def save_image(self, filename=None):
        """
        Public interface: save the current camera frame to the specified path
        :param filename: Custom filename (optional; defaults to self.image_filename)
        :return: bool - Whether saving succeeded
        """
        if filename is None:
            filename = self.image_filename
        
        save_path = os.path.join(self.save_dir, filename)
        
        # Check for a cached image
        if self.latest_image is None:
            self.get_logger().error("No camera image received! Check the camera and topic settings")
            return False
        
        # Save the image to the specified path
        try:
            cv2.imwrite(save_path, self.latest_image)
            self.get_logger().info(f"Image saved successfully: {save_path}")
            return True
        except Exception as e:
            self.get_logger().error(f"Failed to save image: {str(e)}")
            return False

# ------------------- Usage example -------------------
def main(args=None):
    # Initialize ROS2
    rclpy.init(args=args)
    
    # Create the camera-saving node
    camera_saver = CameraImageSaver()
    
    # Wait for an image before saving it
    if camera_saver.wait_for_image(timeout=10.0):
        # Option 1: use the default filename
        camera_saver.save_image()
        
        # Option 2: use a custom filename (optional)
        # timestamp = time.strftime("%Y%m%d_%H%M%S")
        # camera_saver.save_image(f"capture_{timestamp}.png")
    
    # Destroy the node and shut down ROS2
    camera_saver.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

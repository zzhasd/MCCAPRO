import rclpy
import time
from rclpy.node import Node

# Import sensor utilities (keep the original import)
from sensor_utils import SensorReader  

class SensorTestNode(Node):
    def __init__(self, node_name='sensor_test_node'):
        super().__init__(node_name)
        
        # Initialize the sensor reader
        self.sensor_reader = SensorReader(logger=self.get_logger())
        self.sensor_reader.init_sensors()
        
        # Check sensor initialization
        if self.sensor_reader.is_initialized:
            self.get_logger().info("Sensors initialized; reading data every second...")
            # Main change: create a ROS2 timer triggered every second (replacing rate.sleep())
            self.timer = self.create_timer(1.0, self.read_and_print_sensor_data)
        else:
            self.get_logger().error("Sensor initialization failed!")
            raise RuntimeError("Sensor initialization failed; cannot continue the test")

    def read_and_print_sensor_data(self):
        """Read sensor data and print formatted output"""
        sensor_data = self.sensor_reader.read_all_sensors()
        
        # Format output (preserve the original logic)
        self.get_logger().info("="*50)
        self.get_logger().info(f"Sensor data [{time.strftime('%Y-%m-%d %H:%M:%S')}]")
        self.get_logger().info(f"  Temperature: {sensor_data['temperature'] or 'read failed'} °C")
        self.get_logger().info(f"  Humidity: {sensor_data['humidity'] or 'read failed'} %")
        self.get_logger().info(f"  eCO2: {sensor_data['eco2'] or 'read failed'} ppm")
        self.get_logger().info(f"  TVOC: {sensor_data['tvoc'] or 'read failed'} ppb")
        self.get_logger().info("="*50)

    def close(self):
        """Shut down sensors gracefully"""
        self.sensor_reader.close()
        self.get_logger().info("Sensors closed; test finished")

def main():
    rclpy.init()
    
    try:
        # Create the test node
        sensor_test = SensorTestNode()
        
        # Main change: start the node executor (to drive the timer)
        rclpy.spin(sensor_test)
    
    except KeyboardInterrupt:
        sensor_test.get_logger().info("\n Exit signal received; shutting down...")
    except RuntimeError as e:
        sensor_test.get_logger().error(f"Runtime error: {e}")
    finally:
        if 'sensor_test' in locals():
            sensor_test.close()
            sensor_test.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
import rclpy
import time
from rclpy.node import Node

# 导入传感器工具类（保持你的原有导入）
from sensor_utils import SensorReader  

class SensorTestNode(Node):
    def __init__(self, node_name='sensor_test_node'):
        super().__init__(node_name)
        
        # 初始化传感器读取器
        self.sensor_reader = SensorReader(logger=self.get_logger())
        self.sensor_reader.init_sensors()
        
        # 检查传感器初始化状态
        if self.sensor_reader.is_initialized:
            self.get_logger().info("传感器初始化成功，开始每秒读取数据...")
            # 核心修改：创建ROS2定时器，1秒触发一次（替代rate.sleep()）
            self.timer = self.create_timer(1.0, self.read_and_print_sensor_data)
        else:
            self.get_logger().error("传感器初始化失败！")
            raise RuntimeError("传感器初始化失败，测试无法继续")

    def read_and_print_sensor_data(self):
        """读取传感器数据并格式化打印"""
        sensor_data = self.sensor_reader.read_all_sensors()
        
        # 格式化打印（保持原有逻辑）
        self.get_logger().info("="*50)
        self.get_logger().info(f"传感器数据 [{time.strftime('%Y-%m-%d %H:%M:%S')}]")
        self.get_logger().info(f"  温度: {sensor_data['temperature'] or '读取失败'} °C")
        self.get_logger().info(f"  湿度: {sensor_data['humidity'] or '读取失败'} %")
        self.get_logger().info(f"  eCO2: {sensor_data['eco2'] or '读取失败'} ppm")
        self.get_logger().info(f"  TVOC: {sensor_data['tvoc'] or '读取失败'} ppb")
        self.get_logger().info("="*50)

    def close(self):
        """优雅关闭传感器"""
        self.sensor_reader.close()
        self.get_logger().info("传感器已关闭，测试结束")

def main():
    rclpy.init()
    
    try:
        # 创建测试节点
        sensor_test = SensorTestNode()
        
        # 核心修改：启动节点的执行器（驱动定时器运行）
        rclpy.spin(sensor_test)
    
    except KeyboardInterrupt:
        sensor_test.get_logger().info("\n接收到退出信号，正在关闭...")
    except RuntimeError as e:
        sensor_test.get_logger().error(f"运行错误: {e}")
    finally:
        if 'sensor_test' in locals():
            sensor_test.close()
            sensor_test.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
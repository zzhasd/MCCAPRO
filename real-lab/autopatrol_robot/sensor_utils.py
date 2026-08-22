# sensor_utils.py
import smbus2
import time
import rclpy

class SensorReader:
    """温湿度、气体传感器读取工具类"""
    # I2C 总线和设备地址定义（类属性）
    I2C_BUS = 1
    AHT21_ADDR = 0x38
    ENS160_ADDR = 0x53

    def __init__(self, logger=None):
        """
        初始化传感器读取器
        :param logger: ROS2 节点日志器（可选），用于输出传感器日志
        """
        self.logger = logger
        self.bus = None
        self.is_initialized = False

    def init_sensors(self):
        """初始化所有传感器"""
        try:
            # 修复：直接使用类属性 I2C_BUS，无需 self.SensorReader
            self.bus = smbus2.SMBus(self.I2C_BUS)
            
            # 初始化 AHT21 温湿度传感器
            self._init_aht21()
            
            # 初始化 ENS160 气体传感器
            self._init_ens160()
            
            self.is_initialized = True
            if self.logger:
                self.logger.info("所有传感器初始化成功")
            else:
                print("所有传感器初始化成功")
                
        except Exception as e:
            error_msg = f"传感器初始化失败: {str(e)}"
            if self.logger:
                self.logger.error(error_msg)
            else:
                print(error_msg)
            self.is_initialized = False

    def _init_aht21(self):
        """内部方法：初始化 AHT21"""
        try:
            # 修复：直接使用 self.AHT21_ADDR
            self.bus.write_i2c_block_data(self.AHT21_ADDR, 0xBE, [0x08, 0x00])
            time.sleep(0.05)
            if self.logger:
                self.logger.info("AHT21 温湿度传感器初始化成功")
        except Exception as e:
            warn_msg = f"AHT21 初始化警告: {str(e)}"
            if self.logger:
                self.logger.warn(warn_msg)
            else:
                print(warn_msg)

    def _init_ens160(self):
        """内部方法：初始化 ENS160"""
        try:
            # 修复：直接使用 self.ENS160_ADDR
            # 软复位
            self.bus.write_byte_data(self.ENS160_ADDR, 0x10, 0xF0)
            time.sleep(0.1)
            # 标准工作模式
            self.bus.write_byte_data(self.ENS160_ADDR, 0x10, 0x02)
            time.sleep(0.1)
            if self.logger:
                self.logger.info("ENS160 气体传感器初始化成功")
        except Exception as e:
            error_msg = f"ENS160 初始化失败: {str(e)}"
            if self.logger:
                self.logger.error(error_msg)
            else:
                print(error_msg)

    def read_all_sensors(self):
        """
        读取所有传感器数据
        :return: 字典格式数据 {
            'temperature': 温度值(°C) / None,
            'humidity': 湿度值(%) / None,
            'eco2': 二氧化碳浓度(ppm) / None,
            'tvoc': 总挥发性有机物(ppb) / None
        }
        """
        if not self.is_initialized or self.bus is None:
            error_msg = "传感器未初始化，无法读取数据"
            if self.logger:
                self.logger.error(error_msg)
            else:
                print(error_msg)
            return {
                'temperature': None,
                'humidity': None,
                'eco2': None,
                'tvoc': None
            }

        # 读取温湿度
        temp, hum = self._read_aht21()
        # 读取气体数据
        eco2, tvoc = self._read_ens160()

        return {
            'temperature': temp,
            'humidity': hum,
            'eco2': eco2,
            'tvoc': tvoc
        }

    def _read_aht21(self):
        """内部方法：读取 AHT21 温湿度"""
        try:
            # 修复：使用 self.AHT21_ADDR
            self.bus.write_i2c_block_data(self.AHT21_ADDR, 0xAC, [0x33, 0x00])
            time.sleep(0.08)  # 等待测量完成
            
            # 读取数据
            data = self.bus.read_i2c_block_data(self.AHT21_ADDR, 0x00, 6)
            
            # 解析数据（按官方手册公式）
            humidity = ((data[1] << 12) | (data[2] << 4) | (data[3] >> 4)) * 100.0 / 1048576.0
            temp = (((data[3] & 0x0F) << 16) | (data[4] << 8) | data[5]) * 200.0 / 1048576.0 - 50.0
            
            return round(temp, 1), round(humidity, 1)
        except Exception as e:
            error_msg = f"读取温湿度失败: {str(e)}"
            if self.logger:
                self.logger.warn(error_msg)
            else:
                print(error_msg)
            return None, None

    def _read_ens160(self):
        """内部方法：读取 ENS160 气体数据"""
        try:
            # 修复：使用 self.ENS160_ADDR
            # 读取 TVOC (小端模式)
            tvoc_data = self.bus.read_i2c_block_data(self.ENS160_ADDR, 0x22, 2)
            tvoc = tvoc_data[0] | (tvoc_data[1] << 8)
            
            # 读取 eCO2 (小端模式)
            eco2_data = self.bus.read_i2c_block_data(self.ENS160_ADDR, 0x24, 2)
            eco2 = eco2_data[0] | (eco2_data[1] << 8)
            
            return eco2, tvoc
        except Exception as e:
            error_msg = f"读取气体数据失败: {str(e)}"
            if self.logger:
                self.logger.warn(error_msg)
            else:
                print(error_msg)
            return None, None

    def close(self):
        """释放 I2C 总线资源"""
        if self.bus:
            try:
                self.bus.close()
                if self.logger:
                    self.logger.info("传感器 I2C 总线已关闭")
            except Exception as e:
                if self.logger:
                    self.logger.warn(f"关闭 I2C 总线失败: {str(e)}")
            finally:
                self.bus = None
                self.is_initialized = False

# 测试代码（单独运行时）
if __name__ == "__main__":
    sensor_reader = SensorReader()
    sensor_reader.init_sensors()
    
    if sensor_reader.is_initialized:
        print("开始读取传感器数据（按 Ctrl+C 退出）\n")
        print("-" * 70)
        try:
            while True:
                data = sensor_reader.read_all_sensors()
                
                # 格式化输出
                output = "\r"
                if data['temperature'] is not None and data['humidity'] is not None:
                    output += f"温度: {data['temperature']:5.1f} °C | 湿度: {data['humidity']:5.1f} %  ||  "
                else:
                    output += "温湿度: 读取失败   ||  "
                    
                if data['eco2'] is not None and data['tvoc'] is not None:
                    output += f"eCO2: {data['eco2']:4d} ppm | TVOC: {data['tvoc']:4d} ppb"
                else:
                    output += "气体: 读取失败"
                    
                print(output, end="")
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n\n程序退出中...")
        finally:
            sensor_reader.close()
    else:
        print("传感器初始化失败，程序退出")
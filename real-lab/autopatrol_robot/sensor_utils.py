# sensor_utils.py
import smbus2
import time
import rclpy

class SensorReader:
    """Temperature, humidity, and gas sensor reader"""
    # I2C Bus and device addresses (class attributes)
    I2C_BUS = 1
    AHT21_ADDR = 0x38
    ENS160_ADDR = 0x53

    def __init__(self, logger=None):
        """
        Initialize the sensor reader
        :param logger: ROS2 Optional node logger for sensor logs
        """
        self.logger = logger
        self.bus = None
        self.is_initialized = False

    def init_sensors(self):
        """Initialize all sensors"""
        try:
            # Fix: use the I2C_BUS class attribute directly; no need for self.SensorReader
            self.bus = smbus2.SMBus(self.I2C_BUS)
            
            # Initialize the AHT21 temperature/humidity sensor
            self._init_aht21()
            
            # Initialize the ENS160 gas sensor
            self._init_ens160()
            
            self.is_initialized = True
            if self.logger:
                self.logger.info("All sensors initialized successfully")
            else:
                print("All sensors initialized successfully")
                
        except Exception as e:
            error_msg = f"Sensor initialization failed: {str(e)}"
            if self.logger:
                self.logger.error(error_msg)
            else:
                print(error_msg)
            self.is_initialized = False

    def _init_aht21(self):
        """Internal method: initialize AHT21"""
        try:
            # Fix: directly use self.AHT21_ADDR
            self.bus.write_i2c_block_data(self.AHT21_ADDR, 0xBE, [0x08, 0x00])
            time.sleep(0.05)
            if self.logger:
                self.logger.info("AHT21 Temperature/humidity sensor initialized")
        except Exception as e:
            warn_msg = f"AHT21 initialization warning: {str(e)}"
            if self.logger:
                self.logger.warn(warn_msg)
            else:
                print(warn_msg)

    def _init_ens160(self):
        """Internal method: initialize ENS160"""
        try:
            # Fix: directly use self.ENS160_ADDR
            # Soft reset
            self.bus.write_byte_data(self.ENS160_ADDR, 0x10, 0xF0)
            time.sleep(0.1)
            # Standard operating mode
            self.bus.write_byte_data(self.ENS160_ADDR, 0x10, 0x02)
            time.sleep(0.1)
            if self.logger:
                self.logger.info("ENS160 Gas sensor initialized")
        except Exception as e:
            error_msg = f"ENS160 initialization failed: {str(e)}"
            if self.logger:
                self.logger.error(error_msg)
            else:
                print(error_msg)

    def read_all_sensors(self):
        """
        Read all sensor data
        :return: Data dictionary {
            'temperature': Temperature(°C) / None,
            'humidity': Humidity(%) / None,
            'eco2': Carbon dioxide concentration(ppm) / None,
            'tvoc': Total volatile organic compounds(ppb) / None
        }
        """
        if not self.is_initialized or self.bus is None:
            error_msg = "Sensors are not initialized; cannot read data"
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

        # Read temperature and humidity
        temp, hum = self._read_aht21()
        # Read gas data
        eco2, tvoc = self._read_ens160()

        return {
            'temperature': temp,
            'humidity': hum,
            'eco2': eco2,
            'tvoc': tvoc
        }

    def _read_aht21(self):
        """Internal method: read AHT21 temperature and humidity"""
        try:
            # Fix: use self.AHT21_ADDR
            self.bus.write_i2c_block_data(self.AHT21_ADDR, 0xAC, [0x33, 0x00])
            time.sleep(0.08)  # Wait for measurement completion
            
            # Read data
            data = self.bus.read_i2c_block_data(self.AHT21_ADDR, 0x00, 6)
            
            # Parse data (using formulas in the official manual)
            humidity = ((data[1] << 12) | (data[2] << 4) | (data[3] >> 4)) * 100.0 / 1048576.0
            temp = (((data[3] & 0x0F) << 16) | (data[4] << 8) | data[5]) * 200.0 / 1048576.0 - 50.0
            
            return round(temp, 1), round(humidity, 1)
        except Exception as e:
            error_msg = f"Failed to read temperature and humidity: {str(e)}"
            if self.logger:
                self.logger.warn(error_msg)
            else:
                print(error_msg)
            return None, None

    def _read_ens160(self):
        """Internal method: read ENS160 gas data"""
        try:
            # Fix: use self.ENS160_ADDR
            # Read TVOC (little-endian)
            tvoc_data = self.bus.read_i2c_block_data(self.ENS160_ADDR, 0x22, 2)
            tvoc = tvoc_data[0] | (tvoc_data[1] << 8)
            
            # Read eCO2 (little-endian)
            eco2_data = self.bus.read_i2c_block_data(self.ENS160_ADDR, 0x24, 2)
            eco2 = eco2_data[0] | (eco2_data[1] << 8)
            
            return eco2, tvoc
        except Exception as e:
            error_msg = f"Failed to read gas data: {str(e)}"
            if self.logger:
                self.logger.warn(error_msg)
            else:
                print(error_msg)
            return None, None

    def close(self):
        """Release I2C bus resources"""
        if self.bus:
            try:
                self.bus.close()
                if self.logger:
                    self.logger.info("Sensor I2C bus closed")
            except Exception as e:
                if self.logger:
                    self.logger.warn(f"Failed to close the I2C bus: {str(e)}")
            finally:
                self.bus = None
                self.is_initialized = False

# Test code (standalone execution)
if __name__ == "__main__":
    sensor_reader = SensorReader()
    sensor_reader.init_sensors()
    
    if sensor_reader.is_initialized:
        print("Start reading sensor data (press Ctrl+C to exit)\n")
        print("-" * 70)
        try:
            while True:
                data = sensor_reader.read_all_sensors()
                
                # Format output
                output = "\r"
                if data['temperature'] is not None and data['humidity'] is not None:
                    output += f"Temperature: {data['temperature']:5.1f} °C | Humidity: {data['humidity']:5.1f} %  ||  "
                else:
                    output += "Temperature/humidity: read failed   ||  "
                    
                if data['eco2'] is not None and data['tvoc'] is not None:
                    output += f"eCO2: {data['eco2']:4d} ppm | TVOC: {data['tvoc']:4d} ppb"
                else:
                    output += "Gas: read failed"
                    
                print(output, end="")
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n\n Exiting...")
        finally:
            sensor_reader.close()
    else:
        print("Sensor initialization failed; exiting")
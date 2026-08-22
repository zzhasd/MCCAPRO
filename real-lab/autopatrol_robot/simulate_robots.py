import requests
import time
import math
import threading

# 尝试导入你现有的路径生成器，严格遵守生成逻辑
try:
    from generate_target_points import generate_target_points
except ImportError:
    print("⚠️ 警告: 无法导入 generate_target_points，请确保它与此脚本在同一目录下。")
    print("⚠️ 将使用随机点生成作为回退兜底逻辑。")
    def generate_target_points(robot_id, weights, current_pos):
        import random
        return [[random.uniform(-3, 3), random.uniform(-3, 3), 0] for _ in range(5)]

# ================= 配置参数 =================
SERVER_URL = 'http://127.0.0.1:9999/report'
TARGET_ROBOT_ID = 0      # 模拟的机器人编号
MAX_SPEED = 0.6          # 模拟的全速直线移动速度 (米/秒)
SIM_DT = 0.05            # 物理模拟的时间步长 (秒) -> 越小移动越平滑
# ============================================

class SimulatedPatrolNode:
    def __init__(self, robot_id):
        self.robot_id = robot_id
        
        # 初始点为 (0,0)
        self.current_x = 0.0
        self.current_y = 0.0
        
        # 状态变量
        self.current_weights = [0.5, 0.5]
        self.weights_changed_flag = False 
        
        self.current_speed_gear = 4  # 默认全速
        self.returning_home = False  # 标记是否正在低电量返航
        
        # 启动后台心跳汇报线程
        self.report_thread = threading.Thread(target=self.report_position_loop, daemon=True)
        self.report_thread.start()

    def report_position_loop(self):
        """完全模拟 patrol_node.py 中的 report_position_loop"""
        while True:
            time.sleep(1.0) # 1Hz 上报频率
            payload = {"id": self.robot_id, "x": self.current_x, "y": self.current_y}
            
            try:
                response = requests.post(SERVER_URL, json=payload, timeout=2.0)
                if response.status_code == 200:
                    data = response.json()
                    new_weights = data.get("weights")
                    new_speed = data.get("speed")
                    
                    # ----- 1. 速度调控 (优先级最高) -----
                    if new_speed is not None and new_speed != self.current_speed_gear:
                        print(f"[Robot {self.robot_id}] 🚦 检测到速度挡位变化: {self.current_speed_gear} -> {new_speed}")
                        self.current_speed_gear = new_speed
                        
                        if new_speed == -1:
                            print(f"[Robot {self.robot_id}] ⚡ 电量不足10%，触发强制下线！中断当前任务！")
                        elif new_speed == 0:
                            print(f"[Robot {self.robot_id}] 🛑 收到速度挡位 0，立即中断当前任务，原地待命！")
                        elif new_speed == 2:
                            print(f"[Robot {self.robot_id}] 🐢 收到速度挡位 2，调整最高速度上限为 50%。")
                        elif new_speed == 4:
                            print(f"[Robot {self.robot_id}] 🐇 收到速度挡位 4，恢复默认全速 100%。")

                    # ----- 2. 权重/路径调控 -----
                    if self.current_speed_gear != -1:
                        if new_weights and len(new_weights) == len(self.current_weights):
                            diff = sum(abs(a - b) for a, b in zip(self.current_weights, new_weights))
                            if diff > 1e-4:  
                                formatted_old = [round(w, 3) for w in self.current_weights]
                                formatted_new = [round(w, 3) for w in new_weights]
                                print(f"[Robot {self.robot_id}] ⚖️ 权重变化: {formatted_old} -> {formatted_new}，重规路径！")
                                self.current_weights = new_weights
                                self.weights_changed_flag = True
                                
            except requests.exceptions.Timeout:
                pass
            except requests.exceptions.ConnectionError:
                pass
            except Exception as e:
                print(f"[Robot {self.robot_id}] 位置上报线程异常: {str(e)}")

    def nav_to_pose(self, target_x, target_y, is_returning_home=False):
        """
        物理仿真引擎：模拟真实机器人的直线连续逼近过程。
        支持中途被打断（如权重改变、速度变为0或-1）。
        """
        while True:
            # 1. 中断检测
            if is_returning_home:
                # 返航模式下：无视权重变化，无视-1状态（因为就是-1触发的），仅允许 0 挡位（紧急制动）打断
                if self.current_speed_gear == 0:
                    return "CANCELED"
            else:
                # 常规巡检模式下：响应所有打断信号
                if self.current_speed_gear in [0, -1]:
                    return "CANCELED"
                if self.weights_changed_flag:
                    return "CANCELED"
                
            # 2. 换算当前真实速度
            actual_speed = MAX_SPEED
            if not is_returning_home and self.current_speed_gear == 2:
                actual_speed = MAX_SPEED * 0.5
                
            # 3. 计算与目标点的距离和向量
            dx = target_x - self.current_x
            dy = target_y - self.current_y
            dist = math.hypot(dx, dy)
            
            # 4. 判断是否抵达目标点 (距离小于单步步长时直接吸附)
            step_distance = actual_speed * SIM_DT
            if dist <= step_distance:
                self.current_x = target_x
                self.current_y = target_y
                return "SUCCEEDED"
                
            # 5. 按照向量和速度步进累加坐标
            self.current_x += (dx / dist) * step_distance
            self.current_y += (dy / dist) * step_distance
            
            # 6. 休眠一个时间步长，使得在外部看来坐标是平滑连续移动的
            time.sleep(SIM_DT)

    def run(self):
        """核心业务主循环，逻辑严格对照 main() 里的 while 循环"""
        print(f"🚀 机器人 {self.robot_id} 仿真服务启动！当前位于 (0.00, 0.00)")
        
        while True:
            # 【下线返航逻辑】
            if self.current_speed_gear == -1:
                if not self.returning_home:
                    print(f"[Robot {self.robot_id}] >>> 启动自动返航程序，直线返回起点 (0, 0) <<<")
                    self.returning_home = True
                    
                    # 传入 is_returning_home=True，在返航期间屏蔽非必要的外部打断信号
                    self.nav_to_pose(0.0, 0.0, is_returning_home=True)
                    
                    if self.current_x == 0.0 and self.current_y == 0.0:
                        print(f"[Robot {self.robot_id}] 💤 已成功到达起点(0,0)，机器人进入彻底断电休眠状态。")
                    else:
                        print(f"[Robot {self.robot_id}] ⚠️ 返航途中被紧急指令(0挡位)打断。")
                
                # 到达终点或休眠中，只需空转即可，等待服务端指令变化
                time.sleep(1.0)
                continue

            # 【0挡位原地待命阻塞】
            if self.current_speed_gear == 0:
                time.sleep(1.0)
                continue
                
            self.returning_home = False
            self.weights_changed_flag = False
            
            # 调用你的生成算法
            print(f"[Robot {self.robot_id}] 正在获取新的目标点...")
            target_points_list = generate_target_points(self.robot_id, self.current_weights, (self.current_x, self.current_y))
            
            if not target_points_list:
                time.sleep(1.0)
                continue

            # 遍历并依次前往目标点
            for pt in target_points_list:
                if self.weights_changed_flag or self.current_speed_gear in [0, -1]:
                    break
                    
                tx, ty = pt[0], pt[1]
                print(f"[Robot {self.robot_id}] 📍 前往下一个目标点: ({tx:.2f}, {ty:.2f})")
                
                result = self.nav_to_pose(tx, ty)
                
                if self.current_speed_gear == 0:
                    print(f"[Robot {self.robot_id}] 巡检由于收到 0 挡位命令被强制打断！")
                    break
                if self.current_speed_gear == -1:
                    break
                if self.weights_changed_flag:
                    break
                
                if result == "SUCCEEDED":
                    print(f"[Robot {self.robot_id}] ✅ 已到达目标点: ({tx:.2f}, {ty:.2f})")
            
            if self.weights_changed_flag:
                print(f"[Robot {self.robot_id}] 🔄 旧的巡检由于权重变化被打断，重新开始获取任务...")
                continue
                
            if self.current_speed_gear not in [0, -1]:
                print(f"[Robot {self.robot_id}] 🏁 已完成当前批次所有目标点遍历，将开始下一次循环")

if __name__ == '__main__':
    robot_sim = SimulatedPatrolNode(TARGET_ROBOT_ID)
    
    # 将模拟主逻辑放入一个单独的线程，以便主线程能够灵敏捕获 Ctrl+C
    main_logic_thread = threading.Thread(target=robot_sim.run, daemon=True)
    main_logic_thread.start()
        
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n⏹️ 模拟测试结束。")
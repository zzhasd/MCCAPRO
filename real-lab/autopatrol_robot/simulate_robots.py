import requests
import time
import math
import threading

# Try to import the existing path generator and preserve its generation logic
try:
    from generate_target_points import generate_target_points
except ImportError:
    print("⚠️ Warning: cannot import generate_target_points; ensure it is beside this script.")
    print("⚠️ Falling back to random point generation.")
    def generate_target_points(robot_id, weights, current_pos):
        import random
        return [[random.uniform(-3, 3), random.uniform(-3, 3), 0] for _ in range(5)]

# ================= Configuration parameters =================
SERVER_URL = 'http://127.0.0.1:9999/report'
TARGET_ROBOT_ID = 0      # Simulated robot ID
MAX_SPEED = 0.6          # Simulated full-speed linear velocity (meters/second)
SIM_DT = 0.05            # Physics simulation timestep (seconds) -> smaller values produce smoother motion
# ============================================

class SimulatedPatrolNode:
    def __init__(self, robot_id):
        self.robot_id = robot_id
        
        # Initial point is (0,0)
        self.current_x = 0.0
        self.current_y = 0.0
        
        # State variables
        self.current_weights = [0.5, 0.5]
        self.weights_changed_flag = False 
        
        self.current_speed_gear = 4  # Default full speed
        self.returning_home = False  # Track whether a low-battery return is in progress
        
        # Start the background heartbeat reporting thread
        self.report_thread = threading.Thread(target=self.report_position_loop, daemon=True)
        self.report_thread.start()

    def report_position_loop(self):
        """Fully simulate patrol_node.py method report_position_loop"""
        while True:
            time.sleep(1.0) # 1Hz Reporting frequency
            payload = {"id": self.robot_id, "x": self.current_x, "y": self.current_y}
            
            try:
                response = requests.post(SERVER_URL, json=payload, timeout=2.0)
                if response.status_code == 200:
                    data = response.json()
                    new_weights = data.get("weights")
                    new_speed = data.get("speed")
                    
                    # ----- 1. Speed control (highest priority) -----
                    if new_speed is not None and new_speed != self.current_speed_gear:
                        print(f"[Robot {self.robot_id}] 🚦 Speed setting changed: {self.current_speed_gear} -> {new_speed}")
                        self.current_speed_gear = new_speed
                        
                        if new_speed == -1:
                            print(f"[Robot {self.robot_id}] ⚡ Battery below 10%; forcing offline and interrupting the current task!")
                        elif new_speed == 0:
                            print(f"[Robot {self.robot_id}] 🛑 Received speed setting 0; interrupting the task and holding position!")
                        elif new_speed == 2:
                            print(f"[Robot {self.robot_id}] 🐢 Received speed setting 2; setting the maximum speed to 50%.")
                        elif new_speed == 4:
                            print(f"[Robot {self.robot_id}] 🐇 Received speed setting 4; restoring full speed to 100%.")

                    # ----- 2. Weight/path control -----
                    if self.current_speed_gear != -1:
                        if new_weights and len(new_weights) == len(self.current_weights):
                            diff = sum(abs(a - b) for a, b in zip(self.current_weights, new_weights))
                            if diff > 1e-4:  
                                formatted_old = [round(w, 3) for w in self.current_weights]
                                formatted_new = [round(w, 3) for w in new_weights]
                                print(f"[Robot {self.robot_id}] ⚖️ Weights changed: {formatted_old} -> {formatted_new}; replanning!")
                                self.current_weights = new_weights
                                self.weights_changed_flag = True
                                
            except requests.exceptions.Timeout:
                pass
            except requests.exceptions.ConnectionError:
                pass
            except Exception as e:
                print(f"[Robot {self.robot_id}] Position reporting thread error: {str(e)}")

    def nav_to_pose(self, target_x, target_y, is_returning_home=False):
        """
        Physics simulation: continuous linear motion toward a target.
        Supports interruption during motion (such as weight changes or speed becoming 0 or-1).
        """
        while True:
            # 1. Check for interruption
            if is_returning_home:
                # Return mode: ignore weight changes and state -1 (which triggered the return); only speed setting 0 (emergency stop) may interrupt
                if self.current_speed_gear == 0:
                    return "CANCELED"
            else:
                # Normal patrol mode: respond to all interruption signals
                if self.current_speed_gear in [0, -1]:
                    return "CANCELED"
                if self.weights_changed_flag:
                    return "CANCELED"
                
            # 2. Calculate the current effective speed
            actual_speed = MAX_SPEED
            if not is_returning_home and self.current_speed_gear == 2:
                actual_speed = MAX_SPEED * 0.5
                
            # 3. Compute distance and direction to the target
            dx = target_x - self.current_x
            dy = target_y - self.current_y
            dist = math.hypot(dx, dy)
            
            # 4. Check arrival (snap to the target when closer than one step)
            step_distance = actual_speed * SIM_DT
            if dist <= step_distance:
                self.current_x = target_x
                self.current_y = target_y
                return "SUCCEEDED"
                
            # 5. Advance coordinates using the direction and speed
            self.current_x += (dx / dist) * step_distance
            self.current_y += (dy / dist) * step_distance
            
            # 6. Sleep for one timestep so externally observed motion remains smooth
            time.sleep(SIM_DT)

    def run(self):
        """Main application loop, matching the while loop in main()"""
        print(f"🚀 Simulation started for robot {self.robot_id}! Current position (0.00, 0.00)")
        
        while True:
            # [Offline return-to-start logic]
            if self.current_speed_gear == -1:
                if not self.returning_home:
                    print(f"[Robot {self.robot_id}] >>> Starting automatic return in a straight line to the start (0, 0) <<<")
                    self.returning_home = True
                    
                    # Pass is_returning_home=True to suppress unnecessary external interruptions during return
                    self.nav_to_pose(0.0, 0.0, is_returning_home=True)
                    
                    if self.current_x == 0.0 and self.current_y == 0.0:
                        print(f"[Robot {self.robot_id}] 💤 Reached the start (0,0); robot entering powered-off sleep mode.")
                    else:
                        print(f"[Robot {self.robot_id}] ⚠️ Return interrupted by an emergency command (speed setting 0).")
                
                # At the endpoint or while sleeping, idle until server commands change
                time.sleep(1.0)
                continue

            # [0 Block and hold position at this speed setting]
            if self.current_speed_gear == 0:
                time.sleep(1.0)
                continue
                
            self.returning_home = False
            self.weights_changed_flag = False
            
            # Call the supplied generation algorithm
            print(f"[Robot {self.robot_id}] Getting new targets...")
            target_points_list = generate_target_points(self.robot_id, self.current_weights, (self.current_x, self.current_y))
            
            if not target_points_list:
                time.sleep(1.0)
                continue

            # Visit the targets in sequence
            for pt in target_points_list:
                if self.weights_changed_flag or self.current_speed_gear in [0, -1]:
                    break
                    
                tx, ty = pt[0], pt[1]
                print(f"[Robot {self.robot_id}] 📍 Moving to the next target: ({tx:.2f}, {ty:.2f})")
                
                result = self.nav_to_pose(tx, ty)
                
                if self.current_speed_gear == 0:
                    print(f"[Robot {self.robot_id}] Patrol interrupted by speed setting 0!")
                    break
                if self.current_speed_gear == -1:
                    break
                if self.weights_changed_flag:
                    break
                
                if result == "SUCCEEDED":
                    print(f"[Robot {self.robot_id}] ✅ Target reached: ({tx:.2f}, {ty:.2f})")
            
            if self.weights_changed_flag:
                print(f"[Robot {self.robot_id}] 🔄 Previous patrol interrupted by a weight change; requesting tasks again...")
                continue
                
            if self.current_speed_gear not in [0, -1]:
                print(f"[Robot {self.robot_id}] 🏁 All targets in this batch visited; starting the next cycle")

if __name__ == '__main__':
    robot_sim = SimulatedPatrolNode(TARGET_ROBOT_ID)
    
    # Run the simulation logic in a separate thread so the main thread can promptly catch Ctrl+C
    main_logic_thread = threading.Thread(target=robot_sim.run, daemon=True)
    main_logic_thread.start()
        
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n⏹️ Simulation test finished.")
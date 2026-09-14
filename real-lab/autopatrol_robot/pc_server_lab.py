from flask import Flask, request, jsonify, render_template, Response
import time
import threading
import math
from collections import deque
import logging
import os
import io

# --- New dependencies for asynchronous disk writes ---
import csv
from datetime import datetime
from queue import Queue, Empty

try:
    import yaml
    from PIL import Image
    HAS_MAP_LIBS = True
except ImportError:
    HAS_MAP_LIBS = False
    print("⚠️ yaml or PIL is missing; map features may be unavailable. Run: pip install pyyaml Pillow")

# --- New: reuse partitioning/tiling from generate_target_points.py to provide border rectangles to the frontend ---
try:
    import generate_target_points as target_planner
    HAS_TARGET_PLANNER = hasattr(target_planner, 'generate_assignment_rectangles')
except Exception as e:
    target_planner = None
    HAS_TARGET_PLANNER = False
    print(f"⚠️ Unable to load generate_target_points.py; assigned-region borders will be unavailable: {e}")

app = Flask(__name__)

# ================= Configuration parameters =================
PORT = 9999
ROBOT_NUM = 2
UPDATE_INTERVAL = 350.0  

# ✨ New: custom fixed weights applied when the timer expires
CUSTOM_WEIGHTS = [0.72, 0.28]
# [0.72, 0.28]  [0.37, 0.63]

# ✨ New: robot cleaning/patrol coverage radius (meters)
COVERAGE_RADIUS = 2.05   

# Battery endurance settings (seconds); configurable per robot
MAX_BATTERY_SEC_LIST = [60.0 * 60.0, 40.0 * 60.0]

if len(MAX_BATTERY_SEC_LIST) < ROBOT_NUM:
    print("⚠️ Warning: MAX_BATTERY_SEC_LIST is shorter than ROBOT_NUM; padding with the default 40 minutes.")
    MAX_BATTERY_SEC_LIST += [40.0 * 60.0] * (ROBOT_NUM - len(MAX_BATTERY_SEC_LIST))

# Pad the weight array (if fewer weights are configured than robots)
if len(CUSTOM_WEIGHTS) < ROBOT_NUM:
    CUSTOM_WEIGHTS += [1.0 / ROBOT_NUM] * (ROBOT_NUM - len(CUSTOM_WEIGHTS))

# ================= Map configuration =================
MAP_DIR = os.environ.get("AUTOPATROL_MAP_DIR", os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "maps")
))
MAP_YAML = "yahboomcar.yaml"

map_info = None
map_image_bytes = None

def load_map():
    """Parse YAML and convert PGM to PNG bytes"""
    global map_info, map_image_bytes
    if not HAS_MAP_LIBS: return
    try:
        yaml_path = os.path.join(MAP_DIR, MAP_YAML)
        with open(yaml_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        
        img_path = os.path.join(MAP_DIR, cfg['image'])
        img = Image.open(img_path)
        
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        map_image_bytes = buf.getvalue()
        
        res = cfg.get('resolution', 0.05)
        origin = cfg.get('origin', [0.0, 0.0, 0.0])
        w, h = img.size
        
        map_info = {
            'origin_x': origin[0],
            'origin_y': origin[1],
            'real_w': w * res,
            'real_h': h * res
        }
        print(f"🗺️ Map loaded: {cfg['image']} (size: {w}x{h}, physical size: {map_info['real_w']:.2f}m x {map_info['real_h']:.2f}m)")
    except Exception as e:
        print(f"❌ Map loading failed; check the path: {e}")


def get_assignment_rects_snapshot(weights_snapshot):
    """Generate/read assigned-region border rectangles for the current weights; recompute only when weights change."""
    global assignment_cache_key, assignment_rects_cache, assignment_version

    if not HAS_TARGET_PLANNER or target_planner is None:
        return {"version": assignment_version, "rects": []}

    key = tuple(round(float(w), 6) for w in weights_snapshot[:ROBOT_NUM])

    with assignment_lock:
        if assignment_cache_key == key:
            return {"version": assignment_version, "rects": assignment_rects_cache}

    try:
        # Make generate_target_points.py use the same map displayed by the PC server.
        target_planner.YAML_PATH = os.path.join(MAP_DIR, MAP_YAML)
        target_planner.IMAGE_DIR = ""  # The frontend only needs JSON borders; no additional matplotlib image is needed

        result = target_planner.generate_assignment_rectangles(list(key))
        new_rects = result.get("rects", [])

        with assignment_lock:
            assignment_rects_cache = new_rects
            assignment_cache_key = key
            assignment_version += 1
            return {"version": assignment_version, "rects": assignment_rects_cache}

    except Exception as e:
        print(f"❌ Failed to compute assigned-region borders: {e}")
        with assignment_lock:
            # Update the key even on failure to avoid repeated errors every second; restart after changing code/maps to recompute.
            assignment_rects_cache = []
            assignment_cache_key = key
            assignment_version += 1
            return {"version": assignment_version, "rects": assignment_rects_cache}

# ================= Asynchronous trajectory storage settings =================
trajectory_queue = Queue()
trajectory_files = {}

def init_trajectory_files():
    """Initialize a trajectory log with a timestamped filename"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("logs", exist_ok=True)
    
    for i in range(ROBOT_NUM):
        filename = os.path.join("logs", f"robot_{i}_path_{timestamp}.csv")
        # a Append mode ensures live writes do not overwrite existing data
        f = open(filename, 'a', newline='', encoding='utf-8')
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'x', 'y']) # Write the header
        trajectory_files[i] = (f, writer)
        print(f"📝 Robot {i} Trajectory log created: {filename}")

def trajectory_saver_loop():
    """Dedicated background thread: write queued data to disk in real time"""
    while True:
        try:
            item = trajectory_queue.get(timeout=1.0)
            r_id, t, x, y = item
            
            if r_id in trajectory_files:
                f, writer = trajectory_files[r_id]
                writer.writerow([t, x, y])
                f.flush() # Force flushing to disk to reduce data loss on power failure
                
            trajectory_queue.task_done()
        except Empty:
            pass
        except Exception as e:
            print(f"❌ Error writing trajectory log: {e}")

# ================= State data structure =================
state_lock = threading.Lock()
start_time = time.time()
global_phase = 1 # ✨ New global phase: 1=initial, 2=weights adjusted

# Initial weights are still equal (for example, for 2 robots 0.5, 0.5)
weights_array = [1.0 / ROBOT_NUM for _ in range(ROBOT_NUM)]
speeds_array = [4 for _ in range(ROBOT_NUM)] 

# ✨ New: assigned-region border cache. Increment the version on weight changes so the frontend clears and redraws borders.
assignment_lock = threading.Lock()
assignment_cache_key = None
assignment_rects_cache = []
assignment_version = 0

robots_state = {}

for i in range(ROBOT_NUM):
    robots_state[i] = {
        'x': None,
        'y': None,
        'last_time': time.time(),
        'speed_history': deque(maxlen=20),
        'path_history_1': deque(maxlen=2000),  # ✨ Store separately by phase: phase 1 trajectory
        'path_history_2': deque(maxlen=2000),  # ✨ Store separately by phase: phase 2 trajectory
        'start_active_time': None, 
        'battery_percent': 100.0
    }

def update_robot_battery(r_id, current_time):
    """Update battery status; force offline when endurance reaches 0."""
    state = robots_state[r_id]
    just_went_offline = False
    if state['start_active_time'] is not None:
        elapsed = current_time - state['start_active_time']
        max_battery_sec = MAX_BATTERY_SEC_LIST[r_id]
        
        remaining = max(0, max_battery_sec - elapsed)
        state['battery_percent'] = (remaining / max_battery_sec) * 100.0
        
        if state['battery_percent'] < 10.0 and speeds_array[r_id] != -1:
            print(f"⚠️ Robot {r_id} Battery below 10% ({state['battery_percent']:.1f}%); forcing offline!")
            speeds_array[r_id] = -1
            just_went_offline = True
    return just_went_offline

def do_calculate_weights(current_time, force_update=False):
    pass

def calculate_weights_loop():
    """Dedicated background thread: update weights once when the countdown ends"""
    global weights_array, global_phase
    
    print(f"⏳ Weight-switch countdown started; custom weights will be applied in {UPDATE_INTERVAL} s: {CUSTOM_WEIGHTS[:ROBOT_NUM]}")
    
    # 1. Sleep until UPDATE_INTERVAL has elapsed
    time.sleep(UPDATE_INTERVAL)
    
    # 2. At expiry, apply custom weights once and switch the global phase
    with state_lock:
        weights_array = CUSTOM_WEIGHTS[:ROBOT_NUM]
        global_phase = 2  # ✨ The partition has changed; start storing trajectories for the new phase
        print(f"🌟 Timer expired! System weights locked to: {weights_array}; trajectories now use new colors!")

    # 3. Do not update weights again; only continue periodic battery checks
    while True:
        time.sleep(1.0)
        current_time = time.time()
        with state_lock:
            for i in range(ROBOT_NUM):
                update_robot_battery(i, current_time)

# ================= HTTP routes =================

@app.route('/')
def index():
    return render_template('index_lab.html')

@app.route('/map.png')
def get_map_image():
    if map_image_bytes:
        return Response(map_image_bytes, mimetype='image/png')
    return "No map available", 404

@app.route('/status', methods=['GET'])
def get_status():
    global weights_array, speeds_array, global_phase # ✨ Expose global_phase to the frontend
    with state_lock:
        current_time = time.time()
        run_time = int(current_time - start_time)
        
        histories, data_counts, batteries, paths_1, paths_2 = {}, {}, {}, {}, {}
        for i in range(ROBOT_NUM):
            update_robot_battery(i, current_time) 
            history_list = list(robots_state[i]['speed_history'])
            histories[i] = history_list
            data_counts[i] = len(history_list)
            batteries[i] = robots_state[i]['battery_percent']
            
            # Return the two trajectory segments separately
            paths_1[i] = list(robots_state[i]['path_history_1'])
            paths_2[i] = list(robots_state[i]['path_history_2'])

        # Copy a snapshot to avoid holding the state lock while computing region borders
        weights_snapshot = list(weights_array)
        speeds_snapshot = list(speeds_array)
        phase_snapshot = global_phase

    assignment_snapshot = get_assignment_rects_snapshot(weights_snapshot)
        
    return jsonify({
        "run_time": run_time,
        "data_counts": data_counts,
        "weights": weights_snapshot,
        "speeds": speeds_snapshot,
        "batteries": batteries,
        "histories": histories,
        "paths_1": paths_1,
        "paths_2": paths_2,
        "map_info": map_info,
        "coverage_radius": COVERAGE_RADIUS,
        "phase": phase_snapshot, # ✨ Return the current global phase so the frontend can reset coverage area
        "assignment_version": assignment_snapshot["version"],
        "assignment_rects": assignment_snapshot["rects"]
    })

@app.route('/report', methods=['POST'])
def report_position():
    global weights_array, speeds_array, global_phase

    data = request.json
    r_id = data.get('id')
    new_x = data.get('x')
    new_y = data.get('y')
    new_time = time.time()

    if r_id is not None and r_id in robots_state:
        with state_lock:
            state = robots_state[r_id]

            if state['start_active_time'] is None:
                state['start_active_time'] = new_time

            if state['x'] is not None and state['y'] is not None:
                dt = new_time - state['last_time']
                if dt > 0:
                    dx = new_x - state['x']
                    dy = new_y - state['y']
                    speed = math.hypot(dx, dy) / dt
                    state['speed_history'].append(speed)

            state['x'] = new_x
            state['y'] = new_y
            state['last_time'] = new_time

            # Store coordinates in the queue for the current phase
            if global_phase == 1:
                state['path_history_1'].append([new_x, new_y])
            else:
                # On entering phase 2, prepend the last phase 1 point to keep the trajectory continuous
                if len(state['path_history_2']) == 0 and len(state['path_history_1']) > 0:
                    state['path_history_2'].append(state['path_history_1'][-1])
                state['path_history_2'].append([new_x, new_y])

            # Push to the disk-write queue without blocking
            trajectory_queue.put((r_id, new_time, new_x, new_y))

            if update_robot_battery(r_id, new_time):
                do_calculate_weights(new_time, force_update=True)

            # ================================
            # New: separate the speed shown in the frontend from the speed sent to the robot
            # Frontend selects 2: display half speed
            # Robot receives 4: execute at full speed
            # ================================
            display_speed = speeds_array[r_id]

            if display_speed == 2:
                command_speed = 4
            else:
                command_speed = display_speed

            weights_snapshot = list(weights_array)

        return jsonify({
            "weights": weights_snapshot,
            "speed": command_speed
        })

    return jsonify({"error": "Invalid robot id"}), 400

@app.route('/set_speed', methods=['POST'])
def set_speed():
    global speeds_array
    data = request.json
    r_id = int(data.get('id', -1))
    new_speed = int(data.get('speed', 4))
    if 0 <= r_id < ROBOT_NUM:
        with state_lock:
            speeds_array[r_id] = new_speed
        return jsonify({"status": "success", "speeds": speeds_array})
    return jsonify({"status": "error"}), 400

@app.route('/force_offline', methods=['POST'])
def force_offline():
    global speeds_array
    data = request.json
    r_id = int(data.get('id', -1))
    if 0 <= r_id < ROBOT_NUM:
        with state_lock:
            max_battery_sec = MAX_BATTERY_SEC_LIST[r_id]
            fake_elapsed = max_battery_sec * 0.91
            robots_state[r_id]['start_active_time'] = time.time() - fake_elapsed
            update_robot_battery(r_id, time.time())
            do_calculate_weights(time.time(), force_update=True)
        return jsonify({"status": "success"})
    return jsonify({"status": "error"}), 400

if __name__ == '__main__':
    load_map() 
    
    # Initialize the CSV file and start the background writer daemon at startup
    init_trajectory_files()
    threading.Thread(target=trajectory_saver_loop, daemon=True).start()
    threading.Thread(target=calculate_weights_loop, daemon=True).start()
    
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    print(f"🚀 PC Server started on HTTP port {PORT}")
    print(f"📊 Monitoring dashboard URL: http://localhost:{PORT}/")
    app.run(host='0.0.0.0', port=PORT, debug=False)

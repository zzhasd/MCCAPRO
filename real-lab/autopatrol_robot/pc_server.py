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

app = Flask(__name__)

# ================= Configuration parameters =================
PORT = 9999
ROBOT_NUM = 2
DMA_ALPHA = 0.6         
UPDATE_INTERVAL = 1500.0  
WAIT_DURATION = 100.0   
HISTORY_SIZE = 20       

# ✨ New: robot cleaning/patrol coverage radius (meters)
COVERAGE_RADIUS = 2.25   

# Battery endurance settings (seconds); configurable per robot
MAX_BATTERY_SEC_LIST = [40.0 * 60.0, 60.0 * 60.0]

if len(MAX_BATTERY_SEC_LIST) < ROBOT_NUM:
    print("⚠️ Warning: MAX_BATTERY_SEC_LIST is shorter than ROBOT_NUM; padding with the default 40 minutes.")
    MAX_BATTERY_SEC_LIST += [40.0 * 60.0] * (ROBOT_NUM - len(MAX_BATTERY_SEC_LIST))

# ================= Map configuration =================
MAP_DIR = os.environ.get("AUTOPATROL_MAP_DIR", os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "maps")
))
MAP_YAML = "yahboomcar_v3.yaml"

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
weights_array = [1.0 / ROBOT_NUM for _ in range(ROBOT_NUM)]
speeds_array = [4 for _ in range(ROBOT_NUM)] 
robots_state = {}

for i in range(ROBOT_NUM):
    robots_state[i] = {
        'x': None,
        'y': None,
        'last_time': time.time(),
        'speed_history': deque(maxlen=HISTORY_SIZE),
        'path_history': deque(maxlen=2000),  # Retain only the latest 2000 points in memory for frontend rendering
        'start_active_time': None, 
        'battery_percent': 100.0
    }

def update_robot_battery(r_id, current_time):
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
    global weights_array
    time_elapsed = (current_time - start_time) >= WAIT_DURATION
    data_ready = all(len(robots_state[i]['speed_history']) == HISTORY_SIZE for i in range(ROBOT_NUM))

    if force_update or (time_elapsed and data_ready):
        scores = []
        active_count = 0
        for i in range(ROBOT_NUM):
            if speeds_array[i] == -1:
                scores.append(0.0)
            else:
                active_count += 1
                history = list(robots_state[i]['speed_history'])
                filtered_speed = 1.0 if not history else history[0]
                if history:
                    for v in history[1:]:
                        filtered_speed = DMA_ALPHA * v + (1 - DMA_ALPHA) * filtered_speed
                
                battery_ratio = robots_state[i]['battery_percent'] / 100.0
                scores.append(filtered_speed * battery_ratio)
        
        total_score = sum(scores)
        if active_count == 0:
            weights_array = [1.0 / ROBOT_NUM for _ in range(ROBOT_NUM)]
        elif total_score > 1e-5:
            weights_array = [s / total_score for s in scores]
        else:
            weights_array = [(1.0 / active_count if speeds_array[i] != -1 else 0.0) for i in range(ROBOT_NUM)]

def calculate_weights_loop():
    while True:
        time.sleep(UPDATE_INTERVAL)
        current_time = time.time()
        with state_lock:
            any_offline_this_tick = False
            for i in range(ROBOT_NUM):
                if update_robot_battery(i, current_time):
                    any_offline_this_tick = True
            do_calculate_weights(current_time, force_update=any_offline_this_tick)

# ================= HTTP routes =================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/map.png')
def get_map_image():
    if map_image_bytes:
        return Response(map_image_bytes, mimetype='image/png')
    return "No map available", 404

@app.route('/status', methods=['GET'])
def get_status():
    global weights_array, speeds_array
    with state_lock:
        current_time = time.time()
        run_time = int(current_time - start_time)
        
        histories, data_counts, batteries, paths = {}, {}, {}, {}
        for i in range(ROBOT_NUM):
            update_robot_battery(i, current_time) 
            history_list = list(robots_state[i]['speed_history'])
            histories[i] = history_list
            data_counts[i] = len(history_list)
            batteries[i] = robots_state[i]['battery_percent']
            paths[i] = list(robots_state[i]['path_history'])
        
    return jsonify({
        "run_time": run_time,
        "data_counts": data_counts,
        "weights": weights_array,
        "speeds": speeds_array,
        "batteries": batteries,
        "histories": histories,
        "paths": paths,
        "map_info": map_info,
        "coverage_radius": COVERAGE_RADIUS  # ✨ Send the radius setting to the frontend
    })

@app.route('/report', methods=['POST'])
def report_position():
    global weights_array, speeds_array
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
            state['path_history'].append([new_x, new_y]) 
            
            # Push to the disk-write queue without blocking
            trajectory_queue.put((r_id, new_time, new_x, new_y))

            if update_robot_battery(r_id, new_time):
                do_calculate_weights(new_time, force_update=True)

        return jsonify({"weights": weights_array, "speed": speeds_array[r_id]})
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

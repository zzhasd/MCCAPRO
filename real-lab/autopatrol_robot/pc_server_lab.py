from flask import Flask, request, jsonify, render_template, Response
import time
import threading
import math
from collections import deque
import logging
import os
import io

# --- 新增的异步落盘依赖 ---
import csv
from datetime import datetime
from queue import Queue, Empty

try:
    import yaml
    from PIL import Image
    HAS_MAP_LIBS = True
except ImportError:
    HAS_MAP_LIBS = False
    print("⚠️ 缺少 yaml 或 PIL 库，地图功能可能无法使用。请执行: pip install pyyaml Pillow")

# --- 新增：复用 generate_target_points.py 中的分区/填充逻辑，为前端提供边框矩形 ---
try:
    import generate_target_points as target_planner
    HAS_TARGET_PLANNER = hasattr(target_planner, 'generate_assignment_rectangles')
except Exception as e:
    target_planner = None
    HAS_TARGET_PLANNER = False
    print(f"⚠️ 无法加载 generate_target_points.py，分配区域边框将不可用: {e}")

app = Flask(__name__)

# ================= 配置参数 =================
PORT = 9999
ROBOT_NUM = 2
UPDATE_INTERVAL = 300.0  

# ✨ 新增：自定义到期后的固定权重设定
CUSTOM_WEIGHTS = [0.37, 0.63]
# [0.72, 0.28]  [0.37, 0.63]

# ✨ 新增：机器车的扫地/巡检覆盖半径（单位：米）
COVERAGE_RADIUS = 2.25   

# 续航设定（秒）：可为每台机器人独立设置
MAX_BATTERY_SEC_LIST = [60.0 * 60.0, 40.0 * 60.0]

if len(MAX_BATTERY_SEC_LIST) < ROBOT_NUM:
    print("⚠️ 警告: MAX_BATTERY_SEC_LIST 长度小于 ROBOT_NUM，将使用默认值 40 分钟补齐。")
    MAX_BATTERY_SEC_LIST += [40.0 * 60.0] * (ROBOT_NUM - len(MAX_BATTERY_SEC_LIST))

# 补齐权重数组（防止配置的数量少于实际机器人数量）
if len(CUSTOM_WEIGHTS) < ROBOT_NUM:
    CUSTOM_WEIGHTS += [1.0 / ROBOT_NUM] * (ROBOT_NUM - len(CUSTOM_WEIGHTS))

# ================= 地图配置 =================
MAP_DIR = r"D:\桌面\Voronoi-and-Adaptive-Grid\real-lab\maps"
MAP_YAML = "yahboomcar_v3.yaml"

map_info = None
map_image_bytes = None

def load_map():
    """解析 yaml 并将 pgm 转换为 png 字节流"""
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
        print(f"🗺️ 地图加载成功: {cfg['image']} (尺寸: {w}x{h}, 真实大小: {map_info['real_w']:.2f}m x {map_info['real_h']:.2f}m)")
    except Exception as e:
        print(f"❌ 地图加载失败，请检查路径是否正确: {e}")


def get_assignment_rects_snapshot(weights_snapshot):
    """按当前权重生成/读取分配区域矩形边框数据。只在权重变化时重新计算。"""
    global assignment_cache_key, assignment_rects_cache, assignment_version

    if not HAS_TARGET_PLANNER or target_planner is None:
        return {"version": assignment_version, "rects": []}

    key = tuple(round(float(w), 6) for w in weights_snapshot[:ROBOT_NUM])

    with assignment_lock:
        if assignment_cache_key == key:
            return {"version": assignment_version, "rects": assignment_rects_cache}

    try:
        # 让 generate_target_points.py 使用 PC 端正在显示的同一张地图。
        target_planner.YAML_PATH = os.path.join(MAP_DIR, MAP_YAML)
        target_planner.IMAGE_DIR = ""  # 前端只需要 JSON 边框，不需要额外保存 matplotlib 图片

        result = target_planner.generate_assignment_rectangles(list(key))
        new_rects = result.get("rects", [])

        with assignment_lock:
            assignment_rects_cache = new_rects
            assignment_cache_key = key
            assignment_version += 1
            return {"version": assignment_version, "rects": assignment_rects_cache}

    except Exception as e:
        print(f"❌ 计算分配区域边框失败: {e}")
        with assignment_lock:
            # 失败时也更新 key，避免每秒重复刷屏报错；修改代码/地图后重启即可重新计算。
            assignment_rects_cache = []
            assignment_cache_key = key
            assignment_version += 1
            return {"version": assignment_version, "rects": assignment_rects_cache}

# ================= 轨迹异步存储配置 =================
trajectory_queue = Queue()
trajectory_files = {}

def init_trajectory_files():
    """初始化轨迹记录文件，生成带时间戳的文件名"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("logs", exist_ok=True)
    
    for i in range(ROBOT_NUM):
        filename = os.path.join("logs", f"robot_{i}_path_{timestamp}.csv")
        # a 模式追加，确保实时写入不覆盖
        f = open(filename, 'a', newline='', encoding='utf-8')
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'x', 'y']) # 写入表头
        trajectory_files[i] = (f, writer)
        print(f"📝 Robot {i} 轨迹记录文件已创建: {filename}")

def trajectory_saver_loop():
    """后台独立线程：负责将队列中的数据实时写入磁盘"""
    while True:
        try:
            item = trajectory_queue.get(timeout=1.0)
            r_id, t, x, y = item
            
            if r_id in trajectory_files:
                f, writer = trajectory_files[r_id]
                writer.writerow([t, x, y])
                f.flush() # 强制刷新到磁盘，防断电丢失
                
            trajectory_queue.task_done()
        except Empty:
            pass
        except Exception as e:
            print(f"❌ 写入轨迹文件时出错: {e}")

# ================= 状态数据结构 =================
state_lock = threading.Lock()
start_time = time.time()
global_phase = 1 # ✨ 新增全局阶段变量：1代表初始，2代表已调整权重

# 初始权重依然平分 (例如：2车就是 0.5, 0.5)
weights_array = [1.0 / ROBOT_NUM for _ in range(ROBOT_NUM)]
speeds_array = [4 for _ in range(ROBOT_NUM)] 

# ✨ 新增：分配区域边框缓存。权重变化时版本号递增，前端据此清空旧边框并重绘。
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
        'path_history_1': deque(maxlen=2000),  # ✨ 分阶段保存：阶段1轨迹
        'path_history_2': deque(maxlen=2000),  # ✨ 分阶段保存：阶段2轨迹
        'start_active_time': None, 
        'battery_percent': 100.0
    }

def update_robot_battery(r_id, current_time):
    """更新电池状态。续航为0强制下线。"""
    state = robots_state[r_id]
    just_went_offline = False
    if state['start_active_time'] is not None:
        elapsed = current_time - state['start_active_time']
        max_battery_sec = MAX_BATTERY_SEC_LIST[r_id]
        
        remaining = max(0, max_battery_sec - elapsed)
        state['battery_percent'] = (remaining / max_battery_sec) * 100.0
        
        if state['battery_percent'] < 10.0 and speeds_array[r_id] != -1:
            print(f"⚠️ Robot {r_id} 续航不足10% ({state['battery_percent']:.1f}%)，强制下线！")
            speeds_array[r_id] = -1
            just_went_offline = True
    return just_went_offline

def do_calculate_weights(current_time, force_update=False):
    pass

def calculate_weights_loop():
    """后台独立线程：仅在倒计时结束时触发一次权重更新"""
    global weights_array, global_phase
    
    print(f"⏳ 权重切换倒计时已启动，将在 {UPDATE_INTERVAL} 秒后更新为自定义权重: {CUSTOM_WEIGHTS[:ROBOT_NUM]}")
    
    # 1. 睡死，直到 UPDATE_INTERVAL 时间到
    time.sleep(UPDATE_INTERVAL)
    
    # 2. 到期后，仅更新一次自定义权重并切换全局阶段
    with state_lock:
        weights_array = CUSTOM_WEIGHTS[:ROBOT_NUM]
        global_phase = 2  # ✨ 标志着分区变了，开始存入新阶段的轨迹
        print(f"🌟 时间到！当前系统权重已锁定为: {weights_array}，轨迹开始使用新颜色标绘！")

    # 3. 之后不再更新权重，仅保留电池的定时巡检
    while True:
        time.sleep(1.0)
        current_time = time.time()
        with state_lock:
            for i in range(ROBOT_NUM):
                update_robot_battery(i, current_time)

# ================= 路由接口 =================

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
    global weights_array, speeds_array, global_phase # ✨ 引入 global_phase 传给前端
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
            
            # 分别返回两段轨迹
            paths_1[i] = list(robots_state[i]['path_history_1'])
            paths_2[i] = list(robots_state[i]['path_history_2'])

        # 拷贝快照，避免分配边框计算时长期占用状态锁
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
        "phase": phase_snapshot, # ✨ 返回当前全局阶段，前端用于重置覆盖面积
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

            # 根据阶段将坐标存入不同的队列
            if global_phase == 1:
                state['path_history_1'].append([new_x, new_y])
            else:
                # 刚切换到阶段2时，为了保证轨迹不断裂，接上阶段1的最后一个点
                if len(state['path_history_2']) == 0 and len(state['path_history_1']) > 0:
                    state['path_history_2'].append(state['path_history_1'][-1])
                state['path_history_2'].append([new_x, new_y])

            # 极速非阻塞推送至落盘队列
            trajectory_queue.put((r_id, new_time, new_x, new_y))

            if update_robot_battery(r_id, new_time):
                do_calculate_weights(new_time, force_update=True)

            # ================================
            # 新增：前端显示速度 与 机器人实际下发速度 分离
            # 前端按 2：界面仍显示半速
            # 机器人实际收到：4，全速执行
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
    
    # 启动时初始化 CSV 文件并开启后台落盘守护线程
    init_trajectory_files()
    threading.Thread(target=trajectory_saver_loop, daemon=True).start()
    threading.Thread(target=calculate_weights_loop, daemon=True).start()
    
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    print(f"🚀 PC Server 启动，HTTP 端口 {PORT}")
    print(f"📊 监控中心访问地址: http://localhost:{PORT}/")
    app.run(host='0.0.0.0', port=PORT, debug=False)
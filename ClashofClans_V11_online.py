import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import pickle
import os
import subprocess
import time

# --- Configuration ---
GRID_SIZE = 44
CENTER = GRID_SIZE / 2
POPULATION_SIZE = 2000
GENERATIONS = 25000
MUTATION_RATE = 0.15
ELITISM_COUNT = 10
SAVE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "th7_brain_tensor_v11.pkl",
) 
#Macro-Genetics

# Every TH7 Building (Count, Size in tiles, Range, Hex Color, Label)
# UPGRADE: Walls are now segmented lines (w, h) instead of 1x1 pegs!
BUILDINGS = {
    'TownHall':    {'count': 1,   'size': 4,      'range': 0,  'c': '#FFC107', 'lbl': 'TH'},
    'ClanCastle':  {'count': 1,   'size': 3,      'range': 9,  'c': '#9E9E9E', 'lbl': 'CC'},
    'ArmyCamp':    {'count': 4,   'size': 5,      'range': 0,  'c': '#8D6E63', 'lbl': 'Camp'},
    'HeroAltar':   {'count': 1,   'size': 3,      'range': 9,  'c': '#795548', 'lbl': 'King'},
    'Cannon':      {'count': 5,   'size': 3,      'range': 9,  'c': '#424242', 'lbl': 'Can'},
    'ArcherTower': {'count': 4,   'size': 3,      'range': 10, 'c': '#4CAF50', 'lbl': 'AT'},
    'Mortar':      {'count': 3,   'size': 3,      'range': 11, 'c': '#9C27B0', 'lbl': 'Mortar'},
    'AirDefense':  {'count': 2,   'size': 3,      'range': 10, 'c': '#F44336', 'lbl': 'AD'},
    'WizardTower': {'count': 2,   'size': 3,      'range': 7,  'c': '#E91E63', 'lbl': 'Wiz'},
    'HiddenTesla': {'count': 2,   'size': 2,      'range': 7,  'c': '#00BCD4', 'lbl': 'Tesla'},
    'Sweeper':     {'count': 1,   'size': 2,      'range': 14, 'c': '#B0BEC5', 'lbl': 'Sweep'},
    'Storages':    {'count': 5,   'size': 3,      'range': 0,  'c': '#03A9F4', 'lbl': 'Loot'},
    'Collectors':  {'count': 12,  'size': 3,      'range': 0,  'c': '#CDDC39', 'lbl': 'Mine'},
    'MiscApps':    {'count': 7,   'size': 3,      'range': 0,  'c': '#FF9800', 'lbl': 'Misc'},
    'Wall_H':      {'count': 17,  'size': (5, 1), 'range': 0,  'c': '#FFFFFF', 'lbl': ''}, 
    'Wall_V':      {'count': 18,  'size': (1, 5), 'range': 0,  'c': '#FFFFFF', 'lbl': ''}
}

# --- GPU TENSOR INITIALIZATION ---
def initialize_population_tensor():
    """Creates a 3D Tensor for every building type directly on the GPU VRAM."""
    pop = {}
    for b_type, data in BUILDINGS.items():
        w, h = data['size'] if isinstance(data['size'], tuple) else (data['size'], data['size'])
        
        x_coords = cp.random.randint(2, 42 - w, size=(POPULATION_SIZE, data['count'], 1), dtype=cp.int32)
        y_coords = cp.random.randint(2, 42 - h, size=(POPULATION_SIZE, data['count'], 1), dtype=cp.int32)
        pop[b_type] = cp.concatenate([x_coords, y_coords], axis=-1)
        
    # --- THE ANCHOR ---
    # Lock the Town Hall to the absolute dead center of the map to prevent gravity traps.
    pop['TownHall'] = cp.full((POPULATION_SIZE, 1, 2), 20, dtype=cp.int32)
    return pop

# --- MASSIVE PARALLEL EVALUATION (CURRICULUM LEARNING UPGRADE) ---
def evaluate_population_tensor(pop):
    """Evaluates all 2000 bases with Phase-Shifted Curriculum Learning."""
    N = POPULATION_SIZE

    # 1. COLLISION DETECTION (3D BOARD)
    boards = cp.zeros((N, GRID_SIZE, GRID_SIZE), dtype=cp.int32)
    out_of_bounds = cp.zeros(N, dtype=cp.int32)

    for b_type, data in BUILDINGS.items():
        w, h = data['size'] if isinstance(data['size'], tuple) else (data['size'], data['size'])
        coords = pop[b_type]

        dx, dy = cp.meshgrid(cp.arange(w), cp.arange(h))
        dx = dx.flatten()
        dy = dy.flatten()

        X = coords[..., 0, None] + dx[None, None, :]
        Y = coords[..., 1, None] + dy[None, None, :]

        batch_idx = cp.broadcast_to(cp.arange(N)[:, None, None], X.shape)

        valid = (X >= 0) & (X < GRID_SIZE) & (Y >= 0) & (Y < GRID_SIZE)
        out_of_bounds += cp.sum(~valid, axis=(1, 2))

        flat_idx = batch_idx[valid] * (GRID_SIZE * GRID_SIZE) + X[valid] * GRID_SIZE + Y[valid]
        counts = cp.bincount(flat_idx, minlength=N * GRID_SIZE * GRID_SIZE)
        boards += counts.reshape((N, GRID_SIZE, GRID_SIZE))

    overlaps = cp.sum(boards > 1, axis=(1, 2))

    # PHASE 1: SURVIVAL SCORING (Only care about overlaps and bounds)
    survival_scores = cp.full(N, 15000, dtype=cp.int32)
    survival_scores -= (overlaps * 5000)
    survival_scores -= (out_of_bounds * 1000)

    # 2. THREAT MATRIX (Vectorized Heatmap)
    th_centers = pop['TownHall'] + BUILDINGS['TownHall']['size']/2.0
    storage_centers = pop['Storages'] + BUILDINGS['Storages']['size']/2.0
    targets = cp.concatenate([th_centers, storage_centers], axis=1)

    def_centers = []
    def_ranges = []
    for d_type in ['Cannon', 'ArcherTower', 'Mortar', 'WizardTower', 'AirDefense']:
        c = pop[d_type] + BUILDINGS[d_type]['size']/2.0
        def_centers.append(c)
        def_ranges.extend([BUILDINGS[d_type]['range']] * c.shape[1])

    defenses = cp.concatenate(def_centers, axis=1)
    ranges_arr = cp.array(def_ranges, dtype=cp.float32)

    diff = targets[:, :, None, :] - defenses[:, None, :, :]
    dist = cp.linalg.norm(diff, axis=-1)

    covered = dist <= ranges_arr[None, None, :]
    coverage_counts = cp.sum(covered, axis=2)

    # PHASE 2: ARCHITECTURAL SCORING
    arch_scores = cp.zeros(N, dtype=cp.int32)
    arch_scores -= cp.sum(coverage_counts < 3, axis=1) * 500
    arch_scores += cp.sum(coverage_counts, axis=1) * 50

    # 3. SMART WALLS & CHAINING (Tensor Raycasting Macro-Upgrade)
    wall_mask = cp.zeros((N, GRID_SIZE, GRID_SIZE), dtype=cp.bool_)
    for w_type in ['Wall_H', 'Wall_V']:
        w, h = BUILDINGS[w_type]['size']
        coords = pop[w_type]
        dx, dy = cp.meshgrid(cp.arange(w), cp.arange(h))
        dx = dx.flatten()
        dy = dy.flatten()
        X = coords[..., 0, None] + dx[None, None, :]
        Y = coords[..., 1, None] + dy[None, None, :]
        
        valid = (X >= 0) & (X < GRID_SIZE) & (Y >= 0) & (Y < GRID_SIZE)
        batch_idx = cp.broadcast_to(cp.arange(N)[:, None, None], X.shape)
        flat_idx = batch_idx[valid] * (GRID_SIZE * GRID_SIZE) + X[valid] * GRID_SIZE + Y[valid]
        counts = cp.bincount(flat_idx, minlength=N * GRID_SIZE * GRID_SIZE)
        wall_mask |= (counts.reshape((N, GRID_SIZE, GRID_SIZE)) > 0)

    neighbors = cp.zeros((N, GRID_SIZE, GRID_SIZE), dtype=cp.int32)
    neighbors[:, :-1, :] += wall_mask[:, 1:, :]
    neighbors[:, 1:, :] += wall_mask[:, :-1, :]
    neighbors[:, :, :-1] += wall_mask[:, :, 1:]
    neighbors[:, :, 1:] += wall_mask[:, :, :-1]

    arch_scores += cp.sum((neighbors * wall_mask), axis=(1,2)) * 25
    arch_scores -= cp.sum(((neighbors == 0) & wall_mask), axis=(1,2)) * 100

    th_c = th_centers[:, 0, :]
    dirs = cp.array([(0,1), (0,-1), (1,0), (-1,0), (1,1), (-1,1), (1,-1), (-1,-1)], dtype=cp.float32)
    t = cp.arange(1, GRID_SIZE, dtype=cp.float32)

    ray_pts = cp.floor(th_c[:, None, None, :] + t[None, None, :, None] * dirs[None, :, None, :]).astype(cp.int32)
    rx = ray_pts[..., 0]
    ry = ray_pts[..., 1]

    in_bounds = (rx >= 0) & (rx < GRID_SIZE) & (ry >= 0) & (ry < GRID_SIZE)
    safe_rx = cp.clip(rx, 0, GRID_SIZE - 1)
    safe_ry = cp.clip(ry, 0, GRID_SIZE - 1)

    batch_ray = cp.arange(N)[:, None, None]
    ray_hits = wall_mask[batch_ray, safe_rx, safe_ry] & in_bounds

    dir_hits = cp.any(ray_hits, axis=2)
    arch_scores += cp.sum(dir_hits, axis=1) * 150
    arch_scores -= cp.sum(~dir_hits, axis=1) * 300

    # 4. CENTER-OF-MASS GRAVITY 
    center_pt = cp.array([CENTER, CENTER], dtype=cp.float32)

    # Clan Castle Gravity (Core defense, pulls it toward the Anchored TH)
    cc_centers = pop['ClanCastle'] + BUILDINGS['ClanCastle']['size']/2.0
    cc_dist = cp.linalg.norm(cc_centers[:, 0, :] - center_pt, axis=-1)
    arch_scores -= (cc_dist * 100).astype(cp.int32)

    # Storages Gravity (Keep loot inside the core)
    storage_dist = cp.linalg.norm(storage_centers - center_pt, axis=-1)
    arch_scores -= cp.sum(storage_dist * 40, axis=1).astype(cp.int32)

    # --- THE CURRICULUM FILTER ---
    final_scores = cp.where((overlaps > 0) | (out_of_bounds > 0), survival_scores, 100000 + arch_scores)

    return final_scores

# --- CONVERTERS ---
def convert_tensor_to_dict(pop_tensors, index):
    """Pulls a single specific layout off the GPU back to normal Python format for saving/drawing."""
    layout = {}
    for b_type in BUILDINGS:
        layout[b_type] = pop_tensors[b_type][index].get().tolist()
    return layout

def visualize_blueprint(layout, score, generation):
    fig, ax = plt.subplots(figsize=(14, 14), facecolor='#2E3B4E')
    ax.set_facecolor('#4CAF50')
    ax.set_xlim(0, GRID_SIZE); ax.set_ylim(0, GRID_SIZE)
    ax.set_title(f"TENSOR Accelerated TH7 | Gen {generation} | Fitness: {score:.0f}",
                   fontsize=18, fontweight='bold', color='white', pad=20)
    ax.set_xticks(np.arange(0, GRID_SIZE + 1, 1))
    ax.set_yticks(np.arange(0, GRID_SIZE + 1, 1))
    ax.grid(color='#388E3C', linestyle='-', linewidth=1, alpha=0.8)
    ax.set_xticklabels([]); ax.set_yticklabels([])
    ax.tick_params(axis='both', which='both', length=0)
    ax.axhline(CENTER, color='white', linestyle='--', alpha=0.5, linewidth=2)
    ax.axvline(CENTER, color='white', linestyle='--', alpha=0.5, linewidth=2)

    for b_type, coords in layout.items():
        if 'Wall' in b_type:
            w, h = BUILDINGS[b_type]['size']
            for (x, y) in coords:
                rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='#424242', facecolor='#FFFFFF')
                ax.add_patch(rect)
            continue
            
        w, h = BUILDINGS[b_type]['size'] if isinstance(BUILDINGS[b_type]['size'], tuple) else (BUILDINGS[b_type]['size'], BUILDINGS[b_type]['size'])
        color = BUILDINGS[b_type]['c']
        label = BUILDINGS[b_type]['lbl']
        for (x, y) in coords:
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='black', facecolor=color)
            ax.add_patch(rect)
            ax.text(x + w/2, y + h/2, label, color='black', weight='bold', fontsize=8, ha='center', va='center')

    map_border = patches.Rectangle((0,0), GRID_SIZE, GRID_SIZE, fill=False, edgecolor='black', linewidth=5)
    ax.add_patch(map_border)
    plt.show()

# --- THERMAL WATCHDOG ---
def get_gpu_temperature():
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
            encoding='utf-8'
        )
        return int(result.strip())
    except Exception:
        return 0

# --- ENGINE ---
if __name__ == "__main__":
    print(f"Initializing 3D Tensor Population of {POPULATION_SIZE} directly in VRAM...")
    pop = initialize_population_tensor()

    # Memory Injector
    if os.path.exists(SAVE_FILE):
        with open(SAVE_FILE, 'rb') as f:
            print("🧠 Loaded previous Tensor Layout!")
            saved_dict = pickle.load(f)
            for b_type in BUILDINGS:
                pop[b_type][0] = cp.array(saved_dict[b_type])

    global_best_layout = None
    global_best_score = -float('inf')

    # --- ADAPTIVE MUTATION VARIABLES ---
    last_best_score = -float('inf')
    stagnation_counter = 0
    current_mutation_rate = MUTATION_RATE

    try:
        for generation in range(1, GENERATIONS + 1):
            # 1. EVALUATE ENTIRE POPULATION AT ONCE
            scores = evaluate_population_tensor(pop)

            # 2. SELECTION & RECORD KEEPING
            max_idx = int(cp.argmax(scores))
            current_best_score = int(scores[max_idx])

            if current_best_score > global_best_score:
                global_best_score = current_best_score
                global_best_layout = convert_tensor_to_dict(pop, max_idx)
                with open(SAVE_FILE, 'wb') as f:
                    pickle.dump(global_best_layout, f)

            # --- ADAPTIVE MUTATION LOGIC (Phase-Aware Upgrade) ---
            if current_best_score <= last_best_score:
                stagnation_counter += 1
            else:
                stagnation_counter = 0  # Reset if we improved!
                last_best_score = current_best_score

            if current_best_score < 50000:
                current_mutation_rate = 0.15 
                if stagnation_counter > 200:
                    current_mutation_rate = 0.01 
                    if generation % 10 == 0:
                        print(f"Gen {generation:4} | Score: {current_best_score:,.0f} | 🪛 KNOT DETECTED! Micro-Mutation: 1%")
                else:
                    if generation % 10 == 0 or generation == 1:
                        print(f"Gen {generation:4} | Score: {current_best_score:,.0f} | Phase 1 Normal: {current_mutation_rate*100:.0f}%")
            else:
                if stagnation_counter > 1000:
                    current_mutation_rate = 0.10 
                    if generation % 10 == 0:
                        print(f"Gen {generation:4} | Score: {current_best_score:,.0f} | 🏗️ ARCHITECTURE DEADLOCK! Restructure: 10%")
                elif stagnation_counter > 500:
                    current_mutation_rate = 0.05 
                    if generation % 10 == 0:
                        print(f"Gen {generation:4} | Score: {current_best_score:,.0f} | ⚠️ STAGNATION! Shuffling: 5%")
                elif stagnation_counter > 200:
                    current_mutation_rate = 0.02 
                    if generation % 10 == 0:
                        print(f"Gen {generation:4} | Score: {current_best_score:,.0f} | ⚠️ RUT DETECTED! Tweaking: 2%")
                else:
                    current_mutation_rate = 0.01 
                    if generation % 10 == 0 or generation == 1:
                        print(f"Gen {generation:4} | Score: {current_best_score:,.0f} | Phase 2 Normal: {current_mutation_rate*100:.0f}%")

            # 3. VECTORIZED TOURNAMENT SELECTION 
            N = POPULATION_SIZE
            tourney_1 = cp.random.randint(0, N, size=(N, 3))
            tourney_2 = cp.random.randint(0, N, size=(N, 3))

            if stagnation_counter > 1000:
                rand_pick = cp.random.randint(0, 3, size=(N,))
                p1_local_idx = rand_pick
                p2_local_idx = cp.random.randint(0, 3, size=(N,))
            else:
                p1_local_idx = cp.argmax(scores[tourney_1], axis=1)
                p2_local_idx = cp.argmax(scores[tourney_2], axis=1)

            parent1_idx = tourney_1[cp.arange(N), p1_local_idx]
            parent2_idx = tourney_2[cp.arange(N), p2_local_idx]

            # 4. VECTORIZED CROSSOVER & SMART CREEP MUTATION
            next_pop = {}
            best_indices = cp.argsort(scores)[-ELITISM_COUNT:]

            for b_type, data in BUILDINGS.items():
                # DO NOT MUTATE THE TOWN HALL. KEEP IT ANCHORED.
                if b_type == 'TownHall':
                    next_pop[b_type] = pop[b_type]
                    continue

                w, h = data['size'] if isinstance(data['size'], tuple) else (data['size'], data['size'])
                p1 = pop[b_type][parent1_idx]
                p2 = pop[b_type][parent2_idx]

                mask = cp.random.rand(N, data['count'], 1) < 0.5
                child = cp.where(mask, p1, p2)

                mut_mask = cp.random.rand(N, data['count'], 1) < current_mutation_rate
                is_nudge = cp.random.rand(N, data['count'], 1) < 0.80 

                teleport_x = cp.random.randint(2, 42 - w, size=(N, data['count'], 1), dtype=cp.int32)
                teleport_y = cp.random.randint(2, 42 - h, size=(N, data['count'], 1), dtype=cp.int32)
                teleport_coords = cp.concatenate([teleport_x, teleport_y], axis=-1)

                nudge_amount = cp.random.randint(-2, 3, size=child.shape, dtype=cp.int32)
                nudge_x = cp.clip(child[..., 0:1] + nudge_amount[..., 0:1], 2, 41 - w)
                nudge_y = cp.clip(child[..., 1:2] + nudge_amount[..., 1:2], 2, 41 - h)
                nudge_coords = cp.concatenate([nudge_x, nudge_y], axis=-1)

                smart_mut = cp.where(is_nudge, nudge_coords, teleport_coords)
                child = cp.where(mut_mask, smart_mut, child)

                child[:ELITISM_COUNT] = pop[b_type][best_indices]
                next_pop[b_type] = child

            pop = next_pop

            # --- THERMAL THROTTLE LOGIC ---
            if generation % 10 == 0:
                current_temp = get_gpu_temperature()
                if current_temp >= 89:
                    print(f"🔥 THERMAL ALARM! GPU hit {current_temp}°C. Pausing for 60 seconds to cool down...")
                    time.sleep(60)
                    print("❄️ Cooldown complete. Resuming evolution!")

        print(f"\nEvolution complete. Generating Blueprint...")
        visualize_blueprint(global_best_layout, global_best_score, GENERATIONS)

    except KeyboardInterrupt:
        print("\nTraining interrupted!")
        if global_best_layout is not None:
            print("Showing the best layout found so far...")
            visualize_blueprint(global_best_layout, global_best_score, "Interrupted")

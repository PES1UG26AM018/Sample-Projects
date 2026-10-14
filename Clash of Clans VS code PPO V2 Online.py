import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import datetime
import pickle
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib import MaskablePPO
from scipy.signal import convolve2d

# --- Configuration ---
GRID_SIZE = 44
CENTER = GRID_SIZE // 2
NUM_PARALLEL_ENVS = 16  # Optimized for 16-core CPU architectures

# Dynamically generate local save paths relative to the script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, 'th7_ppo_output')
os.makedirs(SAVE_DIR, exist_ok=True)
MODEL_PATH = os.path.join(SAVE_DIR, "th7_ppo_model_v2")

BUILDINGS = {
    'TownHall':    {'count': 1,   'size': (4, 4), 'range': 0,  'c': '#FFC107', 'lbl': 'TH',    'id': 1},
    'ClanCastle':  {'count': 1,   'size': (3, 3), 'range': 9,  'c': '#9E9E9E', 'lbl': 'CC',    'id': 2},
    'ArmyCamp':    {'count': 4,   'size': (5, 5), 'range': 0,  'c': '#8D6E63', 'lbl': 'Camp',  'id': 3},
    'HeroAltar':   {'count': 1,   'size': (3, 3), 'range': 9,  'c': '#795548', 'lbl': 'King',  'id': 4},
    'Cannon':      {'count': 5,   'size': (3, 3), 'range': 9,  'c': '#424242', 'lbl': 'Can',   'id': 5},
    'ArcherTower': {'count': 4,   'size': (3, 3), 'range': 10, 'c': '#4CAF50', 'lbl': 'AT',    'id': 6},
    'Mortar':      {'count': 3,   'size': (3, 3), 'range': 11, 'c': '#9C27B0', 'lbl': 'Mortar','id': 7},
    'AirDefense':  {'count': 2,   'size': (3, 3), 'range': 10, 'c': '#F44336', 'lbl': 'AD',    'id': 8},
    'WizardTower': {'count': 2,   'size': (3, 3), 'range': 7,  'c': '#E91E63', 'lbl': 'Wiz',   'id': 9},
    'HiddenTesla': {'count': 2,   'size': (2, 2), 'range': 7,  'c': '#00BCD4', 'lbl': 'Tesla', 'id': 10},
    'Sweeper':     {'count': 1,   'size': (2, 2), 'range': 14, 'c': '#B0BEC5', 'lbl': 'Sweep', 'id': 11},
    'Storages':    {'count': 5,   'size': (3, 3), 'range': 0,  'c': '#03A9F4', 'lbl': 'Loot',  'id': 12},
    'Collectors':  {'count': 12,  'size': (3, 3), 'range': 0,  'c': '#CDDC39', 'lbl': 'Mine',  'id': 13},
    'MiscApps':    {'count': 7,   'size': (3, 3), 'range': 0,  'c': '#FF9800', 'lbl': 'Misc',  'id': 14},
    'Wall_H':      {'count': 17,  'size': (5, 1), 'range': 0,  'c': '#FFFFFF', 'lbl': 'W_H',   'id': 15},
    'Wall_V':      {'count': 18,  'size': (1, 5), 'range': 0,  'c': '#FFFFFF', 'lbl': 'W_V',   'id': 16}
}

PLACEMENT_SEQUENCE = []
for b_type, data in BUILDINGS.items():
    if b_type == 'TownHall':
        continue
    for _ in range(data['count']):
        PLACEMENT_SEQUENCE.append(b_type)

class ClashBaseEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(GRID_SIZE * GRID_SIZE)
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(low=0, high=20, shape=(GRID_SIZE, GRID_SIZE), dtype=np.uint8),
            "next_building_size": spaces.Box(low=0, high=10, shape=(2,), dtype=np.uint8)
        })

        self.grid = None
        self.layout_dict = None
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        self.layout_dict = {b: [] for b in BUILDINGS}

        th_w, th_h = BUILDINGS['TownHall']['size']
        th_x, th_y = 20, 20
        self.layout_dict['TownHall'].append((th_x, th_y))
        self.grid[th_x : th_x + th_w, th_y : th_y + th_h] = BUILDINGS['TownHall']['id']

        return self._get_obs(), {}

    def _get_obs(self):
        if self.current_step < len(PLACEMENT_SEQUENCE):
            b_type = PLACEMENT_SEQUENCE[self.current_step]
            w, h = BUILDINGS[b_type]['size']
            size_arr = np.array([w, h], dtype=np.uint8)
        else:
            size_arr = np.array([0, 0], dtype=np.uint8)

        return {"grid": self.grid.copy(), "next_building_size": size_arr}

    def action_masks(self):
        if self.current_step >= len(PLACEMENT_SEQUENCE):
            return np.zeros(GRID_SIZE * GRID_SIZE, dtype=bool)

        b_type = PLACEMENT_SEQUENCE[self.current_step]
        w, h = BUILDINGS[b_type]['size']

        grid_binary = (self.grid > 0).astype(np.uint8)
        kernel = np.ones((w, h), dtype=np.uint8)
        conv_res = convolve2d(grid_binary, kernel, mode='valid')

        full_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        full_mask[:GRID_SIZE - w + 1, :GRID_SIZE - h + 1] = (conv_res == 0)
        return full_mask.flatten()

    def step(self, action):
        if self.current_step >= len(PLACEMENT_SEQUENCE):
            return self._get_obs(), 0, True, False, {}

        x = action // GRID_SIZE
        y = action % GRID_SIZE
        b_type = PLACEMENT_SEQUENCE[self.current_step]
        w, h = BUILDINGS[b_type]['size']

        if x + w > GRID_SIZE or y + h > GRID_SIZE or np.sum(self.grid[x:x+w, y:y+h]) > 0:
            return self._get_obs(), -50.0, True, False, {}

        self.grid[x:x+w, y:y+h] = BUILDINGS[b_type]['id']
        self.layout_dict[b_type].append((x, y))
        self.current_step += 1
        done = self.current_step == len(PLACEMENT_SEQUENCE)

        reward = 0.0
        if done:
            reward = self.calculate_fitness() / 1000.0

        return self._get_obs(), reward, done, False, {}

    def calculate_fitness(self):
        score = 0
        th_c = np.array([22.0, 22.0])
        storage_centers = np.array(self.layout_dict['Storages']) + 1.5
        targets = np.vstack((th_c, storage_centers))

        def_centers = []
        def_ranges = []
        for d_type in ['Cannon', 'ArcherTower', 'Mortar', 'WizardTower', 'AirDefense']:
            for (x, y) in self.layout_dict[d_type]:
                w, _ = BUILDINGS[d_type]['size']
                def_centers.append([x + w/2.0, y + w/2.0])
                def_ranges.append(BUILDINGS[d_type]['range'])

        defenses = np.array(def_centers)
        ranges = np.array(def_ranges)

        diff = targets[:, np.newaxis, :] - defenses[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=-1)

        covered = dist <= ranges[np.newaxis, :]
        coverage_counts = np.sum(covered, axis=1)

        score -= np.sum(coverage_counts < 3) * 1000
        score += np.sum(coverage_counts) * 300

        center_pt = np.array([CENTER, CENTER])
        cc_centers = np.array(self.layout_dict['ClanCastle']) + 1.5
        score -= np.linalg.norm(cc_centers - center_pt, axis=-1)[0] * 200
        score -= np.sum(np.linalg.norm(defenses - center_pt, axis=-1)) * 40

        trash_centers = []
        for t_type in ['ArmyCamp', 'Collectors', 'MiscApps']:
            for (x, y) in self.layout_dict[t_type]:
                w, h = BUILDINGS[t_type]['size']
                trash_centers.append([x + w/2.0, y + h/2.0])

        trash = np.array(trash_centers)
        dists = np.linalg.norm(trash - center_pt, axis=-1)
        score -= np.sum(np.abs(dists - 14.0)) * 30

        wall_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        for w_type in ['Wall_H', 'Wall_V']:
            for (x, y) in self.layout_dict[w_type]:
                w, h = BUILDINGS[w_type]['size']
                wall_mask[x:x+w, y:y+h] = True

        neighbors = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        neighbors[:-1, :] += wall_mask[1:, :]
        neighbors[1:, :] += wall_mask[:-1, :]
        neighbors[:, :-1] += wall_mask[:, 1:]
        neighbors[:, 1:] += wall_mask[:, :-1]

        score += np.sum((neighbors * wall_mask)) * 150
        score -= np.sum(((neighbors == 0) & wall_mask)) * 500

        dirs = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (-1,1), (1,-1), (-1,-1)]
        dir_hits = 0
        for dx, dy in dirs:
            for step in range(1, GRID_SIZE):
                rx = int(th_c[0] + dx * step)
                ry = int(th_c[1] + dy * step)
                if 0 <= rx < GRID_SIZE and 0 <= ry < GRID_SIZE:
                    if wall_mask[rx, ry]:
                        dir_hits += 1
                        break
        score += dir_hits * 2000
        score -= (8 - dir_hits) * 1500

        return float(score)

# --- CUSTOM TIMESTAMP CALLBACK ---
class TimestampLoggerCallback(BaseCallback):
    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logger.record("time/timestamp", current_time)

# --- CUSTOM BEST LAYOUT TRACKER ---
class BestLayoutCallback(BaseCallback):
    def __init__(self, eval_env, save_path, eval_freq=10000, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.save_path = save_path
        self.eval_freq = eval_freq
        self.best_reward = -float('inf')

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            obs, _ = self.eval_env.reset()
            done = False
            while not done:
                action_masks = self.eval_env.action_masks()
                action, _ = self.model.predict(obs, action_masks=action_masks, deterministic=True)
                obs, reward, done, _, _ = self.eval_env.step(action)
            
            if reward > self.best_reward:
                self.best_reward = reward
                if self.verbose > 0:
                    print(f"\n🏆 New record architecture locked! Score: {reward * 1000:,.0f} | Saved to disk.")
                with open(self.save_path, 'wb') as f:
                    pickle.dump(self.eval_env.layout_dict, f)
        return True

def mask_fn(env: gym.Env) -> np.ndarray:
    return env.unwrapped.action_masks()

def make_env():
    def _init():
        env = ClashBaseEnv()
        env = Monitor(env) 
        return ActionMasker(env, mask_fn)
    return _init

# --- ENGINE ---
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")

    print(f"Starting {NUM_PARALLEL_ENVS} parallel environments for local hardware...")
    
    # Spawn method is critical for Windows/Linux local execution to prevent PyTorch thread corruption
    vec_env = SubprocVecEnv([make_env() for _ in range(NUM_PARALLEL_ENVS)], start_method='spawn')

    if os.path.exists(MODEL_PATH + ".zip"):
        print("🧠 Checkpoint detected! Resuming training...")
        model = MaskablePPO.load(MODEL_PATH, env=vec_env, device="cuda")
    else:
        print("Starting fresh PPO run...")
        model = MaskablePPO(
            "MultiInputPolicy",
            vec_env,
            gamma=0.99,
            learning_rate=3e-4,
            n_steps=4096,          # Expanded buffer for 32GB+ VRAM
            batch_size=2048,       # Increased throughput for tensor cores
            ent_coef=0.005,
            verbose=1,
            device="cuda"
        )

    # --- LOGGER SETUP ---
    LOG_DIR = os.path.join(SAVE_DIR, "training_logs")
    os.makedirs(LOG_DIR, exist_ok=True)

    csv_logger = configure(LOG_DIR, ["stdout", "csv"])
    model.set_logger(csv_logger)

    # --- CALLBACK BINDING ---
    checkpoint_callback = CheckpointCallback(save_freq=25_000, save_path=SAVE_DIR, name_prefix='th7_local_v2')
    timestamp_callback = TimestampLoggerCallback()
    
    eval_env = ClashBaseEnv()
    best_layout_path = os.path.join(SAVE_DIR, 'th7_best_layout.pkl')
    best_layout_callback = BestLayoutCallback(eval_env, best_layout_path, eval_freq=10000, verbose=1)
    
    callback_list = CallbackList([checkpoint_callback, timestamp_callback, best_layout_callback])

    print("Training Agent... Press Ctrl+C anytime to stop and preview the layout.")
    try:
        model.learn(total_timesteps=10_000_000, callback=callback_list, reset_num_timesteps=False)
        model.save(MODEL_PATH)
        print("Training finished. Saved to disk.")
    except KeyboardInterrupt:
        print("\nSession paused. Saving current weights to disk...")
        model
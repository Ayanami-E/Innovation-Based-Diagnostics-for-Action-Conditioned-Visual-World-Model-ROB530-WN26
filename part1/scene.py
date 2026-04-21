"""MuJoCo PushT rollout helpers (extracted from part1_mujoco_kf.py)."""

from pathlib import Path

import numpy as np
import mujoco

IMG_SIZE = 256
XML_PATH = str(Path(__file__).resolve().parent.parent / "pusht_scene.xml")
DT = 0.05  # 25 substeps * 0.002s per frame (20 Hz)


class MuJoCoPushT:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, IMG_SIZE, IMG_SIZE)

    def reset(self, seed=42):
        rng = np.random.RandomState(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0] = rng.uniform(-0.02, 0.02)
        self.data.qpos[1] = rng.uniform(-0.02, 0.02)
        self.data.qpos[2] = rng.uniform(-np.pi, np.pi)
        self.data.qpos[3] = self.data.qpos[0] + rng.uniform(0.04, 0.07)
        self.data.qpos[4] = self.data.qpos[1] + rng.uniform(-0.02, 0.02)
        mujoco.mj_forward(self.model, self.data)

    def get_gt_state(self):
        return np.array([self.data.qpos[0], self.data.qpos[1],
                         self.data.qpos[2]])

    def get_pusher_pos(self):
        return np.array([self.data.qpos[3], self.data.qpos[4]])

    def step(self, ctrl):
        self.data.ctrl[0] = ctrl[0]
        self.data.ctrl[1] = ctrl[1]
        for _ in range(25):
            mujoco.mj_step(self.model, self.data)

    def render(self):
        self.renderer.update_scene(self.data, camera="top_down")
        return self.renderer.render().copy()


def generate_episode(n_steps=300, seed=42):
    env = MuJoCoPushT()
    env.reset(seed=seed)
    rng = np.random.RandomState(seed + 1)

    gt_states = [env.get_gt_state()]
    images = [env.render()]

    for step in range(n_steps - 1):
        t_pos = env.get_gt_state()[:2]
        a_pos = env.get_pusher_pos()
        phase = (step // 80) % 3
        if phase == 0:
            d = t_pos - a_pos
            n = np.linalg.norm(d)
            if n > 1e-3:
                d /= n
            ctrl = d * 0.15 + rng.randn(2) * 0.02
        elif phase == 1:
            d = -t_pos
            n = np.linalg.norm(d)
            if n > 1e-3:
                d /= n
            ctrl = d * 0.15 + rng.randn(2) * 0.01
        else:
            off = a_pos - t_pos
            p = np.array([-off[1], off[0]])
            n = np.linalg.norm(p)
            if n > 1e-3:
                p /= n
            ctrl = p * 0.10 + rng.randn(2) * 0.02
        env.step(ctrl)
        gt_states.append(env.get_gt_state())
        images.append(env.render())

    return np.array(gt_states), images

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def random_rotate_point_cloud(points):
    angle_rad = np.random.uniform(0, 2*np.pi)
    cos = np.cos(angle_rad)
    sin = np.sin(angle_rad)
    rotation_matrix = np.array([[cos, -sin, 0],
                                [sin,  cos, 0],
                                [0,       0,      1]])
    return points.dot(rotation_matrix.T)

def random_scale_point_cloud(points):
    scale_factor = np.random.uniform(0.8, 1.2)
    return points * scale_factor

def random_translate_point_cloud(points):
    translation_vector = np.random.uniform(-0.1, 0.1, points.shape[1])
    return points + translation_vector

def jitter_point_cloud(points):
    sigma = 0.01
    clip = 0.05
    noise = np.clip(sigma * np.random.randn(*points.shape), -clip, clip)
    return points + noise

def random_dropout_point_cloud(points):
    drop_rate = 0.2
    num_points = points.shape[0]
    num_drop = int(num_points * drop_rate)

    drop_indices = np.random.choice(num_points, num_drop, replace=False)
    keep_mask = np.ones(num_points, dtype=bool)
    keep_mask[drop_indices] = False
    if np.sum(keep_mask) == 0:
        keep_mask[np.random.randint(points.shape[0])] = True
    return points[keep_mask]

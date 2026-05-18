import argparse
import numpy as np
import os
import torch
import trimesh
from torch.utils.data import Dataset, DataLoader
import random
import open3d as o3d

import utils

class ModelNetDataset(Dataset):
    def __init__(self, root='ModelNet10', split='train', num_points=2048, augments=None):
        self.root = root
        self.split = split
        self.num_points = num_points
        self.augments = augments
        self.classes = {
            category: index
            for index, category in enumerate(
                sorted([d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))])
            )
        }
        self.index_to_category = {index: category for category, index in self.classes.items()}
        self.data, self.labels = self.load_data()

    def load_data(self):
        data, labels = [], []

        for category in self.classes:
            category_split_path = os.path.join(self.root, category, self.split)
            for filename in os.listdir(category_split_path):
                if filename.endswith('.off'):
                    data.append(os.path.join(category_split_path, filename))
                    labels.append(self.classes[category])

        return data, labels

    def sample_point_cloud(self, mesh_path):
        mesh = trimesh.load(mesh_path, force='mesh')
        points, _ = trimesh.sample.sample_surface(mesh, self.num_points)

        return points

    def return_class(self, idx):
        label = self.labels[idx]
        return self.index_to_category[label]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mesh_path = self.data[idx]
        points = self.sample_point_cloud(mesh_path)

        if self.augments:
            for augment in self.augments:
                if augment == "rotate":
                    points = utils.random_rotate_point_cloud(points)
                elif augment == "scaling":
                    points = utils.random_scale_point_cloud(points)
                elif augment == "translation":
                    points = utils.random_translate_point_cloud(points)
                elif augment == "jitter":
                    points = utils.jitter_point_cloud(points)
                elif augment == "dropout":
                    points = utils.random_dropout_point_cloud(points)
                else:
                    print("Invailid Augmentation!")

        return points.astype(np.float32), self.labels[idx]

def test_dataloader(density):
    augments = [
        "rotate",
        "scaling",
        "translation",
        "jitter",
        "dropout"
    ]
    dataset = ModelNetDataset(root='ModelNet10', split='train', num_points=density, augments=augments)
    dataloader = DataLoader(dataset)

    for points, labels in dataloader:
        print(f'Points shape: {points.shape}')
        print(f'Labels shape: {labels.shape}')

def export_sample_point_clouds():
    dataset = ModelNetDataset(root='ModelNet10', split='train')
    export_root = "result/sample_point_clouds"
    os.makedirs(export_root, exist_ok=True)

    indices = random.sample(range(len(dataset)), 5)

    for idx in indices:
        mesh_file = dataset.data[idx]
        label = dataset.return_class(idx)
        mesh = trimesh.load(mesh_file, force='mesh')

        mesh.export(os.path.join(export_root, f"{label}_{idx}_mesh.obj"))

        points1, _ = trimesh.sample.sample_surface(mesh, 512)
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(points1)
        o3d.io.write_point_cloud(os.path.join(export_root, f"{label}_{idx}_512points.ply"), pcd1)

        points2, _ = trimesh.sample.sample_surface(mesh, 2048)
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(points2)
        o3d.io.write_point_cloud(os.path.join(export_root, f"{label}_{idx}_2048points.ply"), pcd2)

def export_augmentations():
    export_root = "result/augmentations"
    os.makedirs(export_root, exist_ok=True)

    dataset = ModelNetDataset(root='ModelNet10', split='train', num_points=2048, augments=None)

    idx = random.randint(1, len(dataset))

    mesh_file = dataset.data[idx]
    mesh = trimesh.load(mesh_file, force='mesh')
    mesh.export(os.path.join(export_root, f"original_mesh.obj"))

    points, _ = dataset[idx]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(os.path.join(export_root, f"original_points.ply"), pcd)

    augments = [
        "rotate",
        "scaling",
        "translation",
        "jitter",
        "dropout"
    ]
    # Two variations generated using each augmentation
    for augment in augments:
        dataset = ModelNetDataset(root='ModelNet10', split='train', num_points=2048, augments=[augment])

        points1, _ = dataset[idx]
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(points1)
        o3d.io.write_point_cloud(os.path.join(export_root, f"{augment}_1.ply"), pcd1)

        points2, _ = dataset[idx]
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(points2)
        o3d.io.write_point_cloud(os.path.join(export_root, f"{augment}_2.ply"), pcd2)

    # One final example with all augmentations applied at once
    dataset = ModelNetDataset(root='ModelNet10', split='train', num_points=2048, augments=augments)

    points, _ = dataset[idx]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(os.path.join(export_root, f"all_augment.ply"), pcd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--density', type=int, default=2048, help='The number of points to sample')
    args = parser.parse_args()
    test_dataloader(args.density)

    # export_sample_point_clouds()
    # export_augmentations()



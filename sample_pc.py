import trimesh
import numpy as np
import open3d as o3d

# Load the mesh (supports .off, .obj, .stl, etc.)
# mesh = trimesh.load_mesh("path to mesh", force_mesh=<bool>) # Complete this!

# Sample N points uniformly on the surface
N = 2048
# Method 1:
# points, _ = trimesh.sample.sample_surface(mesh, N)

# Method 2:
# Compute face areas
areas = mesh.area_faces
probabilities = areas / areas.sum()  # Normalize

# Sample faces proportionally to area
face_indices = np.random.choice(len(mesh.faces), size=N, p=probabilities)

# Barycentric coordinate sampling within faces
barycentric_coords = np.random.dirichlet((1,1,1), size=N)
vertices = mesh.vertices[mesh.faces[face_indices]]
points = (vertices * barycentric_coords[:, :, None]).sum(axis=1)

print("Sampled points:", points.shape)  # (N, 3)

# Convert sampled points to Open3D PointCloud for saving
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

o3d.io.write_point_cloud("sampled_points.ply", pcd)


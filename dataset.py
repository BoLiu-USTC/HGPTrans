"""STL mesh -> graph dataset with per-sample vertex standardization.

STLGraphDataset loads STL meshes on the fly, builds a graph from the mesh
(surface mesh faces -> bidirectional edge index) and attaches a scalar drag
label. Meshes above ``simplify_size`` vertices are decimated with quadric
edge-collapse to keep memory bounded.
"""

import os
import time

import numpy as np
import torch
import trimesh
from torch_geometric.data import Data, Dataset


def simplify_mesh(mesh, target_face=417114):
    """Decimate a mesh to ``target_face`` faces via quadric decimation."""
    print(f"Simplify mesh from {len(mesh.faces)} to {target_face} ")
    return mesh.simplify_quadric_decimation(face_count=target_face)


class STLGraphDataset(Dataset):
    """Mesh-graph regression dataset built from a directory of STL files."""

    def __init__(self, stl_files, labels, input_dim=6, transform=None, simplify_size=417114):
        self.stl_files = stl_files
        self.labels = labels
        self.transform = transform
        self.input_dim = input_dim
        self.simplify_size = simplify_size

    def __len__(self):
        return len(self.stl_files)

    def __getitem__(self, idx):
        stl_file = self.stl_files[idx]
        label = self.labels[idx]

        # retry loop for transient IO failures (e.g. busy network storage)
        max_retries = 5
        mesh = None
        for attempt in range(max_retries):
            try:
                mesh = trimesh.load_mesh(stl_file)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    raise RuntimeError(
                        f"Failed to read {stl_file} after {max_retries} attempts: {e}")
        if mesh is None:
            raise RuntimeError(f"Failed to read {stl_file}")

        # decimate oversized meshes
        if self.simplify_size and len(mesh.vertices) > self.simplify_size:
            mesh = simplify_mesh(mesh, self.simplify_size)

        vertices = mesh.vertices
        faces = mesh.faces
        normals = mesh.vertex_normals

        x = torch.tensor(np.concatenate([vertices, normals], -1),
                         dtype=torch.float32)[:, :self.input_dim]

        # undirected edges from the triangle mesh
        edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        y = torch.tensor([label], dtype=torch.float32)
        batch = torch.tensor([0] * x.shape[0], dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, y=y, batch=batch)

        if self.transform:
            data = self.transform(data)

        return data


class Standardize:
    """Per-sample vertex feature standardization (zero mean / unit variance)."""

    def __call__(self, data):
        x_mean = data.x.mean(dim=0)
        x_std = data.x.std(dim=0)
        data.x = (data.x - x_mean) / x_std
        return data

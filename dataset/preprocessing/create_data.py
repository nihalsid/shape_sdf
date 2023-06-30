import trimesh
import torch
import mesh2sdf
import numpy as np
import random
from pathlib import Path


def create_data(mesh_path_in, npz_root_out, num_samples=250000):
    mesh = trimesh.load(mesh_path_in, process=False)
    print('mesh bounds: ', mesh.bounds)   # make sure it is normalized
    # num_samples on surface
    samples_surface = trimesh.sample.sample_surface(mesh, num_samples)[0]
    # num_samples near surface // 2
    samples_near_surface_0 = trimesh.sample.sample_surface(mesh, num_samples // 2)[0] + 0.005 * np.random.randn(num_samples // 2, 3)
    # num_samples near surface // 2
    samples_near_surface_1 = trimesh.sample.sample_surface(mesh, num_samples // 2)[0] + 0.01 * np.random.randn(num_samples // 2, 3)
    # num_samples near surface // 2
    samples_near_surface_2 = trimesh.sample.sample_surface(mesh, num_samples // 2)[0] + 0.1 * np.random.randn(num_samples // 2, 3)
    # num_samples * 0.25 random
    samples_random = np.random.uniform(-0.5, 0.5, size=(num_samples // 4, 3))
    all_samples = np.concatenate([samples_surface, samples_near_surface_0, samples_near_surface_1, samples_near_surface_2, samples_random], axis=0)

    t_vertices = torch.from_numpy(mesh.vertices).cuda().float()
    t_faces = torch.from_numpy(mesh.faces).cuda()

    # Legacy, open source mesh2sdf code
    sdf = mesh2sdf.mesh2sdf_gpu(torch.from_numpy(all_samples).contiguous().float().cuda(), t_vertices[t_faces])[0].cpu().numpy()

    # shuffle
    indices = list(range(all_samples.shape[0]))
    random.shuffle(indices)
    np.savez_compressed(npz_root_out / f"{mesh_path_in.stem}.npz", points=all_samples[indices, :], sdf=sdf[indices])


if __name__ == "__main__":
    create_data(Path("data/raw/human.ply"), Path("data/"))

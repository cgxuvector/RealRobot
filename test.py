import pyrender
import numpy as np

world_pts = np.load("./w_pt_[-1.0000000e+00  0.0000000e+00 -1.2246468e-16].npy")
world_pts = world_pts / world_pts[3, :]
mesh = pyrender.Mesh.from_points(world_pts)

scene = pyrender.Scene()
scene.add(mesh)
pyrender.Viewer(scene)
import numpy as np
import pandas as pd
import pyvista as pv
import trimesh


################################################################################ 
# convert pyvista mesh to trimesh mesh by triangulating elements
def pyvistaToTrimesh(mesh):
    pvTriMesh = pv.PolyDataFilters.triangulate(mesh)
    faces = pvTriMesh.faces.reshape(-1,4)[:,1:]
    points = pvTriMesh.points
    return trimesh.Trimesh(points, faces, process=False)

################################################################################ 
# add third dimension to coordinates for plotting
def addZ(x):
    return np.hstack([x,np.zeros((x.shape[0],1))])
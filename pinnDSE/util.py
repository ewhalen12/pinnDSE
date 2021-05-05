import numpy as np
import pandas as pd
import pyvista as pv
import trimesh


################################################################################ 
# add third dimension to coordinates for plotting
def addZ(x):
    return np.hstack([x,np.zeros((x.shape[0],1))])
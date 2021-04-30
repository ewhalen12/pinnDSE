import numpy as np
import pandas as pd
from pyNastran.op2.op2 import read_op2
from pyNastran.bdf.bdf import BDF
import gmsh
import shutil
import pyvista as pv

def loadOptistructResults(op2File, loadNodalStress=False, colNames=['t1', 't2']):
    res = read_op2(op2File, build_dataframe=True, debug=False, mode='optistruct');
    resDf = res.displacements[1].data_frame
    
    if loadNodalStress:
        colNames += ['oxx', 'oyy', 'txy']
        stressDf = res.cquad4_stress[1].data_frame
        nodeStressDf = stressDf.groupby('NodeID').mean().drop('CEN').drop(['index', 'fiber_distance', 'angle', 'omax', 'omin'], axis=1)
        resDf = resDf.join(nodeStressDf)
    
    resDf = resDf.set_index('NodeID')
    resDf = resDf[colNames]
    resDf = resDf.dropna()
    return resDf

def loadOptistructModel(femFile):
    bdfFile = femFile.replace('.fem', '.bdf')
    vtkFile = femFile.replace('.fem', '.vtk')
    shutil.copyfile(femFile, bdfFile)

    # read solver deck
    gmsh.initialize()
    gmsh.open(bdfFile)

    # write mesh file
    gmsh.option.setNumber('Mesh.Binary', 1) #0=ASCII, 1=binary
    gmsh.write(vtkFile)
    gmsh.finalize()

    # read with pyvista
    mesh = pv.read(vtkFile)
    return mesh


op2File = 'data/controlArm/v1.0/controlArm01_nom.op2'
resDf = loadOptistructResults(op2File, loadNodalStress=True)

femFile = 'data/controlArm/v1.0/controlArm01_nom.fem'
mesh = loadOptistructModel(femFile)


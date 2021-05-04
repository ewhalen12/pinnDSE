import numpy as np
import pandas as pd
from pyNastran.op2.op2 import read_op2
from pyNastran.op2.op2_geom import read_op2_geom
import pyvista as pv
import trimesh
import deepxde as dde

from .util import *

class MeshGeom(dde.geometry.geometry.Geometry):
    def __init__(self, op2File):
        self.dim = 2
        self.idstr = 'MeshGeom'
        self.mesh, self.resDf = loadOptistructModel(op2File, loadResults=True)
        self.bndDict = getAllBoundaries(self.mesh)
        self.bndLenDict, self.bndNormDict = processBoundaries(self.bndDict)

################################################################################ 
# load the mesh and results from an OptiStruct-generated OP2 file
# !assumes nodes ids are contiguous and start at 1
def loadOptistructModel(op2File, loadResults=True):
    # read geometry
    geom = read_op2_geom(op2File, build_dataframe=False, debug=False)
    mesh = op2GeomToPv(geom)
    
    # read resutls
    if loadResults:
        res = read_op2(op2File, build_dataframe=True, debug=False, mode='optistruct');
        resDf = op2ResToDf(res, geom)
        return mesh, resDf
    else:
        return mesh

################################################################################ 
# load the mesh from a pyNastran object and convert it to pyvista. Optionally
# center and scale to unit circle
# !assumes nodes ids are contiguous and start at 1
def op2GeomToPv(geom, center=True, scale=True):
    nodeDict = geom.nodes
    elemDict = geom.elements
    vertices = np.array([n.get_position() for nid, n in nodeDict.items()])
    if center: vertices -= np.mean(vertices, axis=0)
    if scale: vertices /= np.linalg.norm(vertices, axis=1).max()
    faces = np.array([[len(e.nodes)] + [nid-1 for nid in e.nodes] for eid, e in elemDict.items()]).flatten()
    mesh = pv.PolyData(vertices, faces, n_faces=len(elemDict))
    return mesh

################################################################################ 
# load the results from a pyNastran object and convert it to a data frame
# !assumes nodes ids are contiguous and start at 1
def op2ResToDf(res, geom, loadNodalStress=True, colNames=['ux', 'uy']):
    resDf = res.displacements[1].data_frame
    resDf = resDf.set_index('NodeID')
    
    if loadNodalStress:
        colNames += ['sxx', 'syy', 'sxy', 'vonMises']
        stressDf = res.cquad4_stress[1].data_frame        
        stressDf = transformStressToMatCordSys(stressDf, geom)        
        nodeStressDf = stressDf.groupby('NodeID').mean().drop('CEN')
        resDf = resDf.join(nodeStressDf)

    resDf = resDf.rename(columns={'t1':'ux', 't2':'uy', 'oxx':'sxx', 
                      'oyy':'syy','txy':'sxy', 'von_mises':'vonMises'})
    resDf = resDf[colNames]
    return resDf

################################################################################
# convert stresses from the elemental coordinate systems to the global one (slwo)
def transformStressToMatCordSys(stressDf, geom):
    # T = [xmaterial or xelement, y, normal]
    # sigma_elemental = [oxx,txy,0], [txy,oyy,0],[0,0,0]
    for index, row in stressDf.iterrows():
        eid = index[0]
        T = np.vstack(geom.Element(eid).material_coordinate_system())[1:,:]
        stressElemental = np.array([[row.oxx, row.txy, 0], [row.txy, row.oyy, 0], [0, 0, 0]])
        stressElementalGlobal = np.matmul(T.T,np.matmul(stressElemental,T))
        stressDf.at[index,'oxx'] = stressElementalGlobal[0,0]
        stressDf.at[index,'oyy'] = stressElementalGlobal[1,1]
        stressDf.at[index,'txy'] = stressElementalGlobal[1,0]

    return stressDf

################################################################################
# convert stresses from the elemental coordinate systems to the global one (slwo)
def getAllBoundaries(mesh):
    bndEdges = mesh.extract_feature_edges(boundary_edges=True, 
            non_manifold_edges=False, feature_edges=False, manifold_edges=False)
    bndEdgesCon = bndEdges.connectivity()
    bndIds = np.unique(bndEdgesCon['RegionId'])
    bndDict = {}
    for bndId in bndIds:
        cellIds = np.where(bndEdgesCon.cell_arrays['RegionId']==bndId)
        bndDict[bndId] = bndEdgesCon.extract_cells(cellIds)
    
    return bndDict

################################################################################
# convert stresses from the elemental coordinate systems to the global one (slwo)
def processBoundaries(bndDict, perp=[0,0,1]):
    bndLenDict, bndNormDict = {}, {}
    for bndId, bnd in bndDict.items():
        L = 0
        edgeNormals = np.zeros((bnd.n_cells,3))
        com = np.mean(bnd.points, axis=0)
        dirs = np.zeros(bnd.n_cells)

        # get first normal
        vi,vj = bnd.extract_cells(0).points
        nv = np.cross(vi-vj,np.array(perp))
        edgeNormals[0,:] = nv / np.linalg.norm(nv)

        # get the rest
        for eid in range(1, bnd.n_cells):
            vi,vj = bnd.extract_cells(eid).points
            L += np.linalg.norm(vi-vj)
            nv = np.cross(vi-vj,np.array(perp))
            nv /= np.linalg.norm(nv)
            nv *= np.sign(np.dot(nv,edgeNormals[eid-1,:]))
            edgeNormals[eid,:] = nv
            dirs[eid] = np.dot(0.5*(vi+vj)-com, nv)

        edgeNormals = edgeNormals * np.sign(np.mean(dirs)) # point outwards
        bndNormDict[bndId] = edgeNormals
        bndLenDict[bndId] = L
    
    return bndLenDict, bndNormDict
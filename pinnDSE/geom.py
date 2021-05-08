import numpy as np
import pandas as pd
from pyNastran.op2.op2 import read_op2
from pyNastran.op2.op2_geom import read_op2_geom
import pyvista as pv
import trimesh
import deepxde as dde
from time import time
from scipy.sparse import lil_matrix
import pickle
import copy

from .util import *


class MeshGeom(dde.geometry.geometry.Geometry):
    def __init__(self, op2File=None, meshFile=None, pickleFile=None, thickness=None, 
                 center=True, scale=True, toGlobal=True):
        if op2File:
            self.op2File = op2File
            self.dim = 2
            self.idstr = 'MeshGeom'
            self.thickness = thickness

            start = time()
            self.mesh, self.resDf = loadOptistructModel(op2File, loadResults=True, toGlobal=toGlobal)
            print(f'1 total mesh and result loading: {time()-start}')

            if center: 
                # center at origin
                self.mesh.points -= np.mean(self.mesh.points, axis=0)
            if scale:
                # scale to unit circle
                self.scale = 1/np.linalg.norm(self.mesh.points, axis=1).max()
                self.mesh.points *= self.scale
                self.resDf[['sxx', 'syy', 'sxy', 'vonMises']] /= self.scale


            # process edges
            start = time()
            self.bndDict, self.bndNormsDict = processBoundaries(self.mesh)
            self.updateBndLengthAndArea()
            print(f'1 total edge processing: {time()-start}')
        else:
            # alternative format load
            temp = pickle.load(open(pickleFile, 'rb'))
            for key, val in temp.__dict__.items():
                setattr(self, key, val)
            self.mesh = pv.read(meshFile)
            self.bndDict, self.bndNormsDict = processBoundaries(self.mesh)
            self.updateBndLengthAndArea()
        
    def random_points(self, n, random="pseudo", seed=1234):
        samples = sampleDomain(self.mesh, n, seed=seed)
        return samples[:,:2]
    
    def random_boundary_points(self, n, bndId, random="pseudo", seed=1234):
        samples, normals = sampleBoundary(self.bndDict[bndId], n, 
                                          bndNormals=self.bndNormsDict[bndId], 
                                          seed=1234)
        return samples[:,:2], normals[:,:2]
    
    def sampleRes(self, loc):
        fieldList = self.resDf.columns
        pc = pv.PolyData(addZ(loc))
        msh = self.mesh.copy()
        
        for field in fieldList:
            msh[field] = self.resDf[field]

        pc = pc.sample(msh)
        interpolatedData = {}
        for field in fieldList:
            interpolatedData[field] = pc[field]

        intResDf = pd.DataFrame(interpolatedData)
        return intResDf
    
    def saveInAltFormat(self, meshFile=None, pickleFile=None):
        meshFile = meshFile if meshFile else self.op2File.replace('.op2', '.vtk')
        pickleFile = pickleFile if pickleFile else self.op2File.replace('.op2', '.p')
        self.mesh.save(meshFile)
        delattr(self, 'mesh')
        delattr(self, 'bndDict')
        pickle.dump(self, open(pickleFile, 'wb'))
        
    def updateBndLengthAndArea(self):
        self.bndLensDict = {bndId:getBndLength(bnd) for bndId,bnd in self.bndDict.items()}
        self.bndAreaDict = {i:self.thickness*L for i,L in self.bndLensDict.items()}

################################################################################ 
# load the mesh and results from an OptiStruct-generated OP2 file
# !assumes nodes ids are contiguous and start at 1
def loadOptistructModel(op2File, loadResults=True, toGlobal=True):
    # read geometry
    geom = read_op2_geom(op2File, build_dataframe=False, debug=False)
    start = time()
    mesh = op2GeomToPv(geom)
    print(f'2 read mesh: {time()-start}')
    
    # read resutls
    if loadResults:
        start = time()
        res = read_op2(op2File, build_dataframe=True, debug=False, mode='optistruct');
        resDf = op2ResToDf(res, geom,  toGlobal=toGlobal)
        print(f'2 read results: {time()-start}')
        return mesh, resDf
    else:
        return mesh

################################################################################ 
# load the mesh from a pyNastran object and convert it to pyvista. Optionally
# center and scale to unit circle
# !assumes nodes ids are contiguous and start at 1
def op2GeomToPv(geom):
    nodeDict = geom.nodes
    elemDict = geom.elements
    vertices = np.array([n.get_position() for nid, n in nodeDict.items()])
#     if center: vertices -= np.mean(vertices, axis=0)
#     if scale: vertices /= np.linalg.norm(vertices, axis=1).max()
    faces = np.hstack([[len(e.nodes)] + [nid-1 for nid in e.nodes] for eid, e in elemDict.items()]).flatten()
    mesh = pv.PolyData(vertices, faces, n_faces=len(elemDict))
    return mesh

################################################################################ 
# load the results from a pyNastran object and convert it to a data frame
# !assumes nodes ids are contiguous and start at 1
def op2ResToDf(res, geom, loadNodalStress=True, colNames=['ux', 'uy'], toGlobal=True):
    resDf = res.displacements[1].data_frame
    resDf = resDf.set_index('NodeID')
    
    if loadNodalStress:
        colNames += ['sxx', 'syy', 'sxy', 'vonMises']
        stressDf = res.cquad4_stress[1].data_frame  
        stressDf = stressDf.loc[stressDf.index.get_level_values('Location') == 'Top'] # only consider top 
        stressDf = stressDf.loc[stressDf.index.get_level_values('NodeID') != 'CEN'] # only consider vertices 
        stressDf = stressDf.reset_index(level='Location', drop=True) # drop location    
        start = time()
        if toGlobal: stressDf = transformStressToMatCordSys(stressDf, geom)        
        print(f'3 coordinate transform: {time()-start}')
        nodeStressDf = stressDf.groupby('NodeID').mean()
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
# def transformStressToMatCordSys2(stressDf, geom):
#     Tmat = lil_matrix((3*len(stressDf), 3*len(stressDf)))
#     Smat = lil_matrix((3*len(stressDf), 3*len(stressDf)))
#     eidList = stressDf.reset_index()['ElementID'].values
#     stresses = stressDf[['oxx', 'oyy', 'txy']].values
#     for i,eid in enumerate(eidList):
#         # block diagonal matrix of 3x3 coordinate transforms
#         Tmat[3*i:3*i+3,3*i:3*i+3] = np.vstack(geom.Element(eid).material_coordinate_system())[1:,:]
#         # block diagonal matrix of 3x3 stress matrices
#         Smat[3*i:3*i+3,3*i:3*i+3] = np.array([[stresses[i,0], stresses[i,2], 0], [stresses[i,2], stresses[i,1], 0], [0, 0, 0]])
#     stressDf[['oxx', 'oyy', 'txy']] = Tmat.dot(Smat.T).dot(Tmat)
#     return stressDf

################################################################################
# extract the boundaries, their legnths, and edge normals
def processBoundaries(mesh, perp=[0,0,1]):
    # extract edges
    bndEdges = mesh.extract_feature_edges(boundary_edges=True, 
            non_manifold_edges=False, feature_edges=False, manifold_edges=False)
    bndEdgesCon = bndEdges.connectivity()

    # compute normals
    extBnd = bndEdges.extrude(perp)
    extBnd.compute_normals(inplace=True, auto_orient_normals=True) 
    normals = extBnd.cell_arrays['Normals']
    normals = 0.5*(normals[0::2] + normals[1::2])

    # build boundary dictionaries
    bndIds = np.unique(bndEdgesCon['RegionId'])
    bndDict, bndNormsDict = {}, {}
    for bndId in bndIds:
        cellIds = np.where(bndEdgesCon.cell_arrays['RegionId']==bndId)
        bndDict[bndId] = bndEdgesCon.extract_cells(cellIds)
        bndNormsDict[bndId] = normals[cellIds]
        
    return bndDict, bndNormsDict

################################################################################
def getBndLength(bnd):
    L = 0
    for eid in range(bnd.n_cells):
        vi,vj = bnd.extract_cells(eid).points
        L += np.linalg.norm(vi-vj)
    return L

################################################################################
# uniformly sample the mesh using Trimesh
def sampleDomain(mesh, N, seed=1234):
    tmesh = pyvistaToTrimesh(mesh)
    np.random.seed(seed)
    samples, face_index = trimesh.sample.sample_surface_even(tmesh, N)
    return samples

################################################################################ 
# convert pyvista mesh to trimesh mesh by triangulating elements
def pyvistaToTrimesh(mesh):
    pvTriMesh = pv.PolyDataFilters.triangulate(mesh)
    faces = pvTriMesh.faces.reshape(-1,4)[:,1:]
    points = pvTriMesh.points
    return trimesh.Trimesh(points, faces, process=False)

################################################################################ 
# convert pyvista mesh to trimesh mesh by triangulating elements
def sampleBoundary(bnd, N, bndNormals=None, seed=1234):
    np.random.seed(seed)
    samples = np.zeros((N,3))
    normals = np.zeros((N,3))
        
    for n in range(N):
        eid = np.random.choice(range(bnd.n_cells))
        edge = bnd.extract_cells(eid)
        vi,vj = edge.points
        w = np.random.rand()
        p = w*vi+(1-w)*vj
        samples[n,:] = p
        
        if bndNormals is not None:
            normals[n,:] = bndNormals[eid,:]

    return samples, normals

################################################################################
def getCellMask(bnd, component, func):
    return [1 if func(bnd.extract_cells(eid).points) else 0 for eid in range(bnd.n_cells)]

################################################################################
def splitBoundary(bndDict, bndNormsDict, bndId, mask):
    newBndDict = {}
    for key,val in bndDict.items():
        newBndDict[key] = val.copy()
        
    # create a new boundary
    groupA = np.where(np.logical_not(mask))
    newBndDict[bndId] = bndDict[bndId].extract_cells(groupA)
    newEdgeId = len(bndDict)
    groupB = np.where(mask)
    newBndDict[newEdgeId] = bndDict[bndId].extract_cells(groupB)
    
    # update normals in the same way
    newBndNormsDict = copy.deepcopy(bndNormsDict)
    newBndNormsDict[bndId] = bndNormsDict[bndId][groupA]
    newBndNormsDict[newEdgeId] = bndNormsDict[bndId][groupB]
    
    return newBndDict, newBndNormsDict
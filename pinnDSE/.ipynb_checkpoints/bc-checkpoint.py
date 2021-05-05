import numpy as np
import deepxde as dde


################################################################################
# a modified operactorBC for enforcing traction conditions on edges
class TractionBC(dde.boundary_conditions.BC):
    def __init__(self, geom, bndId, T=0, component=0):
        self.geom = geom
        self.on_boundary = lambda x, on_boundary: True   # dummy function
        self.bndId = bndId
        self.traction = T
        self.component = component
        self.func = tractionX if component==0 else tractionY
        
    def error(self, train_x, inputs, outputs, beg, end):
        return self.func(train_x, inputs, outputs, 
                         self.train_n, self.traction)[beg:end]
    
    
################################################################################
# a modified DirichletBC for enforcing displacements on edges
class SupportBC(dde.boundary_conditions.DirichletBC):
    def __init__(self, geom, bndId, U=0, component=0):
        self.geom = geom
        self.func = lambda x: U # enforced displacement
        self.on_boundary = lambda x, on_boundary: True   # dummy function
        self.bndId = bndId
        self.component = component
    
################################################################################
# return residual for traction condition
def tractionX(train_x, inputs, outputs, train_n, Tx): # needs normals and Tx
    sigX = outputs[:,2]
    sigXY = outputs[:,4]
    nx = train_n[:,0]
    ny = train_n[:,1]
    return sigX*nx + sigXY*ny - Tx

################################################################################
# return residual for traction condition
def tractionY(train_x, inputs, outputs, train_n, Ty): # needs normals and Tx
    sigY = outputs[:,3]
    sigXY = outputs[:,4]
    nx = train_n[:,0]
    ny = train_n[:,1]
    return sigXY*nx + sigY*ny - Ty
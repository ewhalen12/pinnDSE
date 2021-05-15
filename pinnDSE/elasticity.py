import numpy as np
import deepxde as dde
from deepxde.utils import get_num_args
from deepxde.backend import tf
from deepxde import config
from deepxde.boundary_conditions import PointSetBC

from .bc import *



class StrctPDE(dde.data.pde.PDE):
    def __init__(
        self,
        geometry,
        pde,
        bcs,
        num_domain=0,
        num_boundary_dir={},
        train_distribution="sobol",
        anchors=None,
        exclusions=None,
        solution=None,
        num_test=None,
        auxiliary_var_function=None,
        paramDict=None
    ):
        self.geom = geometry
        self.pde = pde
        self.bcs = bcs if isinstance(bcs, (list, tuple)) else [bcs]

        self.num_domain = num_domain
        self.num_boundary_dir = num_boundary_dir
        if train_distribution not in ["uniform", "pseudo", "sobol"]:
            raise ValueError(
                "train_distribution == {}. Available choices: {{'uniform'|'pseudo'|'sobol'}}.".format(
                    train_distribution
                )
            )
        self.train_distribution = train_distribution
        self.anchors = anchors
        self.exclusions = exclusions

        self.soln = solution
        self.num_test = num_test

        self.auxiliary_var_fn = auxiliary_var_function
        self.paramDict = paramDict

        self.train_x_all = None
        self.train_x, self.train_y = None, None
        self.train_x_bc = None
        self.num_bcs = None
        self.test_x, self.test_y = None, None
        self.train_aux_vars, self.test_aux_vars = None, None

        # sample domain and each boundary
        self.domainSamples = self.geom.random_points(num_domain)
        self.bndSampleDict = {}
        self.bndNormalsDict = {}
        for bndId in self.geom.bndDict.keys():
            samples, normals = self.geom.random_boundary_points(num_boundary_dir[bndId], bndId)
            self.bndSampleDict[bndId] = samples
            self.bndNormalsDict[bndId] = normals
    
        # organize by which bc they belong to
#         x_bcs = [self.bndSampleDict[bc.bndId] for bc in self.bcs] # bcs will have a bndId attribute
#         n_bcs = [self.bndNormalsDict[bc.bndId] for bc in self.bcs]
        x_bcs, n_bcs = [], []
        for bc in self.bcs:
            if isinstance(bc, (TractionBC, SupportBC)): 
                x_bcs.append(self.bndSampleDict[bc.bndId])
                n_bcs.append(self.bndNormalsDict[bc.bndId])
            elif isinstance(bc, PointSetBC): 
                x_bcs.append(bc.points)
                n_bcs.append(np.zeros_like(bc.points))
            
        self.num_bcs = list(map(len, x_bcs))
        self.train_x_bc = (np.vstack(x_bcs) if x_bcs else np.empty([0, self.train_x_all.shape[-1]]))
        self.train_n_bc = (np.vstack(n_bcs) if n_bcs else np.empty([0, self.train_x_all.shape[-1]]))
        
        # create final training arrays
        self.train_x_all = np.vstack([self.domainSamples]+list(self.bndSampleDict.values()))
        self.train_x = np.vstack((self.train_x_bc, self.train_x_all)) # stack bc and then all points (contains duplicates)
        self.train_n = np.vstack((self.train_n_bc, np.zeros_like(self.train_x_all)))
        self.train_y = self.soln(self.train_x) if self.soln else None
        
        self.test_x = self.geom.random_points(self.num_test, seed=1)
        self.test_y = self.soln(self.test_x) if self.soln else None
        
    ################################################################################
    # re-writing the dde losses function to enable parameter passing to the pde()
    # function which we will use to store material constants
    def losses(self, targets, outputs, loss, model):
        f = []
        # Always build the gradients in the PDE here, so that we can reuse all the gradients in dde.grad. If we build
        # the gradients in losses_train(), then error occurs when we use these gradients in losses_test() during
        # sess.run(), because one branch in tf.cond() cannot use the Tensors created in the other branch.
        if self.pde is not None:
            if get_num_args(self.pde) == 3:
                f = self.pde(model.net.inputs, outputs, self.paramDict)
            elif get_num_args(self.pde) == 4:
                if self.auxiliary_var_fn is None:
                    raise ValueError("Auxiliary variable function not defined.")
                f = self.pde(model.net.inputs, outputs, self.paramDict, model.net.auxiliary_vars)
            if not isinstance(f, (list, tuple)):
                f = [f]

        if not isinstance(loss, (list, tuple)):
            loss = [loss] * (len(f) + len(self.bcs))
        elif len(loss) != len(f) + len(self.bcs):
            raise ValueError(
                "There are {} errors, but only {} losses.".format(
                    len(f) + len(self.bcs), len(loss)
                )
            )

        def losses_train():
            f_train = f
            bcs_start = np.cumsum([0] + self.num_bcs)
            error_f = [fi[bcs_start[-1] :] for fi in f_train]
            losses = [
                loss[i](tf.zeros(tf.shape(error), dtype=config.real(tf)), error)
                for i, error in enumerate(error_f)
            ]
            for i, bc in enumerate(self.bcs):
                beg, end = bcs_start[i], bcs_start[i + 1]
                error = bc.error(self.train_x, model.net.inputs, outputs, beg, end)
                losses.append(
                    loss[len(error_f) + i](
                        tf.zeros(tf.shape(error), dtype=config.real(tf)), error
                    )
                )
            return losses

        def losses_test():
            f_test = f
            return [
                loss[i](tf.zeros(tf.shape(fi), dtype=config.real(tf)), fi)
                for i, fi in enumerate(f_test)
            ] + [tf.constant(0, dtype=config.real(tf)) for _ in self.bcs]

        return tf.cond(tf.equal(model.net.data_id, 0), losses_train, losses_test)
            

################################################################################
# calculates the residuals of the elasticity equations, namely equations of
# motion and the contituitive laws for linear isotropic materials
def elasticityEqs(inputs, outputs, paramDict):
    # get the spatial derivatives of stress
    dSigX_dx = dde.gradients.jacobian(outputs, inputs, i=2, j=0)
    dSigY_dy = dde.gradients.jacobian(outputs, inputs, i=3, j=1)
    dSigXY_dx = dde.gradients.jacobian(outputs, inputs, i=4, j=0)
    dSigXY_dy = dde.gradients.jacobian(outputs, inputs, i=4, j=1)

    # check equations of motion
    divSigX = dSigX_dx + dSigXY_dy
    divSigY = dSigXY_dx + dSigY_dy

    # get the spatial derivatives of displacement
    dU_dx = dde.gradients.jacobian(outputs, inputs, i=0, j=0)
    dU_dy = dde.gradients.jacobian(outputs, inputs, i=0, j=1)
    dV_dx = dde.gradients.jacobian(outputs, inputs, i=1, j=0)
    dV_dy = dde.gradients.jacobian(outputs, inputs, i=1, j=1)

    # check constitutive laws
    E = paramDict['E']
    nu = paramDict['nu']
    sigX, sigY, sigXY = outputs[:, 2:3], outputs[:, 3:4], outputs[:, 4:]
    resSigX = (E/(1-nu**2))*(dU_dx + nu*dV_dy) - sigX
    resSigXY = (E/(2*(1+nu)))*(dU_dy + dV_dx) - sigXY
    resSigY = (E/(1-nu**2))*(dV_dy + nu*dU_dx) - sigY
    return [divSigX, divSigY, resSigX, resSigY, resSigXY]
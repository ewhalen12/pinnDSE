import numpy as np
import pandas as pd
import pyvista as pv
from time import time

from .util import *

################################################################################ 
# plot side-by-side contours of the mesh with specified results and columns
def plotScalarFields(mesh, resDf, fieldList=['ux', 'uy', 'sxx', 'syy', 'sxy'], cpos='xy', 
                     showEdges=False, pointSize=4, bar='hor'):
    plotter = pv.Plotter(shape=(1,len(fieldList)), border=False)
    for i,field in enumerate(fieldList):
        plotter.subplot(0,i)
        plotter.set_background('white')
        plotter.add_text(field, color='k', font_size=7)
        if mesh.n_cells == mesh.n_points:
            # point cloud
            plotter.add_mesh(mesh.copy(), scalars=resDf[field], 
                             render_points_as_spheres=True, point_size=pointSize)
        else:
            # mesh
            plotter.add_mesh(mesh.copy(), scalars=resDf[field], show_edges=showEdges)
            
        if bar=='ver':
            plotter.add_scalar_bar(n_labels=2, label_font_size=10, width=0.1, height=0.3, 
                               vertical=True, position_x=0.65, fmt="%.1e", color='k')
        else:
            plotter.add_scalar_bar(n_labels=2, label_font_size=10, width=0.3, height=0.1, 
                        font_family='arial', vertical=False, position_x=0.6, fmt="%.1e", color='k')
    
    plotter.show(window_size=(200*len(fieldList),200), cpos=cpos);
    
################################################################################ 
# plot a single mesh or point cloud with optional scalar field
def plotField(x, mesh=None, scalar=None, interpolate=False):
    # point coordinates
    pc = pv.PolyData(addZ(x))

    plotter = pv.Plotter(border=False)
    plotter.set_background('white')
    if interpolate:
        pc['field'] = scalar
        vertScalars = mesh.copy().interpolate(pc)['field']
        plotter.add_mesh(mesh, show_edges=False, scalars=vertScalars)
        plotter.add_scalar_bar(n_labels=2, label_font_size=10, width=0.1, height=0.3, 
                                   vertical=True, position_x=0.65, fmt="%.1e", color='k')
    else:
        if mesh is not None:
            plotter.add_mesh(mesh, show_edges=False)
        if scalar is not None:
            plotter.add_mesh(pc, render_points_as_spheres=True, scalars=scalar)
            plotter.add_scalar_bar(n_labels=2, label_font_size=10, width=0.1, height=0.3, 
                                   vertical=True, position_x=0.65, fmt="%.1e", color='k')
        else:
            plotter.add_mesh(pc, render_points_as_spheres=True, color='gray')
    
    plotter.show(window_size=(400,400), cpos='xy');

################################################################################
def drawBoundaries(bndDict, lineWidth=2, offset=[0,0.05,0]):
    # make boundary markers
    labels = [str(bndId) for bndId in bndDict.keys()]
    pos = np.vstack([np.mean(bnd.points, axis=0) for _,bnd in bndDict.items()])

    plotter = pv.Plotter(border=False)
    plotter.set_background('white')
    colors = ['black', 'red', 'green', 'blue', 'orange', 'purple']
    for bndId,color in zip(bndDict.keys(), colors):
        bnd = bndDict[bndId]
        plotter.add_mesh(bnd, show_edges=True, color=color, line_width=lineWidth)
        plotter.add_point_labels(pos[bndId]+offset, labels[bndId], shadow=False, text_color=color, shape=None, font_size=20)
    plotter.show(window_size=(300,300), cpos='xy');
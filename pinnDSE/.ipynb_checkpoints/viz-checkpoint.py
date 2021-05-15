import numpy as np
import pandas as pd
import altair as alt
import pyvista as pv
from time import time

from .util import *

# color schemes
CATEGORY20 = ['3182bd', 'e6550d', '31a354', '756bb1', '636363', 'c6dbef', 'fdd0a2', 'c7e9c0', 'dadaeb'] # modified
CATEGORY10 = ['1f77b4', 'ff7f0e', '2ca02c', 'd62728', '9467bd', '8c564b', 'e377c2', '7f7f7f', 'bcbd22', '17becf']

################################################################################ 
# plot side-by-side contours of the mesh with specified results and columns
def plotScalarFields(mesh, resDf, fieldList=['ux', 'uy', 'sxx', 'syy', 'sxy'], cpos='xy', 
                     showEdges=False, pointSize=4, bar='hor', bndDict=None, shape=None, size=200):
    if shape==None: shape=(1,len(fieldList))
    print(shape)
    plotter = pv.Plotter(shape=shape, border=False)
    for i in range(shape[0]):
        for j in range(shape[1]):
            index = i*shape[1]+j
            if index < len(fieldList):
                field = fieldList[index]

                plotter.subplot(i,j)
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

                if bndDict:
                    for bndId,bnd in bndDict.items():
                         plotter.add_mesh(bnd.copy(), color='k')
    
    windowSize = (shape[1]*size,shape[0]*size)
    plotter.show(window_size=windowSize, cpos=cpos);
    
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
def drawBoundaries(bndDict, lineWidth=2, offset=[0,0.05,0], colors=CATEGORY10):
    # make boundary markers
    labels = [str(bndId) for bndId in bndDict.keys()]
    pos = np.vstack([np.mean(bnd.points, axis=0) for _,bnd in bndDict.items()])

    plotter = pv.Plotter(border=False)
    plotter.set_background('white')
    for bndId,color in zip(bndDict.keys(), colors):
        bnd = bndDict[bndId]
        plotter.add_mesh(bnd, show_edges=True, color=color, line_width=lineWidth)
        plotter.add_point_labels(pos[bndId]+offset, labels[bndId], shadow=False, text_color=color, shape=None, font_size=20)
    plotter.show(window_size=(300,300), cpos='xy');
    
################################################################################
def lossPlot(losshistory, bcNames, dropFirstStep=False, scaleType='log', 
             number=True, pdeTermNames=['divSigX','divSigY','resSigX','resSigY','resSigXY'], scheme='category10'):
    columns = pdeTermNames+bcNames
    if number: columns = [f'{i:02}  {c}' for i,c in enumerate(columns)]
    lossDf = pd.DataFrame(np.vstack(losshistory.loss_train), columns=columns)
    lossDf['step'] = np.array(losshistory.steps)
    if dropFirstStep: lossDf = lossDf.drop(0)

    return alt.Chart(lossDf).transform_fold(
        columns,
        as_=['loss term', 'value']
    ).mark_line().encode(
        x='step:Q',
        y=alt.Y('value:Q', scale=alt.Scale(type=scaleType)),
        color=alt.Color('loss term:N', sort=columns, scale=alt.Scale(scheme=scheme)),
        tooltip=['loss term:N', 'value:Q', 'step:Q']
    )

    
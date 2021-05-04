import numpy as np
import pandas as pd
import pyvista as pv


################################################################################ 
# plot side-by-side contours of the mesh with specified results and columns
def plotScalarFields(mesh, resDf, fieldList=['ux', 'uy', 'sxx', 'syy', 'sxy'], cpos='xy'):
    plotter = pv.Plotter(shape=(1,len(fieldList)), border=False)
    for i,field in enumerate(fieldList):
        plotter.subplot(0,i)
        plotter.set_background('white')
        plotter.add_text(field, color='k', font_size=7)
        plotter.add_mesh(mesh.copy(), show_edges=False, scalars=resDf[field])
        plotter.add_scalar_bar(n_labels=2, label_font_size=10, width=0.1, height=0.3, 
                               vertical=True, position_x=0.65, fmt="%.1e", color='k')
    
    plotter.show(window_size=(200*len(fieldList),200), cpos=cpos);
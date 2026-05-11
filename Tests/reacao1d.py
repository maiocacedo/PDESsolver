import numpy as np
import time
import os, sys

diretorio_atual = os.path.dirname(os.path.abspath(__file__))
diretorio_pai = os.path.abspath(os.path.join(diretorio_atual, '..'))
sys.path.append(diretorio_pai)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PDES import PDES
import PDE


disc_n  = [21]
tf      = 5.0
nt      = 1000

x_lin = np.linspace(0, 1, disc_n[0])
X = np.meshgrid(x_lin, indexing='ij')

v = 0.5
dax = 0.001

PDE_C = PDE.PDE(
    f'dC/dt = -{v}*(dC/dx) + {dax}*(d2C/dx2)'
    f' - 0.1*C',
    'C', ['x'], ['t'],
    ivar_boundary=[(0, 1), (0, 1)],
    expr_ic='1e-6',
    west_bd="Dirichlet",  west_func_bd="1",
    east_bd="Neumann",  east_func_bd="0",
)

PDE_D = PDE.PDE(
    f'dD/dt = -{v}*(dD/dx) + {dax}*(d2D/dx2)'
    f' + 0.1*C',
    'D', ['x'], ['t'],
    ivar_boundary=[(0, 1), (0, 1)],
    expr_ic='0',
    west_bd="Dirichlet",  west_func_bd="0",
    east_bd="Neumann",  east_func_bd="0",
)

PDES1 = PDES([PDE_C, PDE_D], disc_n)

PDES1.discretize(method="backward")
PDES1.solve(method='bdf2', tf=tf, nt=nt, tol=1e-6)
resultado_final_cn = PDES1.results

PDES1.visualize(mode='plot1d', func_idx=0)
PDES1.visualize(mode='plot1d', func_idx=0, time_step=0)

PDES1.visualize(mode='plot1d_all', func_idx=0, n_profiles=10, tf=5.0, cmap='plasma')

PDES1.visualize(mode='heatmap1d', func_idx=0, tf=5.0, cmap='viridis')

PDES1.visualize(mode='animation1d', func_idx=1, tf=5.0, interval=5)
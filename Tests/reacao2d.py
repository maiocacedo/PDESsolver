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

# ---------------------------------------------------------------------------
# Parâmetros
# ---------------------------------------------------------------------------
disc_n  = [11, 11]
tf      = 5.0
nt      = 1000

x_lin = np.linspace(0, 1, disc_n[0])
y_lin = np.linspace(0, 1, disc_n[1])
X, Y  = np.meshgrid(x_lin, y_lin, indexing='ij')

v = 0.5
dax = 0.001

PDE_C = PDE.PDE(
    f'dC/dt = -{v}*(dC/dx+dC/dy) + {dax}*(d2C/dx2+d2C/dy2)'
    f' - 0.1*C',
    'C', ['x', 'y'], ['t'],
    ivar_boundary=[(0, 1), (0, 1)],
    expr_ic='1e-6',
    west_bd="Dirichlet",  west_func_bd="1",
    east_bd="Neumann",  east_func_bd="0",
    north_bd="Neumann", north_func_bd="0",
    south_bd="Neumann", south_func_bd="0",
)

PDE_D = PDE.PDE(
    f'dD/dt = -{v}*(dD/dx+dD/dy) + {dax}*(d2D/dx2+d2D/dy2)'
    f' + 0.1*C',
    'D', ['x', 'y'], ['t'],
    ivar_boundary=[(0, 1), (0, 1)],
    expr_ic='0',
    west_bd="Dirichlet",  west_func_bd="0",
    east_bd="Neumann",  east_func_bd="0",
    north_bd="Neumann", north_func_bd="0",
    south_bd="Neumann", south_func_bd="0",
)

PDES1 = PDES([PDE_C, PDE_D], disc_n)

PDES1.discretize(method="central")
PDES1.solve(method='bdf2', tf=tf, nt=nt, tol=1e-6)
resultado_final_cn = PDES1.results

PDES1.visualize(mode='plot3d', func_idx=0, cmap="RdYlBu_r") # Mostra o calor de F
PDES1.visualize(mode='plot3d', func_idx=1, cmap="RdYlBu_r") # Mostra o calor de F



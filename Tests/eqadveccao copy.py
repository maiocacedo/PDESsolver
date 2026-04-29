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
disc_n  = [10, 10]
nu      = 0.1
cx, cy  = 1.0, 1.0
tf      = 1.0
nt      = 200

x_lin = np.linspace(0, 1, disc_n[0])
y_lin = np.linspace(0, 1, disc_n[1])
X, Y  = np.meshgrid(x_lin, y_lin, indexing='ij')


def analitica(X, Y, t):
    return (np.exp(-2 * nu * np.pi**2 * t)
            * np.sin(np.pi * (X - cx * t))
            * np.sin(np.pi * (Y - cy * t)))
# ---------------------------------------------------------------------------
# EDP com termo fonte embutido
# ---------------------------------------------------------------------------
v = 0
dax = 10e-5

PDE_C = PDE.PDE(
    f'dU/dt = -{v}*(dU/dx+dU/dy) + {dax}*(d2U/dx2+d2U/dy2)'
    f' - 0.1*U*V',
    'U', ['x', 'y'], ['t'],
    ivar_boundary=[(0, 1), (0, 1)],
    expr_ic='0'
)

PDE_D = PDE.PDE(
    f'dV/dt = -{v}*(dV/dx+dV/dy) + {dax}*(d2V/dx2+d2V/dy2)'
    f' + 2*U*V',
    'V', ['x', 'y'], ['t'],
    ivar_boundary=[(0, 1), (0, 1)],
    expr_ic='0'
)

PDES1 = PDES([PDE_C, PDE_D], disc_n)

# BCs Dirichlet dependentes do tempo — o solver atualiza via d_dt() a cada passo
bc_expr = "exp(-2*0.1*pi**2*t) * sin(pi*x) * sin(pi*y)"
PDES1.discretize(
    method="central",
    west_bd="Dirichlet",  west_func_bd="1", 
    east_bd="Neumann",  east_func_bd="0",
    north_bd="Neumann", north_func_bd="0",
    south_bd="Neumann", south_func_bd="0",
)

resultado_analitico = analitica(X, Y, tf).flatten().tolist()

PDES1.solve(method='RKF', tf=tf, nt=100, tol=1e-6)
resultado_final_rk = PDES1.results
mae_resultado_final_rk = sum(abs(r - p) for r, p in zip(resultado_analitico, resultado_final_rk[0])) / len(resultado_analitico)
print(mae_resultado_final_rk)

PDES1.solve(method='CN', tf=tf, nt=100, tol=1e-6)
resultado_final_cn = PDES1.results
mae_resultado_final_cn = sum(abs(r - p) for r, p in zip(resultado_analitico, resultado_final_cn[0])) / len(resultado_analitico)
print(mae_resultado_final_cn)

# PDES1.solve(method='bdf2', tf=tf, nt=100, tol=1e-6)
# resultado_final_bdf2 = PDES1.results
# mae_resultado_final_bdf2 = sum(abs(r - p) for r, p in zip(resultado_analitico, resultado_final_bdf2[0])) / len(resultado_analitico)
# print(mae_resultado_final_bdf2)

PDES1.visualize(mode='plot3d', func_idx=0, cmap="RdYlBu_r") # Mostra o calor de F
PDES1.visualize(mode='plot3d', func_idx=1, cmap="RdYlBu_r") # Mostra o calor de F



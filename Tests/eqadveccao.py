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
PDE_adv = PDE.PDE(
    f'dU/dt = -{cx}*dU/dx + -{cy}*dU/dy'
    f' + {nu}*d2U/dx2 + {nu}*d2U/dy2'
    f' + {cx}*pi*exp(-2*{nu}*pi**2*t)*(cos(pi*x)*sin(pi*y) + sin(pi*x)*cos(pi*y))',
    'U', ['x', 'y'], ['t'],
    ivar_boundary=[(0, 1), (0, 1)],
    expr_ic='sin(pi * x) * sin(pi * y)'
)

PDES1 = PDES([PDE_adv], disc_n)

# BCs Dirichlet dependentes do tempo — o solver atualiza via d_dt() a cada passo
bc_expr = "exp(-2*0.1*pi**2*t) * sin(pi*x) * sin(pi*y)"
PDES1.discretize(
    method="central",
    west_bd="Dirichlet",  west_func_bd=bc_expr, 
    east_bd="Dirichlet",  east_func_bd=bc_expr,
    north_bd="Dirichlet", north_func_bd=bc_expr,
    south_bd="Dirichlet", south_func_bd=bc_expr,
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

PDES1.solve(method='bdf2', tf=tf, nt=100, tol=1e-6)
resultado_final_bdf2 = PDES1.results
mae_resultado_final_bdf2 = sum(abs(r - p) for r, p in zip(resultado_analitico, resultado_final_bdf2[0])) / len(resultado_analitico)
print(mae_resultado_final_bdf2)

PDES1.visualize(mode='plot3d', func_idx=0, cmap="RdYlBu_r") # Mostra o calor de F



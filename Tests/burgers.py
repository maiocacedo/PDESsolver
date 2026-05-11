import numpy as np
import os, sys

diretorio_atual = os.path.dirname(os.path.abspath(__file__))
diretorio_pai = os.path.abspath(os.path.join(diretorio_atual, '..'))
sys.path.append(diretorio_pai)

import matplotlib.pyplot as plt
from PDES import PDES
import PDE

disc_n = [11, 11]
nu     = 0.1
tf     = 1.0
nt     = 500

x_lin = np.linspace(0, 1, disc_n[0])
y_lin = np.linspace(0, 1, disc_n[1])
X, Y  = np.meshgrid(x_lin, y_lin, indexing='ij')


def analitica(X, Y, t):
    return 1.0 / (1.0 + np.exp((X + Y - t) / (2 * nu)))

bc_expr = f"1 / (1 + exp((x + y - t) / (2*{nu})))"

ic_expr = f"1 / (1 + exp((x + y) / (2*{nu})))"

PDE_burgers = PDE.PDE(
    f'dU/dt + U*dU/dx + U*dU/dy = {nu}*d2U/dx2 + {nu}*d2U/dy2',
    'U', ['x', 'y'], ['t'],
    ivar_boundary=[(0, 1), (0, 1)],
    expr_ic=ic_expr,
    west_bd="Dirichlet",  west_func_bd=bc_expr,
    east_bd="Dirichlet",  east_func_bd=bc_expr,
    north_bd="Dirichlet", north_func_bd=bc_expr,
    south_bd="Dirichlet", south_func_bd=bc_expr,
)

PDES1 = PDES([PDE_burgers], disc_n)

PDES1.discretize(
    method="backward",

)

resultado_analitico = analitica(X, Y, tf).flatten().tolist()

PDES1.solve(method='RKF', tf=tf, nt=nt, tol=1e-6)
resultado_final_rk = PDES1.results
mae_rk = sum(abs(r - p) for r, p in zip(resultado_analitico, resultado_final_rk[0])) / len(resultado_analitico)
print(f"MAE RKF  : {mae_rk:.6e}")

PDES1.solve(method='CN', tf=tf, nt=nt, tol=1e-6)
resultado_final_cn = PDES1.results
mae_cn = sum(abs(r - p) for r, p in zip(resultado_analitico, resultado_final_cn[0])) / len(resultado_analitico)
print(f"MAE CN   : {mae_cn:.6e}")


PDES1.solve(method='bdf2', tf=tf, nt=nt, tol=1e-6)
resultado_final_bdf2 = PDES1.results
mae_bdf2 = sum(abs(r - p) for r, p in zip(resultado_analitico, resultado_final_bdf2[0])) / len(resultado_analitico)
print(f"MAE BDF2 : {mae_bdf2:.6e}")

PDES1.visualize(mode='plot3d', func_idx=0, cmap="RdYlBu_r")

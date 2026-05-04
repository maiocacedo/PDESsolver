import numpy as np
import os, sys

diretorio_atual = os.path.dirname(os.path.abspath(__file__))
diretorio_pai = os.path.abspath(os.path.join(diretorio_atual, '..'))
sys.path.append(diretorio_pai)

from PDES import PDES
import PDE
import time

start_time_df = time.time()

alpha_x = 0.1
alpha_y = 0.2

disc_n = [10, 10]

PDE1 = PDE.PDE(
    f'dF/dt = {alpha_x}*d2F/dx2 + {alpha_y}*d2F/dy2',
    'F', ['x', 'y'], ['t'],
    ivar_boundary=[(0, 1), (0, 1)],
    expr_ic='sin(pi * x) * sin(pi * y)',
    west_bd="Dirichlet",  west_func_bd="0",
    east_bd="Dirichlet",  east_func_bd="0",
    north_bd="Dirichlet", north_func_bd="0",
    south_bd="Dirichlet", south_func_bd="0",
)


def analitica_2d(X, Y, t, alpha_x=alpha_x, alpha_y=alpha_y, n_termos=50):
    u = np.zeros_like(X)
    for m in range(1, n_termos, 2):
        coeficiente = 4 / (m * np.pi)
        espacial = np.sin(np.pi * X) * np.sin(m * np.pi * Y)
        decaimento = np.exp(-(np.pi**2) * (alpha_x + alpha_y * (m**2)) * t)
        u += coeficiente * espacial * decaimento
    return u.flatten().tolist()

resultado_analitico = analitica_2d(
    *np.meshgrid(np.linspace(0, 1, disc_n[0]), np.linspace(0, 1, disc_n[1])), t=1.0
)

PDES1 = PDES([PDE1], disc_n)

PDES1.discretize(method="central")

PDES1.solve(method='RKF', tf=1.0, nt=100, tol=1e-6)
resultado_final_rk = PDES1.results
mae_resultado_final_rk = sum(abs(r - p) for r, p in zip(resultado_analitico, resultado_final_rk[0])) / len(resultado_analitico)
print(mae_resultado_final_rk)

PDES1.solve(method='CN', tf=1.0, nt=100, tol=1e-6)
resultado_final_cn = PDES1.results
mae_resultado_final_cn = sum(abs(r - p) for r, p in zip(resultado_analitico, resultado_final_cn[0])) / len(resultado_analitico)
print(mae_resultado_final_cn)

PDES1.solve(method='bdf2', tf=1.0, nt=100, tol=1e-6)
resultado_final_bdf2 = PDES1.results
mae_resultado_final_bdf2 = sum(abs(r - p) for r, p in zip(resultado_analitico, resultado_final_bdf2[0])) / len(resultado_analitico)
print(mae_resultado_final_bdf2)

PDES1.visualize(mode='plot3d', func_idx=0, cmap="RdYlBu_r")
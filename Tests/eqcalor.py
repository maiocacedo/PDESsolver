import sys
import os

diretorio_atual = os.path.dirname(os.path.abspath(__file__))
diretorio_pai = os.path.abspath(os.path.join(diretorio_atual, '..'))
sys.path.append(diretorio_pai)

from PDES import PDES
import PDE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
import time

start_time_df = time.time()

disc_n = [10, 10]

PDE1 = PDE.PDE('dF/dt =  0.1*d2F/dx2 + 0.2*d2F/dy2',
               'F', ['x', 'y'], ['t'], ivar_boundary=[(0, 1), (0, 1)], expr_ic='sin(pi * x) * sin(pi * y)')
PDE2 = PDE.PDE('dT/dt =  0.1*d2T/dx2 + 0.2*d2T/dy2',
               'T', ['x', 'y'], ['t'], ivar_boundary=[(0, 1), (0, 1)], expr_ic='sin(pi * x) * sin(pi * y)')

def analitica_2d(X, Y, t, alpha_x=0.1, alpha_y=0.2, n_termos=50):
    u = np.zeros_like(X)
    for m in range(1, n_termos, 2):  # m = 1, 3, 5... (apenas ímpares)
        coeficiente = 4 / (m * np.pi)
        espacial = np.sin(np.pi * X) * np.sin(m * np.pi * Y)
        decaimento = np.exp(-(np.pi**2) * (alpha_x + alpha_y * (m**2)) * t)
        u += coeficiente * espacial * decaimento
    return u.flatten().tolist()

resultado_analitico = analitica_2d(*np.meshgrid(np.linspace(0, 1, disc_n[0]), np.linspace(0, 1, disc_n[1])), t=1.0)

PDES1 = PDES([PDE1, PDE2], disc_n)

resultado = PDES1.discretize(
    west_func_bd="0",
    west_bd="Dirichlet",
    method="central",
    north_bd="Dirichlet",
    south_bd="Dirichlet",
    east_bd="Dirichlet",
    north_func_bd='0',
    south_func_bd='0',
    east_func_bd='0'
)

PDES1.solve(method='CN', tf=1.0, nt=100, tol=1e-6)
resultado_final_cn = PDES1.results

print("Resultado Final da Simulação cn:")
print(resultado_final_cn)

PDES1.solve(method='RKF', tf=1.0, nt=100, tol=1e-6)
resultado_final_rk = PDES1.results

print("Resultado Final da Simulação RK:")
print(resultado_final_rk)

PDES1.visualize(mode='heatmap', func_idx=0) # Mostra o calor de F
PDES1.visualize(mode='animation', func_idx=1) # Anima a função T
"""
burgers2d.py
------------
Validação da equação de Burgers 2D viscosa.

EDP resolvida:
    ∂u/∂t + u·∂u/∂x + u·∂u/∂y = ν·(∂²u/∂x² + ∂²u/∂y²)

Solução analítica exata (onda viajante):
    u(x, y, t) = 1 / (1 + exp((x + y - t) / (2ν)))

Verificação:
    ∂u/∂t        = u²·(1-u) / (2ν)         ... (i)
    u·∂u/∂x      = -u²·(1-u) / (2ν)        ... (ii)  (idem para y)
    ν·∂²u/∂x²   = u²·(1-u)·(2u-1) / (4ν²) ... (iii) (idem para y)

    Somando (i) + (ii) + (ii) + 2*(iii) = 0  ✓

BCs: Dirichlet dependentes do tempo u = 1 / (1 + exp((x+y-t)/(2*nu)))
CI:  u(x, y, 0) = 1 / (1 + exp((x+y) / (2*nu)))

Domínio: x ∈ [0,1], y ∈ [0,1], t ∈ [0, tf]
"""

import numpy as np
import os, sys

diretorio_atual = os.path.dirname(os.path.abspath(__file__))
diretorio_pai = os.path.abspath(os.path.join(diretorio_atual, '..'))
sys.path.append(diretorio_pai)

import matplotlib.pyplot as plt
from PDES import PDES
import PDE

# ---------------------------------------------------------------------------
# Parâmetros
# ---------------------------------------------------------------------------
disc_n = [11, 11]
nu     = 0.1
tf     = 0.5
nt     = 200

x_lin = np.linspace(0, 1, disc_n[0])
y_lin = np.linspace(0, 1, disc_n[1])
X, Y  = np.meshgrid(x_lin, y_lin, indexing='ij')

# ---------------------------------------------------------------------------
# Solução analítica exata
# ---------------------------------------------------------------------------
def analitica(X, Y, t):
    return 1.0 / (1.0 + np.exp((X + Y - t) / (2 * nu)))

# ---------------------------------------------------------------------------
# BC dependente do tempo (avaliada pelo solver a cada passo)
# ---------------------------------------------------------------------------
bc_expr = f"1 / (1 + exp((x + y - t) / (2*{nu})))"

# ---------------------------------------------------------------------------
# CI
# ---------------------------------------------------------------------------
ic_expr = f"1 / (1 + exp((x + y) / (2*{nu})))"

# ---------------------------------------------------------------------------
# EDP: Burgers 2D viscosa completa
# ---------------------------------------------------------------------------
PDE_burgers = PDE.PDE(
    f'dU/dt + U*dU/dx + U*dU/dy = {nu}*d2U/dx2 + {nu}*d2U/dy2',
    'U', ['x', 'y'], ['t'],
    ivar_boundary=[(0, 1), (0, 1)],
    expr_ic=ic_expr
)

PDES1 = PDES([PDE_burgers], disc_n)

PDES1.discretize(
    method="central",
    west_bd="Dirichlet",  west_func_bd=bc_expr,
    east_bd="Dirichlet",  east_func_bd=bc_expr,
    north_bd="Dirichlet", north_func_bd=bc_expr,
    south_bd="Dirichlet", south_func_bd=bc_expr,
)

# ---------------------------------------------------------------------------
# Referência analítica em tf
# ---------------------------------------------------------------------------
resultado_analitico = analitica(X, Y, tf).flatten().tolist()

# ---------------------------------------------------------------------------
# RKF
# ---------------------------------------------------------------------------
PDES1.solve(method='RKF', tf=tf, nt=nt, tol=1e-6)
resultado_final_rk = PDES1.results
mae_rk = sum(abs(r - p) for r, p in zip(resultado_analitico, resultado_final_rk[0])) / len(resultado_analitico)
print(f"MAE RKF  : {mae_rk:.6e}")

# ---------------------------------------------------------------------------
# CN
# ---------------------------------------------------------------------------
PDES1.solve(method='CN', tf=tf, nt=nt, tol=1e-6)
resultado_final_cn = PDES1.results
mae_cn = sum(abs(r - p) for r, p in zip(resultado_analitico, resultado_final_cn[0])) / len(resultado_analitico)
print(f"MAE CN   : {mae_cn:.6e}")

# ---------------------------------------------------------------------------
# BDF2
# ---------------------------------------------------------------------------
PDES1.solve(method='bdf2', tf=tf, nt=nt, tol=1e-6)
resultado_final_bdf2 = PDES1.results
mae_bdf2 = sum(abs(r - p) for r, p in zip(resultado_analitico, resultado_final_bdf2[0])) / len(resultado_analitico)
print(f"MAE BDF2 : {mae_bdf2:.6e}")

# ---------------------------------------------------------------------------
# Visualização
# ---------------------------------------------------------------------------
PDES1.visualize(mode='plot3d', func_idx=0, cmap="RdYlBu_r")

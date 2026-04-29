"""
fitzhugh_nagumo_2d_visual.py
----------------------------
Simulação visual do sistema FitzHugh-Nagumo 2D.

Sem MMS — roda a EDP real para ver os padrões de ondas que emergem.

Sistema:
    ∂u/∂t = Du·∇²u + u - u³/3 - v
    ∂v/∂t = Dv·∇²v + ε·(u - γ·v + β)

Parâmetros clássicos para ondas espirais:
    Du = 1.0,  Dv = 0.0  (v sem difusão — variável de recuperação local)
    ε  = 0.08, γ  = 0.5, β = 0.7

CI perturbada via Heaviside:
    u(x,y,0) = 1  se x < 0.5 e y > 0.5, -1 caso contrário
    v(x,y,0) = β - γ·y  (gradiente suave em y)

BCs: Neumann homogêneo (sem fluxo nas bordas)
"""

import numpy as np
import os, sys

diretorio_atual = os.path.dirname(os.path.abspath(__file__))
diretorio_pai = os.path.abspath(os.path.join(diretorio_atual, '..'))
sys.path.append(diretorio_pai)

from PDES import PDES
import PDE

# ---------------------------------------------------------------------------
# Parâmetros
# ---------------------------------------------------------------------------
disc_n = [11, 11]
Du     = 1.0
Dv     = 0.0
eps    = 0.08
gamma  = 0.5
beta   = 0.7
tf     = 20.0
nt     = 500

# ---------------------------------------------------------------------------
# EDPs
# ---------------------------------------------------------------------------
PDE_u = PDE.PDE(
    f'dU/dt = {Du}*d2U/dx2 + {Du}*d2U/dy2 + U - U**3/3 - V',
    'U', ['x', 'y'], ['t'],
    ivar_boundary=[(0, 1), (0, 1)],
    expr_ic='2*Heaviside(0.5 - x)*Heaviside(y - 0.5) - 1'
)

PDE_v = PDE.PDE(
    f'dV/dt = {eps}*(U - {gamma}*V + {beta})',
    'V', ['x', 'y'], ['t'],
    ivar_boundary=[(0, 1), (0, 1)],
    expr_ic=f'{beta} - {gamma}*y'
)

PDES1 = PDES([PDE_u, PDE_v], disc_n)

PDES1.discretize(
    method="central",
    west_bd="Neumann",  west_func_bd="0",
    east_bd="Neumann",  east_func_bd="0",
    north_bd="Neumann", north_func_bd="0",
    south_bd="Neumann", south_func_bd="0",
)

# ---------------------------------------------------------------------------
# BDF2
# ---------------------------------------------------------------------------
PDES1.solve(method='bdf2', tf=tf, nt=nt, tol=1e-6)
resultado_final_bdf2 = PDES1.results


# ---------------------------------------------------------------------------
# Visualização
# ---------------------------------------------------------------------------
PDES1.visualize(mode='plot3d', func_idx=0, cmap="RdYlBu_r")
PDES1.visualize(mode='plot3d', func_idx=1, cmap="PRGn") 
# ---------------------------------------------------------------------------
# Resolve com RKF
# ---------------------------------------------------------------------------
PDES1.solve(method='RKF', tf=tf, nt=nt, tol=1e-5)

# ---------------------------------------------------------------------------
# Visualização
# ---------------------------------------------------------------------------
PDES1.visualize(mode='plot3d', func_idx=0, cmap="RdBu_r")   # u — potencial
PDES1.visualize(mode='plot3d', func_idx=1, cmap="PRGn")     # v — recuperação

"""
fitzhugh_nagumo_2d.py
---------------------
Validação da equação de FitzHugh-Nagumo 2D via Método das Soluções
Manufaturadas (MMS).

Sistema de EDPs:
    ∂u/∂t = Du·∇²u + u - u³/3 - v + Su(x,y,t)
    ∂v/∂t = Dv·∇²v + ε·(u - γ·v + β) + Sv(x,y,t)

Soluções manufaturadas escolhidas:
    u*(x,y,t) = sin(πx)·sin(πy)·cos(t)
    v*(x,y,t) = sin(πx)·sin(πy)·sin(t)

Termos fonte Su e Sv são calculados analiticamente substituindo u*, v*
na EDP e isolando o resíduo.

Parâmetros típicos:
    Du = 1e-3,  Dv = 5e-3
    ε  = 0.1,   γ  = 0.5,  β = 0.7

Domínio: x ∈ [0,1], y ∈ [0,1], t ∈ [0, tf]
BCs: Dirichlet u = v = 0 nas bordas (u* e v* são zero nas bordas ✓)
CI:  u(x,y,0) = sin(πx)·sin(πy),  v(x,y,0) = 0
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
Du     = 1e-3
Dv     = 5e-3
eps    = 0.1
gamma  = 0.5
beta   = 0.7
tf     = 1.0
nt     = 200

x_lin = np.linspace(0, 1, disc_n[0])
y_lin = np.linspace(0, 1, disc_n[1])
X, Y  = np.meshgrid(x_lin, y_lin, indexing='ij')

# ---------------------------------------------------------------------------
# Soluções manufaturadas
# ---------------------------------------------------------------------------
def u_analitica(X, Y, t):
    return np.sin(np.pi * X) * np.sin(np.pi * Y) * np.cos(t)

def v_analitica(X, Y, t):
    return np.sin(np.pi * X) * np.sin(np.pi * Y) * np.sin(t)

# ---------------------------------------------------------------------------
# Termos fonte calculados por MMS
#
# Para u*:
#   ∂u*/∂t = -sin(πx)sin(πy)sin(t)
#   Du·∇²u* = -2π²·Du·sin(πx)sin(πy)cos(t)
#   u* - (u*)³/3 - v* = sin(πx)sin(πy)·[cos(t) - sin³(πx)sin³(πy)cos³(t)/3 - sin(t)]
#
#   Su = ∂u*/∂t - Du·∇²u* - u* + (u*)³/3 + v*
#
# Para v*:
#   ∂v*/∂t = sin(πx)sin(πy)cos(t)
#   Dv·∇²v* = -2π²·Dv·sin(πx)sin(πy)sin(t)
#   ε·(u* - γ·v* + β) = ε·[sin(πx)sin(πy)cos(t) - γ·sin(πx)sin(πy)sin(t) + β]
#
#   Sv = ∂v*/∂t - Dv·∇²v* - ε·(u* - γ·v* + β)
#
# Abaixo as expressões como strings SymPy-compatíveis para o solver:
# ---------------------------------------------------------------------------

# Laplaciano de u*: -2π²·sin(πx)sin(πy)cos(t)
# Laplaciano de v*: -2π²·sin(πx)sin(πy)sin(t)

Su = (
    f"(-sin(pi*x)*sin(pi*y)*sin(t))"                               # ∂u*/∂t
    f" - ({Du})*(-2*pi**2*sin(pi*x)*sin(pi*y)*cos(t))"            # -Du·∇²u*
    f" - (sin(pi*x)*sin(pi*y)*cos(t))"                             # -u*
    f" + (sin(pi*x)*sin(pi*y)*cos(t))**3/3"                        # +(u*)³/3
    f" + (sin(pi*x)*sin(pi*y)*sin(t))"                             # +v*
)

Sv = (
    f"(sin(pi*x)*sin(pi*y)*cos(t))"                                # ∂v*/∂t
    f" - ({Dv})*(-2*pi**2*sin(pi*x)*sin(pi*y)*sin(t))"            # -Dv·∇²v*
    f" - {eps}*(sin(pi*x)*sin(pi*y)*cos(t)"                        # -ε·u*
    f" - {gamma}*sin(pi*x)*sin(pi*y)*sin(t)"                       # +ε·γ·v*
    f" + {beta})"                                                   # -ε·β
)

# ---------------------------------------------------------------------------
# BC: u* e v* são zero nas bordas (sin(0)=sin(π)=0) → Dirichlet homogêneo
# ---------------------------------------------------------------------------
bc_zero = "0"

# ---------------------------------------------------------------------------
# CI
# ---------------------------------------------------------------------------
ic_u = "sin(pi*x)*sin(pi*y)"   # u*(x,y,0) = sin(πx)sin(πy)·cos(0) = sin(πx)sin(πy)
ic_v = "0"                      # v*(x,y,0) = sin(πx)sin(πy)·sin(0) = 0

# ---------------------------------------------------------------------------
# Definição das EDPs com fonte MMS
# ---------------------------------------------------------------------------
PDE_u = PDE.PDE(
    f'dU/dt = {Du}*d2U/dx2 + {Du}*d2U/dy2 + U - U**3/3 - V + ({Su})',
    'U', ['x', 'y'], ['t'],
    ivar_boundary=[(0, 1), (0, 1)],
    expr_ic=ic_u
)

PDE_v = PDE.PDE(
    f'dV/dt = {Dv}*d2V/dx2 + {Dv}*d2V/dy2 + {eps}*(U - {gamma}*V + {beta}) + ({Sv})',
    'V', ['x', 'y'], ['t'],
    ivar_boundary=[(0, 1), (0, 1)],
    expr_ic=ic_v
)

PDES1 = PDES([PDE_u, PDE_v], disc_n)

PDES1.discretize(
    method="central",
    west_bd="Dirichlet",  west_func_bd=bc_zero,
    east_bd="Dirichlet",  east_func_bd=bc_zero,
    north_bd="Dirichlet", north_func_bd=bc_zero,
    south_bd="Dirichlet", south_func_bd=bc_zero,
)

# ---------------------------------------------------------------------------
# Referência analítica em tf
# ---------------------------------------------------------------------------
u_ref = u_analitica(X, Y, tf).flatten().tolist()
v_ref = v_analitica(X, Y, tf).flatten().tolist()

# ---------------------------------------------------------------------------
# RKF
# ---------------------------------------------------------------------------
PDES1.solve(method='RKF', tf=tf, nt=nt, tol=1e-6)

u_res = np.array(PDES1.results[0]).flatten().tolist()
v_res = np.array(PDES1.results[1]).flatten().tolist()
mae_u_rk = sum(abs(r - p) for r, p in zip(u_ref, u_res)) / len(u_ref)
mae_v_rk = sum(abs(r - p) for r, p in zip(v_ref, v_res)) / len(v_ref)

print(f"MAE RKF  — u: {mae_u_rk:.6e}  |  v: {mae_v_rk:.6e}")

# ---------------------------------------------------------------------------
# CN
# ---------------------------------------------------------------------------
PDES1.solve(method='CN', tf=tf, nt=nt, tol=1e-6)
u_res = np.array(PDES1.results[0]).flatten().tolist()
v_res = np.array(PDES1.results[1]).flatten().tolist()
mae_u_cn = sum(abs(r - p) for r, p in zip(u_ref, u_res)) / len(u_ref)
mae_v_cn = sum(abs(r - p) for r, p in zip(v_ref, v_res)) / len(v_ref)
print(f"MAE CN   — u: {mae_u_cn:.6e}  |  v: {mae_v_cn:.6e}")

# ---------------------------------------------------------------------------
# BDF2
# ---------------------------------------------------------------------------
PDES1.solve(method='bdf2', tf=tf, nt=nt, tol=1e-6)
u_res = np.array(PDES1.results[0]).flatten().tolist()
v_res = np.array(PDES1.results[1]).flatten().tolist()
mae_u_bdf2 = sum(abs(r - p) for r, p in zip(u_ref, u_res)) / len(u_ref)
mae_v_bdf2 = sum(abs(r - p) for r, p in zip(v_ref, v_res)) / len(v_ref)
print(f"MAE BDF2 — u: {mae_u_bdf2:.6e}  |  v: {mae_v_bdf2:.6e}")

# ---------------------------------------------------------------------------
# Visualização — u e v separadamente
# ---------------------------------------------------------------------------
PDES1.visualize(mode='plot3d', func_idx=0, cmap="RdYlBu_r")  # u
PDES1.visualize(mode='plot3d', func_idx=1, cmap="PRGn")       # v

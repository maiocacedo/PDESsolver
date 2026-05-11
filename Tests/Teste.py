"""
mms_diferentona_2d.py
----------------------------
Metodo de Solucoes Manufaturadas (MMS) para EDP parabolica com gradiente
movel suavizado.

Solucao manufaturada:
    u(x, y, t) = tanh( y - x^2 - 0.3 - 0.2*sin(2t) )

Camada de transicao tem largura ~1 em theta -> ~1 em y. Em malha 21x21
(h ~ 0.05), atravessada por ~20 pontos. Suficiente para ver convergencia
com diferencas centradas + Neumann one-sided 2a ordem.

EDP governante:
    du/dt = x*y*d2u/dx2 + (1+x^2)*d2u/dy2 + y^2*du/dx + S(x,y,t)

Condicoes:
    Dirichlet em W (x=0), E (x=1), N (y=1)
    Neumann   em S (y=0)  -- du/dn (derivada normal exterior, n = -y_hat)

Derivacao (verificada numericamente, residuo ~1e-15):
    theta = y - x^2 - 0.3 - 0.2*sin(2t)
    theta_t = -0.4*cos(2t),  theta_x = -2x,  theta_y = 1
    u_t  = sech^2(theta) * (-0.4*cos(2t))
    u_xx = -2 sech^2 - 8 x^2 tanh sech^2
    u_yy = -2 tanh sech^2
    S = sech^2 * [-0.4 cos(2t) + 2 x y + 2 x y^2]
      + sech^2 * tanh * [8 x^3 y + 2 (1 + x^2)]

Derivada normal exterior em y=0 (n = -y_hat):
    du/dn|_{y=0} = -du/dy|_{y=0} = -sech^2(theta|_{y=0})
                 = -(1 - tanh^2(-x^2 - 0.3 - 0.2*sin(2t)))
"""

import numpy as np
import os
import sys

diretorio_atual = os.path.dirname(os.path.abspath(__file__))
diretorio_pai = os.path.abspath(os.path.join(diretorio_atual, '..'))
sys.path.append(diretorio_pai)

from PDES import PDES
import PDE

# ---------------------------------------------------------------------------
# Parametros
# ---------------------------------------------------------------------------
disc_n = [21, 21]
tf     = 2.0
nt     = 1000

# ---------------------------------------------------------------------------
# Strings simbolicas
# ---------------------------------------------------------------------------
theta   = "(y - x**2 - 0.3 - 0.2*sin(2*t))"
sech2   = f"(1 - tanh({theta})**2)"
tanh_th = f"tanh({theta})"

term1 = "(-0.4*cos(2*t) + 2*x*y + 2*x*y**2)"
term2 = "(8*x**3*y + 2*(1 + x**2))"

S = f"({sech2} * {term1} + {sech2} * {tanh_th} * {term2})"
eq_str = f"x*y*d2U/dx2 + (1 + x**2)*d2U/dy2 + y**2*dU/dx + {S}"

# Condicao inicial
ic_str = "tanh(y - x**2 - 0.3)"

# Contornos Dirichlet
west_str  = "tanh(y - 0.3 - 0.2*sin(2*t))"
east_str  = "tanh(y - 1.3 - 0.2*sin(2*t))"
north_str = "tanh(1 - x**2 - 0.3 - 0.2*sin(2*t))"

# Contorno Neumann sul (du/dn = -du/dy, normal exterior em y=0)
south_str = "-(1 - tanh(-x**2 - 0.3 - 0.2*sin(2*t))**2)"

# ---------------------------------------------------------------------------
# Instanciacao
# ---------------------------------------------------------------------------
PDE_u = PDE.PDE(
    f'dU/dt = {eq_str}',
    'U', ['x', 'y'], ['t'],
    ivar_boundary=[(0, 1), (0, 1)],
    expr_ic=ic_str,
    west_bd="Dirichlet",  west_func_bd=west_str,
    east_bd="Dirichlet",  east_func_bd=east_str,
    north_bd="Dirichlet", north_func_bd=north_str,
    south_bd="Neumann",   south_func_bd=south_str,
)

PDES1 = PDES([PDE_u], disc_n)
PDES1.discretize(method="central")
PDES1.solve(method='bdf2', tf=tf, nt=nt, tol=1e-6)

# ---------------------------------------------------------------------------
# Comparacao com solucao analitica
# ---------------------------------------------------------------------------
N = disc_n[0]
x_vals = np.linspace(0, 1, N)
y_vals = np.linspace(0, 1, N)
X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
U_exato_2d = np.tanh(Y - X**2 - 0.3 - 0.2 * np.sin(2 * tf))

resultado_2d = np.asarray(PDES1.results[0]).reshape(N, N)
erro_2d = np.abs(resultado_2d - U_exato_2d)

print(f"\n--- Analise de Erro MMS (t = {tf}, malha {disc_n}) ---")
print(f"Erro Maximo (L_inf): {erro_2d.max():.6e}")
print(f"Erro RMS    (L2)   : {np.linalg.norm(erro_2d)/np.sqrt(erro_2d.size):.6e}")
print(f"\n--- Erro nas 4 fronteiras ---")
print(f"  W (x=0, Dirichlet)  : {erro_2d[0, :].max():.3e}")
print(f"  E (x=1, Dirichlet)  : {erro_2d[-1, :].max():.3e}")
print(f"  N (y=1, Dirichlet)  : {erro_2d[:, -1].max():.3e}")
print(f"  S (y=0, Neumann)    : {erro_2d[:, 0].max():.3e}")
print(f"  Interior            : {erro_2d[1:-1, 1:-1].max():.3e}")

# ---------------------------------------------------------------------------
# Visualizacao
# ---------------------------------------------------------------------------
PDES1.visualize(mode='plot3d',      func_idx=0, cmap="viridis")
PDES1.visualize(mode='animation3d', func_idx=0, cmap="viridis")
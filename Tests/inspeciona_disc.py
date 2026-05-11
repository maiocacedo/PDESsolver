"""
Inspeciona as strings que Disc.py gera para um caso pequeno.
"""

import os, sys
diretorio_atual = os.path.dirname(os.path.abspath(__file__))
diretorio_pai = os.path.abspath(os.path.join(diretorio_atual, '..'))
sys.path.append(diretorio_pai)

from PDES import PDES
import PDE

theta = "(y - x**2 - 0.3)"
sech2 = f"(1 - tanh({theta})**2)"
tanhT = f"tanh({theta})"
S = f"(2*{sech2} + 8*x**2*{tanhT}*{sech2} + 2*{tanhT}*{sech2})"
eq = f"d2U/dx2 + d2U/dy2 + {S}"

pde = PDE.PDE(
    f'dU/dt = {eq}',
    'U', ['x', 'y'], ['t'],
    ivar_boundary=[(0, 1), (0, 1)],
    expr_ic="tanh(y - x**2 - 0.3)",
    west_bd="Dirichlet",  west_func_bd="tanh(y - 0.3)",
    east_bd="Dirichlet",  east_func_bd="tanh(y - 1.3)",
    north_bd="Dirichlet", north_func_bd="tanh(1 - x**2 - 0.3)",
    south_bd="Dirichlet", south_func_bd="tanh(-x**2 - 0.3)",
)

P = PDES([pde], [5, 5])
P.discretize(method="central")

flat_list, d_vars = P.disc_results

print(f"\nMalha 5x5 -> 25 nos.")
print(f"Total de strings em flat_list: {len(flat_list)}")
print(f"d_vars (primeiros 10): {d_vars[:10]}")
print(f"\n=== STRINGS DOS NOS INTERIORES (i,j em [1,3]) ===\n")

dirichlet_idx = set(P.dirichlet_constraints.keys())
N = 5
for idx, eq_str in enumerate(flat_list):
    i = idx // N
    j = idx % N
    is_bd = idx in dirichlet_idx
    if not is_bd:
        print(f"--- No ({i},{j}) idx={idx} ---")
        print(f"  {eq_str}")
        print()
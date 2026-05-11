import os
import sys
import numpy as np

diretorio_atual = os.path.dirname(os.path.abspath(__file__))
diretorio_pai = os.path.abspath(os.path.join(diretorio_atual, '..'))
sys.path.append(diretorio_pai)

from PDES import PDES
import PDE

theta_str = "(y - x**2 - 0.3 - 0.2*sin(2*t))"
sech2_str = f"(1 - tanh({theta_str})**2)"
tanh_str  = f"tanh({theta_str})"
S_str = (f"({sech2_str}*(-0.4*cos(2*t) + 2*x*y + 2*x*y**2) "
         f"+ {sech2_str}*{tanh_str}*(8*x**3*y + 2*(1 + x**2)))")
eq_str = f"x*y*d2U/dx2 + (1 + x**2)*d2U/dy2 + y**2*dU/dx + {S_str}"

ic_str    = "tanh(y - x**2 - 0.3)"
west_str  = "tanh(y - 0.3 - 0.2*sin(2*t))"
east_str  = "tanh(y - 1.3 - 0.2*sin(2*t))"
north_str = "tanh(1 - x**2 - 0.3 - 0.2*sin(2*t))"
south_str_dir = "tanh(-x**2 - 0.3 - 0.2*sin(2*t))"
south_str_neu = "-(1 - tanh(-x**2 - 0.3 - 0.2*sin(2*t))**2)"


def rodar(N, nt, tf, sul_tipo):
    if sul_tipo == 'dirichlet':
        sb, sf = "Dirichlet", south_str_dir
    elif sul_tipo == 'neumann':
        sb, sf = "Neumann", south_str_neu
    else:
        raise ValueError(sul_tipo)

    pde = PDE.PDE(
        f'dU/dt = {eq_str}',
        'U', ['x', 'y'], ['t'],
        ivar_boundary=[(0, 1), (0, 1)],
        expr_ic=ic_str,
        west_bd="Dirichlet",  west_func_bd=west_str,
        east_bd="Dirichlet",  east_func_bd=east_str,
        north_bd="Dirichlet", north_func_bd=north_str,
        south_bd=sb,          south_func_bd=sf,
    )
    P = PDES([pde], [N, N])
    P.discretize(method="central")
    P.solve(method='bdf2', tf=tf, nt=nt, tol=1e-6)

    x_v = np.linspace(0, 1, N)
    y_v = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x_v, y_v, indexing='ij')
    u_ex = np.tanh(Y - X**2 - 0.3 - 0.2 * np.sin(2 * tf)).flatten()

    u_num = np.asarray(P.results[0]).ravel()
    erro  = np.abs(u_num - u_ex)
    return erro.max(), np.linalg.norm(erro) / np.sqrt(erro.size)


def estudo(sul_tipo, configs, tf=2.0):
    print(f"\n{'='*60}")
    print(f"Convergencia com sul = {sul_tipo.upper()}, tf = {tf}")
    print(f"{'='*60}")
    print(f"{'N':>4} {'nt':>5} | {'L_inf':>11} | {'L2':>11} | razao | ordem")
    print("-" * 60)

    err_ant = None
    for N, nt in configs:
        e_inf, e_l2 = rodar(N, nt, tf, sul_tipo)

        if err_ant is not None:
            razao = err_ant / e_inf
            ordem = np.log2(razao) if razao > 0 else float('nan')
            razao_str = f"{razao:5.2f}x"
            ordem_str = f"{ordem:5.2f}"
        else:
            razao_str = "    -"
            ordem_str = "    -"

        print(f"{N:>4} {nt:>5} | {e_inf:>11.3e} | {e_l2:>11.3e} | {razao_str} | {ordem_str}")
        err_ant = e_inf


if __name__ == "__main__":
    configs = [(11, 4000), (21, 4000)]

    estudo('dirichlet', configs, tf=2.0)
    estudo('neumann',   configs, tf=2.0)
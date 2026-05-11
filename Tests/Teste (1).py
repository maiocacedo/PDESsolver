import os, sys, numpy as np
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

PDE_obj = PDE.PDE(
    f'dU/dt = {eq_str}',
    'U', ['x', 'y'], ['t'],
    ivar_boundary=[(0, 1), (0, 1)],
    expr_ic="tanh(y - x**2 - 0.3)",
    west_bd="Dirichlet",  west_func_bd="tanh(y - 0.3 - 0.2*sin(2*t))",
    east_bd="Dirichlet",  east_func_bd="tanh(y - 1.3 - 0.2*sin(2*t))",
    north_bd="Dirichlet", north_func_bd="tanh(1 - x**2 - 0.3 - 0.2*sin(2*t))",
    south_bd="Dirichlet", south_func_bd="tanh(-x**2 - 0.3 - 0.2*sin(2*t))",
)
P = PDES([PDE_obj], [21, 21])
P.discretize(method="central")
P.solve(method='bdf2', tf=2.0, nt=400, tol=1e-6)

N = 21
x_v = np.linspace(0,1,N); y_v = np.linspace(0,1,N)
X, Y = np.meshgrid(x_v, y_v, indexing='ij')
u_ex_tf = np.tanh(Y - X**2 - 0.3 - 0.2*np.sin(2*2.0)).flatten()

u_ret    = np.asarray(P.results[0]).ravel()
u_hist   = np.asarray(P.results[1][0][-1]).ravel()

print(f"\n=== TESTES BASICOS ===")
print(f"u_ret e u_hist sao iguais?  {np.allclose(u_ret, u_hist)}")
print(f"|u_ret - u_hist|_inf       = {np.abs(u_ret - u_hist).max():.3e}")
print()
print(f"|u_ret - u_ex(t=2.0)|_inf  = {np.abs(u_ret - u_ex_tf).max():.6e}")
print(f"|u_hist- u_ex(t=2.0)|_inf  = {np.abs(u_hist - u_ex_tf).max():.6e}")

print(f"\n=== SEGUNDA RODADA (mesmo problema, recriado do zero) ===")
P2 = PDES([PDE_obj], [21, 21])
P2.discretize(method="central")
P2.solve(method='bdf2', tf=2.0, nt=400, tol=1e-6)
u_ret2 = np.asarray(P2.results[0]).ravel()
print(f"|u_ret - u_ret2|_inf       = {np.abs(u_ret - u_ret2).max():.3e}")
print(f"|u_ret2- u_ex(t=2.0)|_inf  = {np.abs(u_ret2 - u_ex_tf).max():.6e}")

err1 = np.abs(u_ret  - u_ex_tf).max()
err2 = np.abs(u_hist - u_ex_tf).max()
err3 = np.abs(u_ret2 - u_ex_tf).max()
print(f"\nResumo: {err1:.6e}, {err2:.6e}, {err3:.6e}")
print("Se os tres forem iguais, o solver e deterministico e o bug 0.19 vs 0.08")
print("foi de outra coisa (talvez tf diferente entre rodadas, BC diferente).")
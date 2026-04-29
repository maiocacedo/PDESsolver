"""
adveccao_naolinear_1d.py
------------------------
Validação de EDP não-linear 1D com condição de Neumann.

EDP (advecção não-linear / Burgers inviscida com fonte):
    ∂u/∂t + u·∂u/∂x = f(x, t)

Solução manufaturada:
    u(x, t) = exp(-t) · sin(πx)

Derivando f:
    ∂u/∂t = -exp(-t)·sin(πx)
    u·∂u/∂x = exp(-t)·sin(πx) · π·exp(-t)·cos(πx)
             = π·exp(-2t)·sin(πx)·cos(πx)
             = (π/2)·exp(-2t)·sin(2πx)

    f = ∂u/∂t + u·∂u/∂x
      = -exp(-t)·sin(πx) + (π/2)·exp(-2t)·sin(2πx)

Condições de contorno (Neumann):
    ∂u/∂x|_{x=0} = π·exp(-t)·cos(0) = π·exp(-t)
    ∂u/∂x|_{x=1} = π·exp(-t)·cos(π) = -π·exp(-t)

CI: u(x, 0) = sin(πx)

Nota sobre CN e BDF2:
    O detector de linearidade identificará esta EDP como NÃO-LINEAR
    (devido ao produto U*dU/dx) e usará Newton automaticamente.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from PDES import PDES
import PDE

# ---------------------------------------------------------------------------
# Parâmetros
# ---------------------------------------------------------------------------
disc_n = [10]
tf     = 0.5
nt     = 100

x_lin = np.linspace(0, 1, disc_n[0])

# ---------------------------------------------------------------------------
# Solução analítica
# ---------------------------------------------------------------------------
def analitica(x, t):
    return np.exp(-t) * np.sin(np.pi * x)

# ---------------------------------------------------------------------------
# Expressões string
# ---------------------------------------------------------------------------
# Fonte: f = -exp(-t)*sin(πx) + (π/2)*exp(-2t)*sin(2πx)
src_expr = (
    "-exp(-t)*sin(pi*x)"
    " + (pi/2)*exp(-2*t)*sin(2*pi*x)"
)

# BCs de Neumann: ∂u/∂x no contorno
west_neumann = "pi*exp(-t)"    # du/dx em x=0
east_neumann = "-pi*exp(-t)"   # du/dx em x=1

# ---------------------------------------------------------------------------
# EDP: dU/dt = -U*dU/dx + f(x,t)
# ---------------------------------------------------------------------------
PDE_nl = PDE.PDE(
    f'dU/dt = -U*dU/dx + {src_expr}',
    'U', ['x'], ['t'],
    ivar_boundary=[(0, 1)],
    expr_ic='sin(pi * x)'
)

PDES1 = PDES([PDE_nl], disc_n)

PDES1.discretize(
    method="central",
    west_bd="Neumann", west_func_bd=west_neumann,
    east_bd="Neumann", east_func_bd=east_neumann,
)

# Confirma que o discretizador gerou termos não-lineares
print("Primeiras equações do flat_list:")
for eq in PDES1.disc_results[0][:3]:
    print(f"  {eq}")
print()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
tempos_fixos = np.linspace(0, tf, nt + 1)

def erro_l2(hist, t_idx, tempos):
    u_num = np.array(hist[0][t_idx])
    u_ana = analitica(x_lin, tempos[t_idx])
    return np.sqrt(np.mean((u_num - u_ana) ** 2))


def plot_comparacao(hist, tempos, titulo):
    u_num = np.array(hist[0][-1])
    u_ana = analitica(x_lin, tempos[-1])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(x_lin, u_ana, 'k-',  lw=2,   label='Analítico')
    axes[0].plot(x_lin, u_num, 'ro--', lw=1.5, label=f'Numérico ({titulo})')
    axes[0].set_title(f"Solução em t={tf:.2f} — {titulo}")
    axes[0].set_xlabel('x'); axes[0].set_ylabel('u')
    axes[0].legend(); axes[0].grid(True, alpha=0.4)

    axes[1].plot(x_lin, np.abs(u_num - u_ana), 'b-', lw=1.5)
    l2 = np.sqrt(np.mean((u_num - u_ana)**2))
    axes[1].set_title(f"|Erro| pontual — L2={l2:.2e}")
    axes[1].set_xlabel('x'); axes[1].set_ylabel('|erro|')
    axes[1].grid(True, alpha=0.4)

    plt.suptitle(f"Advecção não-linear 1D — {titulo}", fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_erro_temporal(resultados, tf):
    plt.figure(figsize=(8, 4))
    for label, hist in resultados.items():
        n_frames = len(hist[0])
        t_frames = np.linspace(0, tf, n_frames)
        erros    = [erro_l2(hist, k, t_frames) for k in range(n_frames)]
        plt.semilogy(t_frames, erros, label=label)
    plt.xlabel('t')
    plt.ylabel('Erro L2')
    plt.title('Erro L2 ao longo do tempo — Advecção não-linear 1D')
    plt.legend()
    plt.grid(True, which='both', alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_evolucao(hist, tempos, titulo, n_snapshots=5):
    """Plota a evolução da solução em vários instantes de tempo."""
    indices = np.linspace(0, len(hist[0])-1, n_snapshots, dtype=int)
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.cm.viridis(np.linspace(0, 1, n_snapshots))

    for k, idx in enumerate(indices):
        t_k  = tempos[idx] if idx < len(tempos) else tempos[-1]
        u_num = np.array(hist[0][idx])
        u_ana = analitica(x_lin, t_k)
        ax.plot(x_lin, u_num, '-',  color=cmap[k], label=f't={t_k:.2f} (num)')
        ax.plot(x_lin, u_ana, '--', color=cmap[k], alpha=0.5)

    ax.set_xlabel('x'); ax.set_ylabel('u')
    ax.set_title(f"Evolução temporal — {titulo}\n(sólido=numérico, tracejado=analítico)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# CN (Newton automático para não-linear)
# ---------------------------------------------------------------------------
print("Resolvendo com Crank-Nicolson (Newton)...")
t0 = time.time()
PDES1.solve(method='CN', tf=tf, nt=nt)
_, hist_cn = PDES1.results
print(f"  Concluído em {time.time()-t0:.2f}s")
print(f"  Erro L2 final (t={tf}): {erro_l2(hist_cn, -1, tempos_fixos):.4e}")
plot_comparacao(hist_cn, tempos_fixos, "CN-Newton")
plot_evolucao(hist_cn, tempos_fixos, "CN-Newton")

# ---------------------------------------------------------------------------
# CN com Picard (para comparar convergência)
# ---------------------------------------------------------------------------
print("\nResolvendo com Crank-Nicolson (Picard)...")
t0 = time.time()
PDES1.solve(method='CN', tf=tf, nt=nt, nonlinear_method='picard')
_, hist_cn_p = PDES1.results
print(f"  Concluído em {time.time()-t0:.2f}s")
print(f"  Erro L2 final (t={tf}): {erro_l2(hist_cn_p, -1, tempos_fixos):.4e}")
plot_comparacao(hist_cn_p, tempos_fixos, "CN-Picard")

# ---------------------------------------------------------------------------
# BDF2 (Newton automático)
# ---------------------------------------------------------------------------
print("\nResolvendo com BDF2 (Newton)...")
t0 = time.time()
PDES1.solve(method='bdf2', tf=tf, nt=nt)
_, hist_bdf2 = PDES1.results
print(f"  Concluído em {time.time()-t0:.2f}s")
print(f"  Erro L2 final (t={tf}): {erro_l2(hist_bdf2, -1, tempos_fixos):.4e}")
plot_comparacao(hist_bdf2, tempos_fixos, "BDF2-Newton")

# ---------------------------------------------------------------------------
# RKF
# ---------------------------------------------------------------------------
print("\nResolvendo com RKF...")
t0 = time.time()
PDES1.solve(method='RKF', tf=tf, nt=nt, tol=1e-6)
_, hist_rkf = PDES1.results
tempos_rkf  = np.linspace(0, tf, len(hist_rkf[0]))
print(f"  Concluído em {time.time()-t0:.2f}s")
print(f"  Erro L2 final (t={tf}): {erro_l2(hist_rkf, -1, tempos_rkf):.4e}")
plot_comparacao(hist_rkf, tempos_rkf, "RKF")

# ---------------------------------------------------------------------------
# Erro L2 temporal — todos os métodos
# ---------------------------------------------------------------------------
plot_erro_temporal(
    {"CN-Newton": hist_cn, "CN-Picard": hist_cn_p,
     "BDF2-Newton": hist_bdf2, "RKF": hist_rkf},
    tf
)
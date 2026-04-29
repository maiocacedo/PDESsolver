"""
bdf2.py
-------
Solver BDF2 com suporte a EDPs lineares e não-lineares.

Para EDPs lineares: L é fixado uma vez (rápido).
Para EDPs não-lineares: itera via Newton (padrão) ou Picard a cada passo.

Fórmula BDF2:
    (I - 2dt/3 * L) * u^{n+1} = (4u^n - u^{n-1})/3 + (2dt/3)*f^{n+1}

Arranque BDF1:
    (I - dt * L) * u^1 = u^0 + dt*f^1
"""

import time
import numpy as np
import scipy.sparse as sp_sparse
from scipy.sparse.linalg import spsolve

from .solver_base import (
    compile_equations, extract_linear_structure, detect_linearity,
    eval_F, picard_step, newton_step,
    make_history, save_to_history,
)


def bdf2(flat_list, d_vars, tf, nt, ic, n_funcs=None,
         nonlinear_method='newton', tol_nl=1e-8, max_iter_nl=20,
         verbose_nl=False):
    """
    Solver BDF2.

    Parâmetros
    ----------
    flat_list : list[str]
        Equações discretizadas espacialmente.
    d_vars : list[str]
        Nomes das variáveis discretizadas.
    tf : float
        Tempo final.
    nt : int
        Número de passos de tempo.
    ic : array-like
        Condição inicial.
    n_funcs : int, opcional
        Número de funções dependentes.
    nonlinear_method : str
        Método para EDPs não-lineares: 'newton' (padrão) ou 'picard'.
    tol_nl : float
        Tolerância de convergência do método não-linear.
    max_iter_nl : int
        Máximo de iterações por passo.
    verbose_nl : bool
        Se True, imprime resíduo a cada iteração.

    Retorna
    -------
    u : np.ndarray
    final_list : list[list]
    """
    dt = tf / nt
    n  = len(d_vars)
    u  = np.array(ic, dtype=np.float64).flatten()

    final_list, use_groups, n_elements = make_history(n_funcs, n)

    # --- Compilação ---
    funcs = compile_equations(flat_list, d_vars)

    # --- Detecção de linearidade ---
    is_linear, L = detect_linearity(funcs, n)

    I = sp_sparse.eye(n, format='csr')

    if is_linear:
        _, fonte_func = extract_linear_structure(funcs, n, verbose=False)
        A_bdf1 = I - dt * L
        A_bdf2 = I - (2.0 * dt / 3.0) * L
    else:
        print(f"  [BDF2] EDP não-linear detectada — usando {nonlinear_method.upper()} "
              f"(tol={tol_nl:.0e}, max_iter={max_iter_nl})")
        _, fonte_func = extract_linear_structure(funcs, n, verbose=False)

    save_to_history(u, final_list, use_groups, n_funcs, n_elements)

    # -----------------------------------------------------------------------
    # Passo 0→1: BDF1 (arranque)
    # -----------------------------------------------------------------------
    t0_loop = time.time()
    total_iters = 0
    tempo_1 = dt

    if is_linear:
        rhs_1  = u + dt * fonte_func(tempo_1)
        u_prev = u.copy()
        u      = spsolve(A_bdf1, rhs_1)
    else:
        F_0      = eval_F(funcs, 0.0, u)
        rhs_hist = u + dt * eval_F(funcs, 0.0, np.zeros(n))  # fonte em u=0

        if nonlinear_method == 'newton':
            u_new, n_iter = newton_step(
                funcs, u, tempo_1, dt, n, u,
                alpha=1.0, max_iter=max_iter_nl,
                tol_nl=tol_nl, verbose=verbose_nl
            )
        else:
            u_new, n_iter = picard_step(
                funcs, u, tempo_1, dt, n, u,
                alpha=1.0, max_iter=max_iter_nl,
                tol_nl=tol_nl, verbose=verbose_nl
            )
        total_iters += n_iter
        u_prev = u.copy()
        u      = u_new

    save_to_history(u, final_list, use_groups, n_funcs, n_elements)

    # -----------------------------------------------------------------------
    # Passos 1→nt: BDF2
    # -----------------------------------------------------------------------
    for passo in range(1, nt):
        tempo_n1 = (passo + 1) * dt
        rhs_hist = (4.0 * u - u_prev) / 3.0

        if is_linear:
            rhs_vec = rhs_hist + (2.0 * dt / 3.0) * fonte_func(tempo_n1)
            u_prev  = u.copy()
            u       = spsolve(A_bdf2, rhs_vec)

        else:
            if nonlinear_method == 'newton':
                u_new, n_iter = newton_step(
                    funcs, u, tempo_n1, dt, n, rhs_hist,
                    alpha=2.0/3.0, max_iter=max_iter_nl,
                    tol_nl=tol_nl, verbose=verbose_nl
                )
            else:
                u_new, n_iter = picard_step(
                    funcs, u, tempo_n1, dt, n, rhs_hist,
                    alpha=2.0/3.0, max_iter=max_iter_nl,
                    tol_nl=tol_nl, verbose=verbose_nl
                )
            total_iters += n_iter
            u_prev = u.copy()
            u      = u_new

        save_to_history(u, final_list, use_groups, n_funcs, n_elements)

    elapsed = time.time() - t0_loop
    print(f"  [BDF2] Loop de tempo: {elapsed:.3f}s", end="")
    if not is_linear:
        print(f" | Média iterações/passo: {total_iters/nt:.1f}", end="")
    print()

    return u, final_list
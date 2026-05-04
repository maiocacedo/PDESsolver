"""
bdf2.py
-------
Solver BDF2 com suporte a EDPs lineares e não-lineares.

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
         verbose_nl=False, dirichlet_constraints=None):
    """
    Solver BDF2.

    Parâmetros
    ----------
    flat_list : list[str]
    d_vars : list[str]
    tf : float
    nt : int
    ic : array-like
    n_funcs : int, opcional
    nonlinear_method : str  ('newton' ou 'picard')
    tol_nl : float
    max_iter_nl : int
    verbose_nl : bool
    dirichlet_constraints : dict[int, dict] | None
        {idx: {'expr': str, 'x': float, 'y': float}}

    Retorna
    -------
    u : np.ndarray
    final_list : list[list]
    """
    import math

    dt = tf / nt
    n  = len(d_vars)
    u  = np.array(ic, dtype=np.float64).flatten()

    dirichlet_constraints = dirichlet_constraints or {}

    def _apply_dirichlet(u, t_val):
        for idx, info in dirichlet_constraints.items():
            try:
                u[idx] = float(eval(
                    info['expr'],
                    {'t': t_val, 'x': info['x'], 'y': info['y'],
                     'exp': math.exp, 'sin': math.sin, 'cos': math.cos,
                     'pi': math.pi, '__builtins__': {}}
                ))
            except Exception:
                try:
                    u[idx] = float(info['expr'])
                except Exception:
                    pass
        return u

    u = _apply_dirichlet(u, 0.0)

    final_list, use_groups, n_elements = make_history(n_funcs, n)

    # --- Compilação ---
    funcs = compile_equations(flat_list, d_vars)

    # --- Detecção de linearidade (excluindo nós Dirichlet) ---
    is_linear, L = detect_linearity(funcs, n,
                                    dirichlet_indices=list(dirichlet_constraints.keys()))

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
        u      = _apply_dirichlet(u, tempo_1)
    else:
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
        u      = _apply_dirichlet(u_new, tempo_1)

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
            u       = _apply_dirichlet(u, tempo_n1)

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
            u      = _apply_dirichlet(u_new, tempo_n1)

        save_to_history(u, final_list, use_groups, n_funcs, n_elements)

    elapsed = time.time() - t0_loop
    print(f"  [BDF2] Loop de tempo: {elapsed:.3f}s", end="")
    if not is_linear:
        print(f" | Média iterações/passo: {total_iters/nt:.1f}", end="")
    print()

    return u, final_list

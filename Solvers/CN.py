"""
CN.py
-----
Solver Crank-Nicolson com suporte a EDPs lineares e não-lineares.

Para EDPs lineares: L é fixado uma vez (rápido).
Para EDPs não-lineares: itera via Newton (padrão) ou Picard a cada passo.

Fórmula CN:
    (I - dt/2 * L) * u^{n+1} = (I + dt/2 * L) * u^n + dt/2 * (f^n + f^{n+1})
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


def cn(flat_list, d_vars, tf, nt, ic, n_funcs=None,
       nonlinear_method='newton', tol_nl=1e-8, max_iter_nl=20,
       verbose_nl=False, dirichlet_constraints=None):
    """
    Solver Crank-Nicolson.

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
        A_impl = I - (dt / 2.0) * L
        A_expl = I + (dt / 2.0) * L
    else:
        print(f"  [CN] EDP não-linear detectada — usando {nonlinear_method.upper()} "
              f"(tol={tol_nl:.0e}, max_iter={max_iter_nl})")
        _, fonte_func = extract_linear_structure(funcs, n, verbose=False)

    save_to_history(u, final_list, use_groups, n_funcs, n_elements)

    # --- Loop de tempo ---
    t0 = time.time()
    total_iters = 0

    for passo in range(nt):
        tempo_atual   = passo * dt
        tempo_proximo = (passo + 1) * dt

        if is_linear:
            f_n  = fonte_func(tempo_atual)
            f_n1 = fonte_func(tempo_proximo)
            rhs  = A_expl.dot(u) + (dt / 2.0) * (f_n + f_n1)
            u    = spsolve(A_impl, rhs)
            u    = _apply_dirichlet(u, tempo_proximo)

        else:
            F_n      = eval_F(funcs, tempo_atual, u)
            rhs_hist = u + (dt / 2.0) * F_n

            if nonlinear_method == 'newton':
                u, n_iter = newton_step(
                    funcs, u, tempo_proximo, dt, n, rhs_hist,
                    alpha=0.5, max_iter=max_iter_nl,
                    tol_nl=tol_nl, verbose=verbose_nl
                )
            else:
                u, n_iter = picard_step(
                    funcs, u, tempo_proximo, dt, n, rhs_hist,
                    alpha=0.5, max_iter=max_iter_nl,
                    tol_nl=tol_nl, verbose=verbose_nl
                )
            u = _apply_dirichlet(u, tempo_proximo)
            total_iters += n_iter

        save_to_history(u, final_list, use_groups, n_funcs, n_elements)

    elapsed = time.time() - t0
    print(f"  [CN] Loop de tempo: {elapsed:.3f}s", end="")
    if not is_linear:
        print(f" | Média iterações/passo: {total_iters/nt:.1f}", end="")
    print()

    return u, final_list

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
       verbose_nl=False):
    """
    Solver Crank-Nicolson.

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
        # Pré-computa matrizes fixas
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

        else:
            # rhs_hist = u^n + dt/2 * (L*u^n + f^n)  avaliado no estado atual
            F_n      = eval_F(funcs, tempo_atual, u)
            rhs_hist = u + (dt / 2.0) * F_n

            if nonlinear_method == 'newton':
                u, n_iter = newton_step(
                    funcs, u, tempo_proximo, dt, n, rhs_hist,
                    alpha=0.5, max_iter=max_iter_nl,
                    tol_nl=tol_nl, verbose=verbose_nl
                )
            else:  # picard
                u, n_iter = picard_step(
                    funcs, u, tempo_proximo, dt, n, rhs_hist,
                    alpha=0.5, max_iter=max_iter_nl,
                    tol_nl=tol_nl, verbose=verbose_nl
                )
            total_iters += n_iter

        save_to_history(u, final_list, use_groups, n_funcs, n_elements)

    elapsed = time.time() - t0
    print(f"  [CN] Loop de tempo: {elapsed:.3f}s", end="")
    if not is_linear:
        print(f" | Média iterações/passo: {total_iters/nt:.1f}", end="")
    print()

    return u, final_list
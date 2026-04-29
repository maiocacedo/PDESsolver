"""
solver_base.py
--------------
Utilitários compartilhados entre CN e BDF2.

Responsabilidades:
- Compilação das equações (lambdify)
- Detecção automática de linearidade
- Extração numérica da matriz L por perturbação
- Iteração de Picard e Newton para EDPs não-lineares
- Gerenciamento do histórico de solução
"""

import time
import warnings
import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
import scipy.sparse as sp_sparse
from scipy.sparse.linalg import spsolve


# ---------------------------------------------------------------------------
# Compilação
# ---------------------------------------------------------------------------

def compile_equations(flat_list, d_vars, verbose=True):
    """
    Compila cada equação como uma função lambdify f(t, *XX).

    Retorna
    -------
    funcs : list[callable]
        Uma função por equação. Assinatura: f(t, *u_vec) -> float
    """
    t0 = time.time()
    t_sym    = sp.Symbol('t')
    sym_list = [sp.Symbol(v) for v in d_vars]
    all_args = (t_sym, *sym_list)

    funcs = [
        sp.lambdify(all_args, parse_expr(eq_str), modules='numpy')
        for eq_str in flat_list
    ]

    if verbose:
        print(f"  Compilação lambdify: {time.time()-t0:.3f}s")
    return funcs


# ---------------------------------------------------------------------------
# Detecção de linearidade
# ---------------------------------------------------------------------------

def detect_linearity(funcs, n, t0_val=0.0, verbose=True):
    """
    Verifica se as equações são lineares nas variáveis discretizadas XX.

    Estratégia: extrai L em u=0 e em u=1. Se forem iguais (dentro de
    tolerância numérica), a EDP é linear e L pode ser fixado para sempre.

    Retorna
    -------
    is_linear : bool
    L : scipy.sparse.csr_matrix
        Matriz extraída em u=0 (válida para todo t se linear).
    """
    t0 = time.time()
    zeros = np.zeros(n)
    ones  = np.ones(n)

    L0, _ = _extract_L(funcs, n, zeros, t0_val)
    L1, _ = _extract_L(funcs, n, ones,  t0_val)

    diff = (L1 - L0)
    is_linear = np.allclose(diff.data, 0, atol=1e-10) if diff.nnz > 0 else True

    if verbose:
        status = "LINEAR" if is_linear else "NÃO-LINEAR"
        print(f"  Detecção de linearidade: {status} ({time.time()-t0:.3f}s)")

    return is_linear, L0


def _extract_L(funcs, n, u_ref, t_val):
    fonte = []
    for i, f in enumerate(funcs):
        try:
            val = f(t_val, *np.zeros(n))
            fonte.append(val)
        except Exception as e:
            print(f"Equação {i} falhou: {e}")
            raise
    fonte = np.array(fonte, dtype=np.float64)

    # ← faltava montar e retornar L aqui
    eps = 1e-6
    rows, cols, vals = [], [], []
    for j in range(n):
        u_pert = u_ref.copy()
        u_pert[j] += eps
        F_pert = np.array([f(t_val, *u_pert) for f in funcs], dtype=np.float64)
        F_ref  = np.array([f(t_val, *u_ref)  for f in funcs], dtype=np.float64)
        dF = (F_pert - F_ref) / eps
        nz = np.nonzero(np.abs(dF) > 1e-15)[0]
        rows.extend(nz.tolist())
        cols.extend([j] * len(nz))
        vals.extend(dF[nz].tolist())

    L = sp_sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return L, fonte

def extract_linear_structure(funcs, n, t0_val=0.0, verbose=True):
    """
    Extrai L (linearização em u=0) e retorna a função de fonte.
    Usado quando is_linear=True ou como ponto de partida para iteração.
    """
    t0 = time.time()
    L, _ = _extract_L(funcs, n, np.zeros(n), t0_val)

    def fonte_func(t_val):
        z = np.zeros(n)
        return np.array([f(float(t_val), *z) for f in funcs], dtype=np.float64)

    if verbose:
        print(f"  Extração estrutura A: {time.time()-t0:.3f}s")

    return L, fonte_func


# ---------------------------------------------------------------------------
# Avaliação vetorial de F
# ---------------------------------------------------------------------------

def eval_F(funcs, t_val, u):
    """Avalia F(t, u) — o lado direito das EDOs no ponto u."""
    return np.array([f(float(t_val), *u) for f in funcs], dtype=np.float64)


# ---------------------------------------------------------------------------
# Iteração de Picard
# ---------------------------------------------------------------------------

def picard_step(funcs, u, t_new, dt, n, rhs_hist,
                alpha, max_iter=50, tol_nl=1e-8, verbose=False):
    """
    Resolve um passo implícito via iteração de Picard.

    A cada iteração k:
        L^k = L(u^k)  (remontado no estado atual)
        (I - alpha*dt*L^k) * u^{k+1} = rhs_hist + alpha*dt*fonte^{k+1}

    Parâmetros
    ----------
    alpha : float
        0.5 para CN, 2/3 para BDF2.
    rhs_hist : np.ndarray
        Lado direito histórico já montado (sem o termo implícito).

    Retorna
    -------
    u_new : np.ndarray
    n_iter : int
    """
    I = sp_sparse.eye(n, format='csr')
    u_k = u.copy()

    for k in range(max_iter):
        L_k, _ = _extract_L(funcs, n, u_k, t_new)
        fonte_k = eval_F(funcs, t_new, np.zeros(n))  # fonte independente de u

        A  = I - alpha * dt * L_k
        b  = rhs_hist + alpha * dt * fonte_k
        u_new = spsolve(A, b)

        res = np.linalg.norm(u_new - u_k)
        if verbose:
            print(f"    Picard iter {k+1}: ||res|| = {res:.2e}")
        if res < tol_nl:
            return u_new, k + 1
        u_k = u_new

    warnings.warn(f"[Picard] Não convergiu em {max_iter} iterações "
                  f"(||res||={res:.2e}). Usando último iterate.")
    return u_k, max_iter


# ---------------------------------------------------------------------------
# Iteração de Newton
# ---------------------------------------------------------------------------

def newton_step(funcs, u, t_new, dt, n, rhs_hist,
                alpha, max_iter=20, tol_nl=1e-8, eps=1e-6, verbose=False):
    """
    Resolve um passo implícito via método de Newton com jacobiana numérica.

    Equação a resolver: G(u) = u - alpha*dt*F(u) - rhs_hist = 0

    Jacobiana por diferenças finitas:
        J[i,j] = (G_i(u + eps*e_j) - G_i(u)) / eps

    Parâmetros
    ----------
    alpha : float
        0.5 para CN, 2/3 para BDF2.
    eps : float
        Perturbação para diferenças finitas da jacobiana.

    Retorna
    -------
    u_new : np.ndarray
    n_iter : int
    """
    I = sp_sparse.eye(n, format='csr')
    u_k = u.copy()

    for k in range(max_iter):
        F_k = eval_F(funcs, t_new, u_k)
        G_k = u_k - alpha * dt * F_k - rhs_hist

        res = np.linalg.norm(G_k)
        if verbose:
            print(f"    Newton iter {k+1}: ||G|| = {res:.2e}")
        if res < tol_nl:
            return u_k, k

        # Jacobiana numérica de G: J_G = I - alpha*dt * J_F
        rows, cols, vals = [], [], []
        for j in range(n):
            u_pert = u_k.copy()
            u_pert[j] += eps
            F_pert = eval_F(funcs, t_new, u_pert)
            dF_j   = (F_pert - F_k) / eps
            dG_j   = -alpha * dt * dF_j
            dG_j[j] += 1.0  # contribuição de I

            nz = np.nonzero(np.abs(dG_j) > 1e-15)[0]
            rows.extend(nz.tolist())
            cols.extend([j] * len(nz))
            vals.extend(dG_j[nz].tolist())

        J_G = sp_sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))

        # Passo de Newton: u^{k+1} = u^k - J_G^{-1} * G(u^k)
        delta  = spsolve(J_G, G_k)
        u_k    = u_k - delta

    warnings.warn(f"[Newton] Não convergiu em {max_iter} iterações "
                  f"(||G||={res:.2e}). Usando último iterate.")
    return u_k, max_iter


# ---------------------------------------------------------------------------
# Histórico
# ---------------------------------------------------------------------------

def make_history(n_funcs, n):
    """Inicializa a estrutura de histórico."""
    use_groups = n_funcs is not None and (n % n_funcs == 0)
    n_elements = (n // n_funcs) if use_groups else n
    final_list = [[] for _ in range(n_funcs)] if use_groups else []
    return final_list, use_groups, n_elements


def save_to_history(u, final_list, use_groups, n_funcs, n_elements):
    """Salva o estado atual no histórico."""
    if use_groups:
        u_r = u.reshape((n_funcs, n_elements))
        for j in range(n_funcs):
            final_list[j].append(u_r[j].tolist())
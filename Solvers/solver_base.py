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


def _extract_L(funcs, n, u_ref, t_val, eps=1e-6):
    """
    Extrai a matriz jacobiana L = dF/du por diferenças finitas,
    aproveitando esparsidade via coloração de grafo.

    Retorna
    -------
    L : scipy.sparse.csr_matrix
    fonte : np.ndarray
        F(t, 0) — vetor fonte independente de u.
    """
    F_ref  = np.array([f(t_val, *u_ref) for f in funcs], dtype=np.float64)
    fonte  = np.array([f(t_val, *np.zeros(n)) for f in funcs], dtype=np.float64)

    # --- Detecção de esparsidade: quais (i,j) são não-nulos ---
    # Perturbamos cada coluna individualmente apenas para detectar estrutura.
    # Isso ocorre só uma vez em detect_linearity / primeira chamada.
    rows_all, cols_all, vals_all = [], [], []

    for j in range(n):
        u_pert    = u_ref.copy()
        u_pert[j] += eps
        F_pert = np.array([f(t_val, *u_pert) for f in funcs], dtype=np.float64)
        dF_j   = (F_pert - F_ref) / eps

        nz = np.nonzero(np.abs(dF_j) > 1e-15)[0]
        rows_all.extend(nz.tolist())
        cols_all.extend([j] * len(nz))
        vals_all.extend(dF_j[nz].tolist())

    L = sp_sparse.csr_matrix(
        (vals_all, (rows_all, cols_all)), shape=(n, n)
    )
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
# Esparsidade e coloração de grafo
# ---------------------------------------------------------------------------

def _detect_sparsity_pattern(funcs, n, t_val=0.0, eps=1e-6):
    """
    Detecta o padrão de esparsidade da jacobiana dF/du.

    Retorna
    -------
    sparsity : scipy.sparse.csr_matrix (booleano)
        sparsity[i, j] = True se dF_i/du_j pode ser não-nulo.
    colors : np.ndarray, shape (n,)
        Cor de cada coluna (inteiro >= 0). Colunas da mesma cor
        têm jacobianas disjuntas e podem ser perturbadas juntas.
    n_colors : int
    """
    u_ref = np.zeros(n)
    F_ref = np.array([f(t_val, *u_ref) for f in funcs], dtype=np.float64)

    rows_sp, cols_sp = [], []
    for j in range(n):
        u_pert    = u_ref.copy()
        u_pert[j] += eps
        F_pert = np.array([f(t_val, *u_pert) for f in funcs], dtype=np.float64)
        dF_j   = (F_pert - F_ref) / eps
        nz = np.nonzero(np.abs(dF_j) > 1e-15)[0]
        rows_sp.extend(nz.tolist())
        cols_sp.extend([j] * len(nz))

    data_sp  = np.ones(len(rows_sp), dtype=bool)
    sparsity = sp_sparse.csr_matrix(
        (data_sp, (rows_sp, cols_sp)), shape=(n, n)
    )

    # Coloração greedy das colunas: duas colunas j1, j2 conflitam se
    # existe alguma linha i com sparsity[i,j1] e sparsity[i,j2] ambos True.
    # Conflito ⟺ (sparsity.T @ sparsity)[j1, j2] > 0
    conflict = (sparsity.T @ sparsity).tocsr()
    colors   = np.full(n, -1, dtype=int)
    for j in range(n):
        # vizinhos já coloridos (colunas que conflitam com j)
        _, neighbors = conflict[j].nonzero()
        used = set(colors[neighbors[colors[neighbors] >= 0]])
        c = 0
        while c in used:
            c += 1
        colors[j] = c

    n_colors = int(colors.max()) + 1
    return sparsity, colors, n_colors


def _jacobian_sparse_colored(funcs, n, u_k, t_val, sparsity, colors, n_colors, eps=1e-6):
    """
    Calcula J_F = dF/du usando coloração de grafo.

    Em vez de n perturbações individuais, usa apenas n_colors perturbações
    (tipicamente 5-10x para grades de EDP), pois colunas da mesma cor
    têm linhas não-nulas disjuntas.

    Custo: n_colors avaliações de F  (vs n sem coloração)
    """
    F_k = np.array([f(t_val, *u_k) for f in funcs], dtype=np.float64)

    rows_J, cols_J, vals_J = [], [], []

    for c in range(n_colors):
        cols_c = np.where(colors == c)[0]

        # Perturbação simultânea de todas as colunas da cor c
        u_pert = u_k.copy()
        u_pert[cols_c] += eps

        F_pert = np.array([f(t_val, *u_pert) for f in funcs], dtype=np.float64)
        dF     = (F_pert - F_k) / eps  # contribuições mescladas das cols_c

        for j in cols_c:
            # Linhas não-nulas desta coluna (do padrão de esparsidade)
            rows_j = sparsity.getcol(j).nonzero()[0]
            if len(rows_j) == 0:
                continue
            rows_J.extend(rows_j.tolist())
            cols_J.extend([j] * len(rows_j))
            vals_J.extend(dF[rows_j].tolist())

    J_F = sp_sparse.csr_matrix((vals_J, (rows_J, cols_J)), shape=(n, n))
    return J_F, F_k


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
        fonte_k = eval_F(funcs, t_new, np.zeros(n))

        A     = I - alpha * dt * L_k
        b     = rhs_hist + alpha * dt * fonte_k
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
# Iteração de Newton  (com jacobiana esparsa por coloração)
# ---------------------------------------------------------------------------

def newton_step(funcs, u, t_new, dt, n, rhs_hist,
                alpha, max_iter=20, tol_nl=1e-8, eps=1e-6, verbose=False,
                _cache={}):
    """
    Resolve um passo implícito via método de Newton com jacobiana numérica
    esparsa (coloração de grafo).

    Equação a resolver: G(u) = u - alpha*dt*F(u) - rhs_hist = 0

    Melhoria vs versão anterior:
    - Detecta o padrão de esparsidade UMA VEZ (cacheado por n).
    - Usa coloração de grafo: apenas n_colors << n avaliações de F
      por iteração Newton, em vez de n avaliações.

    Parâmetros
    ----------
    alpha : float
        0.5 para CN, 2/3 para BDF2.
    eps : float
        Perturbação para diferenças finitas da jacobiana.
    _cache : dict
        Cache interno (não passar manualmente). Armazena padrão de
        esparsidade e coloração por chave n.

    Retorna
    -------
    u_new : np.ndarray
    n_iter : int
    """
    # --- Cache de esparsidade/coloração (detectado apenas na 1ª chamada) ---
    if n not in _cache:
        t_cache = time.time()
        sparsity, colors, n_colors = _detect_sparsity_pattern(funcs, n, eps=eps)
        _cache[n] = (sparsity, colors, n_colors)
        print(f"  [Newton] Esparsidade detectada: {sparsity.nnz} entradas não-nulas, "
              f"{n_colors} cores (vs {n} colunas) — {time.time()-t_cache:.3f}s")
    else:
        sparsity, colors, n_colors = _cache[n]

    I   = sp_sparse.eye(n, format='csr')
    u_k = u.copy()

    for k in range(max_iter):
        J_F, F_k = _jacobian_sparse_colored(
            funcs, n, u_k, t_new, sparsity, colors, n_colors, eps=eps
        )
        G_k = u_k - alpha * dt * F_k - rhs_hist

        res = np.linalg.norm(G_k)
        if verbose:
            print(f"    Newton iter {k+1}: ||G|| = {res:.2e}")
        if res < tol_nl:
            return u_k, k

        # J_G = I - alpha*dt * J_F
        J_G   = I - alpha * dt * J_F
        delta = spsolve(J_G, G_k)
        u_k   = u_k - delta

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

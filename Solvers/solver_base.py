import time
import warnings
import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
import scipy.sparse as sp_sparse
from scipy.sparse.linalg import spsolve

def compile_equations(flat_list, d_vars, verbose=False):
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

def detect_linearity(funcs, n, t0_val=0.0, verbose=False,
                     dirichlet_indices=None):
    t0 = time.time()
    zeros = np.zeros(n)

    L0, fonte = _extract_L(funcs, n, zeros, t0_val)
    F0 = fonte  

    rng = np.random.default_rng(42)
    is_linear = True
    for _ in range(3):
        u_rand = rng.standard_normal(n) * 0.5
        if dirichlet_indices:
            for idx in dirichlet_indices:
                u_rand[idx] = 0.0
        F_rand = np.array([f(float(t0_val), *u_rand) for f in funcs], dtype=np.float64)
        F_pred = L0 @ u_rand + F0
        err = np.max(np.abs(F_rand - F_pred))
        if err > 1e-6:
            is_linear = False
            break

    if verbose:
        status = "LINEAR" if is_linear else "NÃO-LINEAR"
        print(f"  Detecção de linearidade: {status} ({time.time()-t0:.3f}s)")

    return is_linear, L0


def _extract_L(funcs, n, u_ref, t_val, eps=1e-6):
    F_ref  = np.array([f(t_val, *u_ref) for f in funcs], dtype=np.float64)
    fonte  = np.array([f(t_val, *np.zeros(n)) for f in funcs], dtype=np.float64)

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


def extract_linear_structure(funcs, n, t0_val=0.0, verbose=False):
    t0 = time.time()
    L, _ = _extract_L(funcs, n, np.zeros(n), t0_val)

    z  = np.zeros(n)
    F0 = np.array([f(0.0, *z) for f in funcs], dtype=np.float64)
    F1 = np.array([f(1.0, *z) for f in funcs], dtype=np.float64)
    t_dependent = np.where(np.abs(F1 - F0) > 1e-14)[0]

    if len(t_dependent) == 0:
        _fonte_const = F0.copy()
        def fonte_func(t_val):
            return _fonte_const
        if verbose:
            print(f"  Extração estrutura A: {time.time()-t0:.3f}s  "
                  f"[fonte constante — pré-computada]")
    else:
        _fonte_const  = F0.copy()
        _t_dep_idx    = t_dependent
        _t_dep_funcs  = [funcs[i] for i in t_dependent]
        def fonte_func(t_val):
            out = _fonte_const.copy()
            for local_i, global_i in enumerate(_t_dep_idx):
                out[global_i] = _t_dep_funcs[local_i](float(t_val), *z)
            return out
        if verbose:
            print(f"  Extração estrutura A: {time.time()-t0:.3f}s  "
                  f"[fonte parcial: {len(t_dependent)}/{n} eqs dependem de t]")

    return L, fonte_func

def _detect_sparsity_pattern(funcs, n, t_val=0.0, eps=1e-6):
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

    conflict = (sparsity.T @ sparsity).tocsr()
    colors   = np.full(n, -1, dtype=int)
    for j in range(n):
        _, neighbors = conflict[j].nonzero()
        used = set(colors[neighbors[colors[neighbors] >= 0]])
        c = 0
        while c in used:
            c += 1
        colors[j] = c

    n_colors = int(colors.max()) + 1
    return sparsity, colors, n_colors


def _jacobian_sparse_colored(funcs, n, u_k, t_val, sparsity, colors, n_colors,
                              eps=1e-6, _csc_cache={}):

    F_k = np.array([f(t_val, *u_k) for f in funcs], dtype=np.float64)

    sp_id = id(sparsity)
    if sp_id not in _csc_cache:
        sp_csc   = sparsity.tocsc()
        col_rows = [sp_csc.indices[sp_csc.indptr[j]:sp_csc.indptr[j+1]]
                    for j in range(n)]
        _csc_cache[sp_id] = col_rows
    else:
        col_rows = _csc_cache[sp_id]

    all_rows = np.empty(sparsity.nnz, dtype=np.int32)
    all_cols = np.empty(sparsity.nnz, dtype=np.int32)
    all_vals = np.empty(sparsity.nnz, dtype=np.float64)

    ptr = 0
    for c in range(n_colors):
        cols_c = np.where(colors == c)[0]

        u_pert = u_k.copy()
        u_pert[cols_c] += eps
        F_pert = np.array([f(t_val, *u_pert) for f in funcs], dtype=np.float64)
        dF     = (F_pert - F_k) / eps

        for j in cols_c:
            rows_j = col_rows[j]
            k      = len(rows_j)
            if k == 0:
                continue
            all_rows[ptr:ptr+k] = rows_j
            all_cols[ptr:ptr+k] = j
            all_vals[ptr:ptr+k] = dF[rows_j]
            ptr += k

    J_F = sp_sparse.csr_matrix(
        (all_vals[:ptr], (all_rows[:ptr], all_cols[:ptr])), shape=(n, n)
    )
    return J_F, F_k


def eval_F(funcs, t_val, u):
    return np.array([f(float(t_val), *u) for f in funcs], dtype=np.float64)

def picard_step(funcs, u, t_new, dt, n, rhs_hist,
                alpha, max_iter=50, tol_nl=1e-8, verbose=False):
    
    I   = sp_sparse.eye(n, format='csr')
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

def newton_step(funcs, u, t_new, dt, n, rhs_hist,
                alpha, max_iter=20, tol_nl=1e-8, eps=1e-6, verbose=False,
                _cache={}):
    if n not in _cache:
        t_cache = time.time()
        sparsity, colors, n_colors = _detect_sparsity_pattern(funcs, n, eps=eps)
        _cache[n] = (sparsity, colors, n_colors)
        if verbose:
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

        J_G   = I - alpha * dt * J_F
        delta = spsolve(J_G, G_k)
        u_k   = u_k - delta

    warnings.warn(f"[Newton] Não convergiu em {max_iter} iterações "
                  f"(||G||={res:.2e}). Usando último iterate.")
    return u_k, max_iter

def make_history(n_funcs, n):

    use_groups = n_funcs is not None and (n % n_funcs == 0)
    n_elements = (n // n_funcs) if use_groups else n
    final_list = [[] for _ in range(n_funcs)] if use_groups else []
    return final_list, use_groups, n_elements


def save_to_history(u, final_list, use_groups, n_funcs, n_elements):
    if use_groups:
        u_r = u.reshape((n_funcs, n_elements))
        for j in range(n_funcs):
            final_list[j].append(u_r[j].tolist())
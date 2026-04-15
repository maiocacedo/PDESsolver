import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
import scipy.sparse as sp_sparse
from scipy.sparse.linalg import spsolve

def thomas_solver(a, b, c, d):
    """Resolve o sistema tridiagonal Ax = d."""
    n = len(d)
    cp, dp, x = np.zeros(n-1), np.zeros(n), np.zeros(n)
    
    # Eliminação progressiva
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        denom = b[i] - a[i] * cp[i-1]
        if i < n-1:
            cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i] * dp[i-1]) / denom
    
    # Substituição regressiva
    x[-1] = dp[-1]
    for i in range(n-2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i+1]
    return x

def cn(flat_list, d_vars, tf, nt, ic, n_funcs=None):
    """
    Solver genérico Crank-Nicolson N-Dimensional com saída compatível com SERKF45.
    """
    dt = tf / nt
    n = len(d_vars)
    u = np.array(ic, dtype=np.float64).flatten()
    
    t_sym = sp.Symbol('t')
    sym_list = [sp.Symbol(v) for v in d_vars]
    parsed_eqs = [parse_expr(eq_str) for eq_str in flat_list]
    
    # --- PREPARAÇÃO DO HISTÓRICO (Igual ao SERKF45) ---
    final_list = [[] for _ in range(n_funcs)] if (n_funcs and (n % n_funcs == 0)) else []
    n_elements = (n // n_funcs) if (n_funcs and (n % n_funcs == 0)) else n

    def save_to_history(current_u):
        if n_funcs and (n % n_funcs == 0):
            # Faz o reshape para separar os grupos e salva no histórico
            u_reshaped = current_u.reshape((n_funcs, n_elements))
            for jgrp in range(n_funcs):
                final_list[jgrp].append(u_reshaped[jgrp].tolist())

    # Salva a condição inicial
    save_to_history(u)

    # --- ETAPA DE COMPILAÇÃO ---
    mapa_coeficientes = []
    for expr in parsed_eqs:
        linha = {'coeffs': [], 'fonte': None}
        for j, sym in enumerate(sym_list):
            coeff_sym = expr.coeff(sym)
            if coeff_sym != 0:
                func_coeff = sp.lambdify((t_sym, *sym_list), coeff_sym)
                linha['coeffs'].append((j, func_coeff))
        
        fonte_sym = expr.as_coeff_Add()[0]
        linha['fonte'] = sp.lambdify((t_sym, *sym_list), fonte_sym)
        mapa_coeficientes.append(linha)

    # --- LOOP DE TEMPO ---
    for passo in range(nt):
        tempo_atual = passo * dt
        A_mat = sp_sparse.lil_matrix((n, n), dtype=np.float64)
        rhs = np.zeros(n, dtype=np.float64)
        u_args = tuple(u)
        
        for i in range(n):
            A_mat[i, i] = 1.0
            soma_implicita_un = 0.0
            
            for j, func_c in mapa_coeficientes[i]['coeffs']:
                c_val = func_c(tempo_atual, *u_args) 
                val_implicito = (dt / 2.0) * c_val
                A_mat[i, j] -= val_implicito
                soma_implicita_un += val_implicito * u[j]
            
            fonte_val = mapa_coeficientes[i]['fonte'](tempo_atual, *u_args)
            rhs[i] = u[i] + soma_implicita_un + dt * fonte_val
            
        A_csr = A_mat.tocsr()
        u = spsolve(A_csr, rhs)
        
        # Salva o estado atualizado no histórico
        save_to_history(u)
        
    # Retorna (u_final, final_list) para manter a consistência
    return u, final_list
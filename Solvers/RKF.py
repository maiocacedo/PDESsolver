
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
import cupy as cp
from Auxs.FuncAux import symbol_references
import re

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def SERKF45_cuda(oldexpr, ivar, funcs, yn, x0, xn, n, n_funcs, sp_vars, dt_max=None, tol=1e-5, dt_init=None):
    iteration = 0
    # =========================
    # 1) Simbolos consistentes
    # =========================

    olddvar = sorted(symbol_references(funcs), key=natural_sort_key)   # ['y1','y2',...]
    oldivar = symbol_references(ivar)    # ['t']
    sym_map = {name: sp.Symbol(name) for name in (oldivar + olddvar)}
    t_sym   = sym_map[oldivar[0]]
    y_syms  = [sym_map[name] for name in olddvar]

    exprs = [parse_expr(e, local_dict=sym_map, evaluate=False) for e in oldexpr]
    m = len(exprs)
    if m == 0:
        raise ValueError("Lista de EDOs vazia.")

    # =========================
    # 2) lambdify: retorna vetor (m,)
    # =========================
    cupy_map = {
        'sin': cp.sin, 'cos': cp.cos, 'tan': cp.tan,
        'asin': cp.arcsin, 'acos': cp.arctan, 'atan': cp.arctan, 'atan2': cp.arctan2,
        'sinh': cp.sinh, 'cosh': cp.cosh, 'tanh': cp.tanh,
        'exp': cp.exp, 'log': cp.log, 'sqrt': cp.sqrt,
        'Abs': cp.abs, 'sign': cp.sign,
        'Max': cp.maximum, 'Min': cp.minimum, 'mod': cp.mod,
        'floor': cp.floor, 'ceil': cp.ceil,
        'sech': lambda x: 1.0/cp.cosh(x),
    }
    F_tuple = sp.lambdify((t_sym, *y_syms), tuple(exprs), modules=[cupy_map, cp])

    def F_all(t_scalar, y_vec):
        out = F_tuple(t_scalar, *[y_vec[k] for k in range(m)])
        if not isinstance(out, (tuple, list)) or len(out) != m:
            raise RuntimeError("F retornou forma inesperada; espere tuple/list de tamanho m.")
        return cp.stack([cp.asarray(out[k]) for k in range(m)], axis=0)

    # =========================
    # 3) Estados e historico
    # =========================
    dtype = cp.float64
    print(f"Integrando {m} EDOs com SERKF45 (CUDA)...")
    y  = cp.asarray(yn, dtype=dtype).reshape(m,)
    if y.size != m:
        raise ValueError(f"len(yn) ({y.size}) != numero de EDOs ({m}).")

    if dt_init is None:
        h = dtype((xn - x0)/max(int(n), 1))
    else:
        h = dtype(float(dt_init))

    def clamp_h(h, t, t1, dt_max):
        if dt_max is not None and h > dt_max:
            h = dtype(dt_max)
        # Ajuste final para nao ultrapassar t1
        if t + h > t1:
            h = dtype(t1 - t)
        # Evita h muito pequeno nulo
        h = cp.maximum(h, dtype(1e-14))
        return h

    t = dtype(float(x0))
    t1 = dtype(float(xn))
    h = clamp_h(h, t, t1, dt_max)

    # Historico (opcional): estrutura igual ao seu codigo
    final_list = [[] for _ in range(n_funcs)] if (n_funcs and (m % n_funcs == 0)) else []
    n_elements = (m // n_funcs) if (n_funcs and (m % n_funcs == 0)) else m

    if n_funcs and (m % n_funcs == 0):
        y_host = y.get().reshape((n_funcs, n_elements))
        for jgrp in range(n_funcs):
            final_list[jgrp].append(y_host[jgrp].tolist())

    # =========================
    # 4) Buffers
    # =========================
    k1 = cp.empty_like(y); k2 = cp.empty_like(y); k3 = cp.empty_like(y)
    k4 = cp.empty_like(y); k5 = cp.empty_like(y); k6 = cp.empty_like(y)
    y4 = cp.empty_like(y); y5 = cp.empty_like(y)

    # Coeficientes Dormand-Prince
    c2, c3, c4, c5, c6 = 1/4, 3/8, 12/13, 1.0, 1/2
    a21 = 1/4
    a31, a32 = 3/32, 9/32
    a41, a42, a43 = 1932/2197, -7200/2197, 7296/2197
    a51, a52, a53, a54 = 439/216, -8.0, 3680/513, -845/4104
    a61, a62, a63, a64, a65 = -8/27, 2.0, -3544/2565, 1859/4104, -11/40
    # pesos 5a ordem
    b1, b3, b4, b5, b6 = 16/135, 6656/12825, 28561/56430, -9/50, 2/55
    # pesos 4a ordem
    b1s, b3s, b4s, b5s = 25/216, 1408/2565, 2197/4104, -1/5

    # Loop principal (aceita/rejeita passos)
    while float(t) < float(t1) - 1e-14:
        # Estagios em tempo ESCALAR (sincronizado)
        k1[...] = F_all(t, y)
        k2[...] = F_all(t + c2*h, y + h*(a21*k1))
        k3[...] = F_all(t + c3*h, y + h*(a31*k1 + a32*k2))
        k4[...] = F_all(t + c4*h, y + h*(a41*k1 + a42*k2 + a43*k3))
        k5[...] = F_all(t + c5*h, y + h*(a51*k1 + a52*k2 + a53*k3 + a54*k4))
        k6[...] = F_all(t + c6*h, y + h*(a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5))

        y4[...] = y + h*(b1s*k1 + b3s*k3 + b4s*k4 + b5s*k5)
        y5[...] = y + h*(b1 *k1 + b3 *k3 + b4 *k4 + b5 *k5 + b6 *k6)

        # Erro embutido (norma infinito)
        diff = cp.abs(y5 - y4)
        err  = float(diff.max().get())  # escalar host para decisao

        if err > tol:
            # Rejeita passo: reduz h (com clamp e teto)
            s = 0.84 * (tol/err)**0.25
            s = min(4.0, max(0.1, s))
            h = clamp_h(dtype(s)*h, t, t1, dt_max)

            continue
        else:
            
            y[...] = y5  # Aceita passo: avanca estado e tempo
            # Aceita passo
            if n_funcs and (m % n_funcs == 0):
                y_host = y.get().reshape((n_funcs, n_elements))
                for jgrp in range(n_funcs):
                    final_list[jgrp].append(y_host[jgrp].tolist())

            # y[...] = y4
            t = t + h
            iteration = iteration + 1
            #print(iteration)
            # Proximo h sugerido
            s = 0.84 * (tol/max(err, 1e-16))**0.25
            s = min(4.0, max(0.1, s))
            h = clamp_h(dtype(s)*h, t, t1, dt_max)

    # Estado final no host
    y_final = y.get()
    return y_final, final_list

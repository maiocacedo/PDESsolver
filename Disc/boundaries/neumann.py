"""
neumann.py
----------
Condicao de contorno de Neumann: du/dn = f(x, y, t).

A aproximacao usada e de primeira ordem (ghost-cell):
    u_ghost = h * f(x_bd, y_bd, t) + u_interior

As coordenadas x e y sao substituidas pelos valores numericos
de cada ponto do contorno, exatamente como faz o DirichletBC.
Isso garante que as expressoes geradas nao contenham simbolos
livres alem de 't' e das variaveis discretizadas XX*, o que e
exigido pelo lambdify em solver_base._extract_L.
"""

from typing import List
from Auxs.FuncAux import repl_symbol as _repl_symbol
from .boundary_base import BoundaryCondition


class NeumannBC(BoundaryCondition):

    def __init__(self, bd_func: str):
        super().__init__(bd_func)

    # ------------------------------------------------------------------
    # Utilitario: substitui x e y pelo valor numerico do ponto
    # ------------------------------------------------------------------

    def _replace_xy(self, expr: str, X: str, Y: str, str_sp_vars: str) -> str:
        out = _repl_symbol(expr, str_sp_vars[0], X)
        if len(str_sp_vars) == 2:
            out = _repl_symbol(out, str_sp_vars[1], Y)
        return out

    # ------------------------------------------------------------------
    # Despacho principal
    # ------------------------------------------------------------------

    def apply(
        self,
        bd: str,
        list_eq: List[List[str]],
        n_part: List[int],
        xd_var: List[str],
        str_sp_vars: str = "",
    ) -> List[List[str]]:

        is_2d = len(str_sp_vars) == 2
        self._check_side(bd, is_2d)
        bd = bd.lower()

        if is_2d:
            return self._apply_2d(bd, list_eq, n_part, xd_var, str_sp_vars)
        return self._apply_1d(bd, list_eq, n_part, xd_var, str_sp_vars)

    # ------------------------------------------------------------------
    # 2D
    # ------------------------------------------------------------------

    def _apply_2d(self, bd, list_eq, n_part, xd_var, str_sp_vars):
        Nx, Ny = n_part[0], n_part[1]
        n_funcs = len(list_eq)
        result = [[] for _ in range(n_funcs)]
        hx = f"h{xd_var[0]}_"
        hy = f"h{xd_var[0]}_"   # malha uniforme: hx == hy

        # Os pontos internos estao ordenados como:
        #   list_eq[func][k]  onde k = (i-1)*(Ny-2) + (j-1)
        #   para i in [1, Nx-2], j in [1, Ny-2]

        if bd == "west":
            # i = 0, j varia de 0 ate Ny-1
            # interior vizinho: i=1  ->  list_eq index j-1  (para j=1..Ny-2)
            # Cantos (j=0 e j=Ny-1) usam o vizinho interior mais proximo
            for func in range(n_funcs):
                for j in range(Ny):
                    x_val = f"0 * {hx}"
                    y_val = f"{j} * {hy}"
                    bc_expr = self._replace_xy(self.bd_func, x_val, y_val, str_sp_vars)

                    # indice do interior vizinho em x (i=1): coluna 0 de list_eq
                    j_inner = max(0, min(j - 1, Ny - 3))
                    interior = list_eq[func][j_inner]           # eq em (1, j_inner+1)
                    result[func].append(f"{hx}*({bc_expr})+{interior}")

        elif bd == "east":
            # i = Nx-1, j varia de 0 ate Ny-1
            # interior vizinho: i=Nx-2 -> coluna (Nx-3) de list_eq
            for func in range(n_funcs):
                for j in range(Ny):
                    x_val = f"{Nx-1} * {hx}"
                    y_val = f"{j} * {hy}"
                    bc_expr = self._replace_xy(self.bd_func, x_val, y_val, str_sp_vars)

                    j_inner = max(0, min(j - 1, Ny - 3))
                    # indice base da ultima coluna interna (i=Nx-2)
                    base = (Nx - 3) * (Ny - 2)
                    interior = list_eq[func][base + j_inner]
                    result[func].append(f"{hx}*({bc_expr})+{interior}")

        elif bd == "south":
            # j = 0, i varia de 0 ate Nx-1
            # interior vizinho: j=1 -> linha 0 de cada bloco i
            for func in range(n_funcs):
                for i in range(Nx):
                    x_val = f"{i} * {hx}"
                    y_val = f"0 * {hy}"
                    bc_expr = self._replace_xy(self.bd_func, x_val, y_val, str_sp_vars)

                    i_inner = max(0, min(i - 1, Nx - 3))
                    interior = list_eq[func][i_inner * (Ny - 2)]   # j=1 -> indice 0 do bloco
                    result[func].append(f"{hy}*({bc_expr})+{interior}")

        elif bd == "north":
            # j = Ny-1, i varia de 0 ate Nx-1
            # interior vizinho: j=Ny-2 -> ultimo elemento de cada bloco i
            for func in range(n_funcs):
                for i in range(Nx):
                    x_val = f"{i} * {hx}"
                    y_val = f"{Ny-1} * {hy}"
                    bc_expr = self._replace_xy(self.bd_func, x_val, y_val, str_sp_vars)

                    i_inner = max(0, min(i - 1, Nx - 3))
                    interior = list_eq[func][i_inner * (Ny - 2) + (Ny - 3)]
                    result[func].append(f"{hy}*({bc_expr})+{interior}")

        return result

    # ------------------------------------------------------------------
    # 1D
    # ------------------------------------------------------------------

    def _apply_1d(self, bd, list_eq, n_part, xd_var, str_sp_vars):
        Nx = n_part[0]
        hx = f"h{xd_var[0]}_"
        result = [[] for _ in range(len(list_eq))]

        if bd == "west":
            for func in range(len(list_eq)):
                x_val = f"0 * {hx}"
                bc_expr = self._replace_xy(self.bd_func, x_val, "", str_sp_vars)
                interior = list_eq[func][0]
                result[func].append(f"{hx}*({bc_expr})+{interior}")

        elif bd == "east":
            for func in range(len(list_eq)):
                x_val = f"{Nx-1} * {hx}"
                bc_expr = self._replace_xy(self.bd_func, x_val, "", str_sp_vars)
                interior = list_eq[func][-1]
                result[func].append(f"{hx}*({bc_expr})+{interior}")

        return result
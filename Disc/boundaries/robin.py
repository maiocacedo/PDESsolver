"""
robin.py
--------
Condicao de contorno de Robin (mista): alpha*u + beta*du/dn = f.

Forma discreta:
    u_ghost = (h*f(x_bd, y_bd, t) - u_interior*(alpha*h - beta)) / beta

Os parametros alpha e beta sao strings (podem conter expressoes
simbolicas). A expressao bd_func pode depender de 'x', 'y' e 't';
as coordenadas x e y sao substituidas numericamente em cada ponto
do contorno (igual ao DirichletBC/NeumannBC), e 't' e mantido como
simbolo livre avaliado pelo solver.
"""

from typing import List
from Auxs.FuncAux import repl_symbol as _repl_symbol
from .boundary_base import BoundaryCondition


class RobinBC(BoundaryCondition):

    def __init__(self, bd_func: str, alpha: str = "0", beta: str = "1"):
        super().__init__(bd_func)
        self.alpha = alpha
        self.beta  = beta

    # ------------------------------------------------------------------
    # Utilitario: substitui x e y pelo valor numerico do ponto
    # ------------------------------------------------------------------

    def _replace_xy(self, expr: str, X: str, Y: str, str_sp_vars: str) -> str:
        out = _repl_symbol(expr, str_sp_vars[0], X)
        if len(str_sp_vars) == 2:
            out = _repl_symbol(out, str_sp_vars[1], Y)
        return out

    # ------------------------------------------------------------------
    # Formula de Robin (reutilizada em todos os lados)
    # ------------------------------------------------------------------

    def _robin_expr(self, h_sym: str, bc_expr: str, u_interior: str) -> str:
        return (
            f"({h_sym}*({bc_expr})"
            f"-({u_interior})*({self.alpha}*{h_sym}-{self.beta}))"
            f"/({self.beta})"
        )

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

        if bd == "west":
            # i = 0, j varia de 0 ate Ny-1
            for func in range(n_funcs):
                for j in range(Ny):
                    x_val = f"0 * {hx}"
                    y_val = f"{j} * {hy}"
                    bc_expr = self._replace_xy(self.bd_func, x_val, y_val, str_sp_vars)
                    j_inner = max(0, min(j - 1, Ny - 3))
                    interior = list_eq[func][j_inner]
                    result[func].append(self._robin_expr(hx, bc_expr, interior))

        elif bd == "east":
            # i = Nx-1, j varia de 0 ate Ny-1
            for func in range(n_funcs):
                for j in range(Ny):
                    x_val = f"{Nx-1} * {hx}"
                    y_val = f"{j} * {hy}"
                    bc_expr = self._replace_xy(self.bd_func, x_val, y_val, str_sp_vars)
                    j_inner = max(0, min(j - 1, Ny - 3))
                    base = (Nx - 3) * (Ny - 2)
                    interior = list_eq[func][base + j_inner]
                    result[func].append(self._robin_expr(hx, bc_expr, interior))

        elif bd == "south":
            # j = 0, i varia de 0 ate Nx-1
            for func in range(n_funcs):
                for i in range(Nx):
                    x_val = f"{i} * {hx}"
                    y_val = f"0 * {hy}"
                    bc_expr = self._replace_xy(self.bd_func, x_val, y_val, str_sp_vars)
                    i_inner = max(0, min(i - 1, Nx - 3))
                    interior = list_eq[func][i_inner * (Ny - 2)]
                    result[func].append(self._robin_expr(hy, bc_expr, interior))

        elif bd == "north":
            # j = Ny-1, i varia de 0 ate Nx-1
            for func in range(n_funcs):
                for i in range(Nx):
                    x_val = f"{i} * {hx}"
                    y_val = f"{Ny-1} * {hy}"
                    bc_expr = self._replace_xy(self.bd_func, x_val, y_val, str_sp_vars)
                    i_inner = max(0, min(i - 1, Nx - 3))
                    interior = list_eq[func][i_inner * (Ny - 2) + (Ny - 3)]
                    result[func].append(self._robin_expr(hy, bc_expr, interior))

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
                result[func].append(self._robin_expr(hx, bc_expr, interior))

        elif bd == "east":
            for func in range(len(list_eq)):
                x_val = f"{Nx-1} * {hx}"
                bc_expr = self._replace_xy(self.bd_func, x_val, "", str_sp_vars)
                interior = list_eq[func][-1]
                result[func].append(self._robin_expr(hx, bc_expr, interior))

        return result
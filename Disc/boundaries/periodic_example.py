"""
periodic.py  — EXEMPLO de como adicionar um novo tipo de contorno
-----------------------------------------------------------------
Condição de contorno periódica: u(0) = u(L).

Como registrar:
    Em boundaries/__init__.py, adicione:
        from .periodic import PeriodicBC
        BOUNDARY_REGISTRY["periodic"] = PeriodicBC
"""

from typing import List
from .boundary_base import BoundaryCondition


class PeriodicBC(BoundaryCondition):
    """
    Condição periódica: o valor no contorno é igual ao valor no lado oposto.
    (Implementação ilustrativa — adapte conforme sua malha.)
    """

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

        result = [[] for _ in range(len(list_eq))]

        if not is_2d:
            # 1D: oeste = leste (último ponto interno)
            for func in range(len(list_eq)):
                if bd == "west":
                    result[func].append(list_eq[func][-1])
                elif bd == "east":
                    result[func].append(list_eq[func][0])
        else:
            # 2D: implementar conforme necessidade
            raise NotImplementedError("Periódico 2D ainda não implementado.")

        return result

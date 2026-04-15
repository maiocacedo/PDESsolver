import sympy as sp
import numpy as np

class PDE:
    
    eq = ''
    func = ''
    expr_ic = ''
    sp_var = []
    ivar = []
    ivar_boundary = []
    ic = []
    
    def __init__(self, eq, func, sp_var, ivar, ivar_boundary,expr_ic):
        self.eq = eq
        self.func = func     
        self.expr_ic = expr_ic
        self.sp_var = sp_var     
        self.ivar = ivar
        self.ivar_boundary = ivar_boundary

    
   
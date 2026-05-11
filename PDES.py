from Disc.Disc import df
from Solvers.CN import cn
import Solvers.RKF as SERKF45
from Solvers.bdf2 import bdf2
from Auxs.Visualize import visualize as _visualize
import sympy as sp
import numpy as np
import matplotlib
matplotlib.use('TkAgg')


class PDES:

    def __init__(self, pdes, disc_n, n_sp=1, n_temp=1):
        self.pdes    = pdes
        self.eqs     = [pde.eq   for pde in pdes]
        self.ivars   = pdes[0].ivar
        self.disc_results          = None
        self.dirichlet_constraints = {}
        self.neumann_constraints   = {}

        self.funcs   = [pde.func for pde in pdes]
        self.sp_vars = pdes[0].sp_var
        self.disc_n  = disc_n
        self.ic      = self._ic_calc(pdes)
        self.results = None

    def _ic_calc(self, pdes):
        all_ics = []
        for pde in pdes:
            expr       = sp.parse_expr(pde.expr_ic)
            sp_symbols = [sp.Symbol(v) for v in pde.sp_var]
            grids      = []
            for i in range(len(pde.sp_var)):
                a, b = pde.ivar_boundary[i]
                grids.append(np.linspace(a, b, self.disc_n[i]))
            mesh      = np.meshgrid(*grids, indexing='ij')
            f_ic      = sp.lambdify(sp_symbols, expr, modules='numpy')
            ic_values = f_ic(*mesh)
            if np.isscalar(ic_values):
                ic_values = np.broadcast_to(ic_values, mesh[0].shape)
            all_ics.extend(ic_values.flatten().tolist())
        return all_ics

    def xs(self, vars):
        nvars = vars.copy()
        for i in range(len(nvars)):
            nvars[i] = f'XX{i}'
        return nvars

    def discretize(self, method='central'):
        flat_list, d_vars, dirichlet_constraints, neumann_constraints = df(
            self, self.disc_n,
            method=method,
            west_bd       = [pde.west_bd       for pde in self.pdes],
            west_func_bd  = [pde.west_func_bd  for pde in self.pdes],
            east_bd       = [pde.east_bd       for pde in self.pdes],
            east_func_bd  = [pde.east_func_bd  for pde in self.pdes],
            north_bd      = [pde.north_bd      for pde in self.pdes],
            north_func_bd = [pde.north_func_bd for pde in self.pdes],
            south_bd      = [pde.south_bd      for pde in self.pdes],
            south_func_bd = [pde.south_func_bd for pde in self.pdes],
        )
        self.disc_results          = (flat_list, d_vars)
        self.dirichlet_constraints = dirichlet_constraints
        self.neumann_constraints   = neumann_constraints

    def solve(self, method='bdf2', tf=1.0, nt=100, tol=1e-6, verbose=False, **kwargs):
        dt = tf / nt
        dc = self.dirichlet_constraints
        nc = getattr(self, 'neumann_constraints', {})

        if method == 'bdf2':
            self.results = bdf2(
                self.disc_results[0], self.disc_results[1],
                tf=tf, nt=nt, ic=self.ic,
                n_funcs=len(self.funcs),
                dirichlet_constraints=dc,
                neumann_constraints=nc,
                verbose=verbose,
                **kwargs
            )
        elif method == 'CN':
            self.results = cn(
                self.disc_results[0], self.disc_results[1],
                tf=tf, nt=nt, ic=self.ic,
                n_funcs=len(self.funcs),
                dirichlet_constraints=dc,
                neumann_constraints=nc,
                verbose=verbose,
                **kwargs
            )
        elif method == 'RKF':
            self.results = SERKF45.SERKF45_cuda(
                self.disc_results[0],
                ivar=self.ivars,
                funcs=self.disc_results[1],
                yn=self.ic,
                sp_vars=self.sp_vars,
                n=100,
                n_funcs=len(self.funcs),
                dt_init=dt,
                tol=tol,
                x0=0,
                xn=nt * dt,
                dirichlet_constraints=dc,
                neumann_constraints=nc,
                verbose=verbose
            )
        else:
            raise ValueError(
                f"Metodo '{method}' desconhecido. Use: 'bdf2', 'CN' ou 'RKF'."
            )
        return self.results

    def visualize(self, mode='heatmap', func_idx=0, time_step=-1, **kwargs):
        _visualize(self, mode=mode, func_idx=func_idx,
                   time_step=time_step, **kwargs)

    def __repr__(self):
        status = 'resolvido' if self.results is not None else 'nao resolvido'
        return (f"PDES(funcs={self.funcs}, disc_n={self.disc_n}, "
                f"sp_vars={self.sp_vars}, status='{status}')")

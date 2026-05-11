import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PDE import PDE
from PDES import PDES

u_eq = PDE(
    "dU/dt = 0.16*d2U/dx2 - U*V*2 + 0.035*(1 - U)",
    func="U", sp_var=["x"], ivar=["t"],
    ivar_boundary=[(0, 1)],
    expr_ic="1 - 0.5*exp(-200*(x-0.5)**2)",
    west_bd="Neumann", west_func_bd="0",
    east_bd="Neumann", east_func_bd="0"
)

v_eq = PDE(
    "dV/dt = 0.08*d2V/dx2 + U*V**2 - (0.035+0.065)*V",
    func="V", sp_var=["x"], ivar=["t"],
    ivar_boundary=[(0, 1)],
    expr_ic="0.25*exp(-200*(x-0.5)**2)",
    west_bd="Neumann", west_func_bd="0",
    east_bd="Neumann", east_func_bd="0"
)

sim = PDES([u_eq, v_eq], disc_n=[41])
sim.discretize(method="central")
sim.solve(method="bdf2", tf=20.0, nt=1000, tol=1e-6)
sim.visualize(mode="plot1d", func_idx=1) 
sim.visualize(mode="animation1d", func_idx=1)  
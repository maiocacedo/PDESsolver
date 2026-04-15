from matplotlib.animation import FuncAnimation

from Auxs.FuncAux import symbol_references
from Disc.Disc import df
import Solvers.CN as CN
import Solvers.RKF as SERKF45
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


class PDES:
    eqs = []
    ivars = []
    sp_vars = []
    funcs = []
    ic = []
    n_sp = 1
    n_temp = 1
    disc_n = []
    disc_results = None
    results = None
    
    def __init__(self, pdes, disc_n, n_sp=1, n_temp=1):
        self.eqs = [pde.eq for pde in pdes]
        
        # Coleta as funções de todas as PDEs
        self.funcs = [pde.func for pde in pdes]
        
        # Como é um sistema, o domínio de tempo e espaço é compartilhado
        # Pegamos a referência da primeira PDE para guiar a malha geral
        self.ivars = pdes[0].ivar
        self.sp_vars = pdes[0].sp_var
        
        self.n_sp = n_sp
        self.n_temp = n_temp
        self.disc_n = disc_n
        
        # CHAMA O CÁLCULO DAS CONDIÇÕES INICIAIS AQUI
        self.ic = self.ic_calc(pdes)
        
        self.disc_results = None
        self.results = None
    
    def ic_calc(self, pdes):
        """Calcula a condição inicial numérica para cada PDE no grid definido."""
        all_ics = []
        
        for pde in pdes:
            expr = sp.parse_expr(pde.expr_ic)
            
            # CORREÇÃO: Agora usamos sp_var em vez de ivar para construir a malha
            sp_symbols = [sp.Symbol(v) for v in pde.sp_var]

            grids = []
            for i, var_name in enumerate(pde.sp_var):
                # Usa os limites e a discretização correspondentes à variável espacial
                a, b = pde.ivar_boundary[i]
                n = self.disc_n[i]
                grids.append(np.linspace(a, b, n))
            
            mesh = np.meshgrid(*grids, indexing='ij')
            
            # Compila a expressão usando os símbolos espaciais
            f_ic = sp.lambdify(sp_symbols, expr, modules='numpy')
            
            ic_values = f_ic(*mesh)
            
            if np.isscalar(ic_values):
                ic_values = np.broadcast_to(ic_values, mesh[0].shape)
            
            all_ics.extend(ic_values.flatten().tolist())
            
        return all_ics
    
    def xs(self,vars):
        nvars = vars.copy()
        for i in range(len(nvars)): nvars[i] = f'XX{i}'
        return nvars

    
    def discretize(self, west_func_bd="0", method="central", west_bd="Dirichlet", north_bd="Dirichlet", south_bd="Dirichlet", east_bd="Dirichlet",  north_func_bd='0', south_func_bd='0', east_func_bd='0'):
        
        disc_results = df( 
                        self, self.disc_n,
                        west_func_bd=west_func_bd,  
                        west_bd=west_bd,
                        method=method,
                        north_bd=north_bd, south_bd=south_bd, east_bd=east_bd,  
                        north_func_bd= north_func_bd,   
                        south_func_bd= south_func_bd,
                        east_func_bd= east_func_bd
                    )
        
        self.disc_results = disc_results
        return disc_results
    
    def solve(self, method='CN', tf=1.0, nt=100, tol=1e-6):
        dt = tf / nt
        if method == 'CN':
            results = CN.cn(self.disc_results[0], self.disc_results[1], tf = tf, nt = nt, ic = self.ic, n_funcs = len(self.funcs))
            self.results = results
            return results
        
        elif method == 'RKF':
            results = SERKF45.SERKF45_cuda(self.disc_results[0], 
                                           ivar = self.ivars, 
                                           funcs = self.disc_results[1], 
                                           yn = self.ic, 
                                           sp_vars = self.sp_vars, 
                                           n = 100,
                                           n_funcs = len(self.funcs), 
                                           dt_init = dt, 
                                           tol = tol, 
                                           x0=0, 
                                           xn=nt*dt )
            self.results = results
            return results
        else:
            raise ValueError(f"Método de solução desconhecido: {method}")


    def visualize(self, mode='heatmap', func_idx=0, time_step=-1, **kwargs):
        if self.results is None:
            print("Erro: Você precisa rodar .solve() antes de visualizar.")
            return

        # Desempacota o resultado final e o histórico
        _, historico = self.results
        
        # Prepara a malha (assumindo 2D)
        nx, ny = self.disc_n
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y, indexing='ij')

        if mode == 'heatmap':
            self._plot_heatmap(historico, X, Y, func_idx, time_step)
        elif mode == 'animation':
            self._animate(historico, X, Y, func_idx)
        # Adicione outros modos conforme necessário...

    def _plot_heatmap(self, historico, X, Y, func_idx, time_step):
        data = np.array(historico[func_idx][time_step]).reshape(self.disc_n)
        plt.figure(figsize=(7, 6))
        contorno = plt.contourf(X, Y, data, levels=30, cmap='hot')
        plt.title(f"Distribuição de {self.funcs[func_idx]} no passo {time_step}")
        plt.colorbar(contorno)
        plt.show()
        
    def _animate(self, historico, X, Y, func_idx, **kwargs):

        # Extrai os parâmetros opcionais (ou usa o padrão)
        frames_step = kwargs.get('frames_step', 1)
        interval = kwargs.get('interval', 50)
        
        fig, ax = plt.subplots(figsize=(7, 6))
        
        # Pega a malha do tempo t=0
        Z_inicial = np.array(historico[func_idx][0]).reshape(self.disc_n)
        
        # Plota o frame inicial e fixa a barra de cores
        contorno = ax.contourf(X, Y, Z_inicial, levels=30, cmap='hot')
        fig.colorbar(contorno, ax=ax)
        
        def update(frame):
            ax.clear() # Limpa o frame anterior
            
            # Pega a malha do tempo atual
            Z = np.array(historico[func_idx][frame]).reshape(self.disc_n)
            
            # Redesenha
            cont = ax.contourf(X, Y, Z, levels=30, cmap='hot')
            ax.set_title(f"Evolução de {self.funcs[func_idx]} - Passo {frame}")
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            return cont,

        # Cria a lista de frames pulando de acordo com frames_step
        total_frames = len(historico[func_idx])
        frames_list = range(0, total_frames, frames_step)
        
        # Roda a animação
        anim = FuncAnimation(fig, update, frames=frames_list, interval=interval, blit=False)
        plt.show()
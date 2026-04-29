from matplotlib.animation import FuncAnimation
from Disc.Disc import df
from Solvers.CN import cn
import Solvers.RKF as SERKF45
from Solvers.bdf2 import bdf2
import sympy as sp
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # ou 'Qt5Agg'
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

    def solve(self, method='CN', tf=1.0, nt=100, tol=1e-6, **kwargs):
        dt = tf / nt
        if method == 'CN':
            results = cn(self.disc_results[0], self.disc_results[1],
                         tf=tf, nt=nt, ic=self.ic,
                         n_funcs=len(self.funcs), **kwargs)
            self.results = results
            return results

        elif method == 'RKF':
            results = SERKF45.SERKF45_cuda(self.disc_results[0],
                                           ivar=self.ivars,
                                           funcs=self.disc_results[1],
                                           yn=self.ic,
                                           sp_vars=self.sp_vars,
                                           n=100,
                                           n_funcs=len(self.funcs),
                                           dt_init=dt,
                                           tol=tol,
                                           x0=0,
                                           xn=nt * dt)
            self.results = results
            return results

        elif method == 'bdf2':
            results = bdf2(self.disc_results[0], self.disc_results[1],
                           tf=tf, nt=nt, ic=self.ic,
                           n_funcs=len(self.funcs), **kwargs)
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
            self._animate(historico, X, Y, func_idx, **kwargs)
        elif mode == 'plot3d':
            self._plot3d(historico, X, Y, func_idx, time_step, **kwargs)
        elif mode == 'animation3d':
            self._animate3d(historico, X, Y, func_idx, **kwargs)
        else:
            print(f"Modo '{mode}' desconhecido. Use: 'heatmap', 'animation', 'plot3d' ou 'animation3d'.")

    def _plot_heatmap(self, historico, X, Y, func_idx, time_step):
        data = np.array(historico[func_idx][time_step]).reshape(self.disc_n)
        plt.figure(figsize=(7, 6))
        contorno = plt.contourf(X, Y, data, levels=30, cmap='hot')
        plt.title(f"Distribuição de {self.funcs[func_idx]} no passo {time_step}")
        plt.colorbar(contorno)
        plt.show()

    def _animate(self, historico, X, Y, func_idx, **kwargs):
        frames_step = kwargs.get('frames_step', 1)
        interval = kwargs.get('interval', 50)

        fig, ax = plt.subplots(figsize=(7, 6))

        Z_inicial = np.array(historico[func_idx][0]).reshape(self.disc_n)
        contorno = ax.contourf(X, Y, Z_inicial, levels=30, cmap='hot')
        fig.colorbar(contorno, ax=ax)

        def update(frame):
            ax.clear()
            Z = np.array(historico[func_idx][frame]).reshape(self.disc_n)
            cont = ax.contourf(X, Y, Z, levels=30, cmap='hot')
            ax.set_title(f"Evolução de {self.funcs[func_idx]} - Passo {frame}")
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            return cont,

        total_frames = len(historico[func_idx])
        frames_list = range(0, total_frames, frames_step)
        anim = FuncAnimation(fig, update, frames=frames_list, interval=interval, blit=False)
        plt.show()

    def _plot3d(self, historico, X, Y, func_idx, time_step, **kwargs):
        """
        Superfície 3D da solução em um instante de tempo.

        Parâmetros kwargs opcionais
        --------------------------
        cmap      : str   — mapa de cores (padrão: 'hot')
        alpha     : float — transparência da superfície (padrão: 1.0)
        elev      : float — ângulo de elevação da câmera (padrão: 30)
        azim      : float — ângulo azimutal da câmera   (padrão: -60)
        """
        cmap = kwargs.get('cmap', 'hot')
        alpha = kwargs.get('alpha', 1.0)
        elev = kwargs.get('elev', 30)
        azim = kwargs.get('azim', -60)

        data = np.array(historico[func_idx][time_step]).reshape(self.disc_n)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_surface(X, Y, data, cmap=cmap, alpha=alpha)

        label = time_step if time_step >= 0 else len(historico[func_idx]) + time_step
        ax.set_title(f"Superfície 3D de {self.funcs[func_idx]} - Passo {label}")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel(str(self.funcs[func_idx]))
        ax.view_init(elev=elev, azim=azim)

        plt.tight_layout()
        plt.show()

    def _animate3d(self, historico, X, Y, func_idx, **kwargs):
        """
        Animação 3D da evolução temporal da solução.

        Parâmetros kwargs opcionais
        --------------------------
        frames_step : int   — intervalo entre frames (padrão: 1)
        interval    : int   — milissegundos entre frames (padrão: 50)
        cmap        : str   — mapa de cores (padrão: 'hot')
        alpha       : float — transparência da superfície (padrão: 1.0)
        elev        : float — ângulo de elevação da câmera (padrão: 30)
        azim        : float — ângulo azimutal da câmera    (padrão: -60)
        """
        frames_step = kwargs.get('frames_step', 1)
        interval = kwargs.get('interval', 50)
        cmap = kwargs.get('cmap', 'hot')
        alpha = kwargs.get('alpha', 1.0)
        elev = kwargs.get('elev', 30)
        azim = kwargs.get('azim', -60)

        # Calcula limites globais de Z para manter a escala fixa durante a animação
        todos_frames = [
            np.array(historico[func_idx][f]).reshape(self.disc_n)
            for f in range(0, len(historico[func_idx]), frames_step)
        ]
        z_min = min(Z.min() for Z in todos_frames)
        z_max = max(Z.max() for Z in todos_frames)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        def update(frame):
            ax.clear()
            Z = todos_frames[frame]
            ax.plot_surface(X, Y, Z, cmap=cmap, alpha=alpha)
            ax.set_title(f"Evolução 3D de {self.funcs[func_idx]} - Passo {frame * frames_step}")
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel(str(self.funcs[func_idx]))
            ax.set_zlim(z_min, z_max)
            ax.view_init(elev=elev, azim=azim)

        anim = FuncAnimation(fig, update, frames=len(todos_frames), interval=interval, blit=False)
        plt.show()
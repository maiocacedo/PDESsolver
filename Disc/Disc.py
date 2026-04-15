
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
import re
from Auxs.FuncAux import d_dt, repl_symbol as _repl_symbol
from Auxs.FuncAux import symbol_references


# Função principal de discretização por diferenças finitas
def df(pdes, n_part, west_bd = "neumann", method="forward", north_bd = "neumann", south_bd = "neumann", east_bd = "neumann",
       north_func_bd = "0", south_func_bd = "0", west_func_bd = "0", east_func_bd = "0", north_alpha_bd = "0", south_alpha_bd = "0", east_alpha_bd = "0", north_beta_bd = "1", south_beta_bd = "1", east_beta_bd = "1" ):

        xd_var = pdes.xs(pdes.funcs)
        eqrs = [eq.split('=')[1] for eq in pdes.eqs]
        str_sp_vars = ''

        for i in range(len(pdes.sp_vars)):
            str_sp_vars = str_sp_vars + pdes.sp_vars[i]

        for j in range(len(eqrs)):
            for i in range(len(xd_var)):
                eqrs[j] = eqrs[j].replace(f'{str(pdes.funcs[i])}', f'{xd_var[i]}{str_sp_vars}')
        
        # substituir as derivadas parciais pelas diferenças finitas avançadas
        if (method == "forward"):
            for j in range(len(eqrs)):
                for k in range(len(str_sp_vars)):
                    if k == 0:
                        for i in range(len(xd_var)):
                            eqrs[j] = eqrs[j].replace(f'd2{xd_var[i]}{str_sp_vars}/d{str_sp_vars[k]}2', f'({xd_var[i]}_i+1_j - 2*{xd_var[i]}_ii_j + {xd_var[i]}_i-1_j)/ h{xd_var[i]}_ ** 2')
                            eqrs[j] = eqrs[j].replace(f'd{xd_var[i]}{str_sp_vars}/d{str_sp_vars[k]}', f'({xd_var[i]}_i+1_j - {xd_var[i]}_ii_j)/ h{xd_var[i]}_')
                    elif k == 1:
                        for i in range(len(xd_var)):
                            eqrs[j] = eqrs[j].replace(f'd2{xd_var[i]}{str_sp_vars}/d{str_sp_vars[k]}2', f'({xd_var[i]}_ii_j+1 - 2*{xd_var[i]}_ii_j + {xd_var[i]}_ii_j-1)/ h{xd_var[i]}_ ** 2')
                            eqrs[j] = eqrs[j].replace(f'd{xd_var[i]}{str_sp_vars}/d{str_sp_vars[k]}', f'({xd_var[i]}_ii_j+1 - {xd_var[i]}_ii_j)/ h{xd_var[i]}_')
                for i in range(len(xd_var)):
                    eqrs[j] = eqrs[j].replace(f'{xd_var[i]}{str_sp_vars}', f'{xd_var[i]}_ii_j')
                
            for j in range(len(eqrs)):
                eqrs[j] = _repl_symbol(eqrs[j], f'{str_sp_vars[0]}', f'ii * h{xd_var[0]}_')
                if len(str_sp_vars) == 2:
                    eqrs[j] = _repl_symbol(eqrs[j], f'{str_sp_vars[1]}', f'j * h{xd_var[0]}_')

        # substituir as derivadas parciais pelas diferenças finitas centradas
        elif (method == "central"):
            for j in range(len(eqrs)):
                for k in range(len(str_sp_vars)):
                    if k == 0:
                        for i in range(len(xd_var)):
                            eqrs[j] = eqrs[j].replace(f'd2{xd_var[i]}{str_sp_vars}/d{str_sp_vars[k]}2', f'({xd_var[i]}_i+1_j - 2*{xd_var[i]}_ii_j + {xd_var[i]}_i-1_j)/ h{xd_var[i]}_ ** 2')
                            eqrs[j] = eqrs[j].replace(f'd{xd_var[i]}{str_sp_vars}/d{str_sp_vars[k]}', f'({xd_var[i]}_i+1_j - {xd_var[i]}_i-1_j)/(2* h{xd_var[i]}_)')
                    elif k == 1:
                        for i in range(len(xd_var)):
                            eqrs[j] = eqrs[j].replace(f'd2{xd_var[i]}{str_sp_vars}/d{str_sp_vars[k]}2', f'({xd_var[i]}_ii_j+1 - 2*{xd_var[i]}_ii_j + {xd_var[i]}_ii_j-1)/ h{xd_var[i]}_ ** 2')
                            eqrs[j] = eqrs[j].replace(f'd{xd_var[i]}{str_sp_vars}/d{str_sp_vars[k]}', f'({xd_var[i]}_ii_j+1 - {xd_var[i]}_ii_j-1)/(2* h{xd_var[i]}_)')
                for i in range(len(xd_var)):
                    eqrs[j] = eqrs[j].replace(f'{xd_var[i]}{str_sp_vars}', f'{xd_var[i]}_ii_j')
                
            for j in range(len(eqrs)):
                eqrs[j] = _repl_symbol(eqrs[j], f'{str_sp_vars[0]}', f'ii * h{xd_var[0]}_')
                if len(str_sp_vars) == 2:
                    eqrs[j] = _repl_symbol(eqrs[j], f'{str_sp_vars[1]}', f'j * h{xd_var[0]}_')

        # substituir as derivadas parciais pelas diferenças finitas atrasadas
        elif (method == "backward"):
            for j in range(len(eqrs)): # loop para cada equação
                for k in range(len(str_sp_vars)): # loop para cada variável espacial
                    if k == 0:
                        for i in range(len(xd_var)): # loop para cada função
                            eqrs[j] = eqrs[j].replace(f'd2{xd_var[i]}{str_sp_vars}/d{str_sp_vars[k]}2', f'({xd_var[i]}_i+1_j - 2*{xd_var[i]}_ii_j + {xd_var[i]}_i-1_j)/ h{xd_var[i]}_ ** 2')
                            eqrs[j] = eqrs[j].replace(f'd{xd_var[i]}{str_sp_vars}/d{str_sp_vars[k]}', f'({xd_var[i]}_ii_j - {xd_var[i]}_i-1_j)/ h{xd_var[i]}_')
                    elif k == 1:
                        for i in range(len(xd_var)):
                            eqrs[j] = eqrs[j].replace(f'd2{xd_var[i]}{str_sp_vars}/d{str_sp_vars[k]}2', f'({xd_var[i]}_ii_j+1 - 2*{xd_var[i]}_ii_j + {xd_var[i]}_ii_j-1)/ h{xd_var[i]}_ ** 2')
                            eqrs[j] = eqrs[j].replace(f'd{xd_var[i]}{str_sp_vars}/d{str_sp_vars[k]}', f'({xd_var[i]}_ii_j - {xd_var[i]}_ii_j-1)/ h{xd_var[i]}_')
                for i in range(len(xd_var)):
                        eqrs[j] = eqrs[j].replace(f'{xd_var[i]}{str_sp_vars}', f'{xd_var[i]}_ii_j')
                
            for j in range(len(eqrs)):
                eqrs[j] = _repl_symbol(eqrs[j], f'{str_sp_vars[0]}', f'ii * h{xd_var[0]}_')
                if len(str_sp_vars) == 2:
                    eqrs[j] = _repl_symbol(eqrs[j], f'{str_sp_vars[1]}', f'j * h{xd_var[0]}_')
        else:
            raise ValueError("Método de discretização inválido. Use 'forward', 'central' ou 'backward'.")

        partial_list_eq = []
        for j in range(len(eqrs)):
            partial_list_eq.append([eqrs[j].replace('i+1', str(i+1)).replace('i-1', str(i-1)).replace('i-2', str(i-2)).replace('i+2', str(i+2)).replace('ii',str(i)) for i in range(1,n_part[0]-1)])

        list_eq = [[] for i in range(len(partial_list_eq))]

        if len(str_sp_vars) == 2:
            for j in range(len(partial_list_eq)): # loop nas equações
                for i in range(len(partial_list_eq[j])): # loop nos pontos da malha na direção x
                    for k in range(1,n_part[1]-1): # loop nos pontos da malha na direção y
                        list_eq[j].append(partial_list_eq[j][i].replace('j+1', str(k+1)).replace('j-1', str(k-1)).replace('j-2', str(k-2)).replace('j+2', str(k+2)).replace('j',str(k)))
        elif len(str_sp_vars) == 1:
            for j in range(len(partial_list_eq)):
                for i in range(len(partial_list_eq[j])):
                    list_eq[j].append(partial_list_eq[j][i].replace('j', str(0)))

        list_positions = []
        # loop para definir as posições dos pontos de contorno
        if len(str_sp_vars) == 2:
            for func in range(len(pdes.funcs)):
                list_aux = []
                for i in range(n_part[0]):
                    for j in range(n_part[1]):
                        if i == 0:                 list_aux.append(f'W{func}_{i}_{j}')
                        elif i == n_part[0]-1:     list_aux.append(f'E{func}_{i}_{j}')
                        elif j == 0:               list_aux.append(f'S{func}_{i}_{j}')
                        elif j == n_part[1]-1:     list_aux.append(f'N{func}_{i}_{j}')
                        else:                      list_aux.append(f'Ce{func}_{i}_{j}')
                list_positions.append(list_aux)
        # loop para definir os pontos de contorno 1D
        elif len(str_sp_vars) == 1:
            for func in range(len(pdes.funcs)):
                list_aux = []
                for i in range(0,n_part[0]):
                    if i == 0:
                        if west_bd.lower() == 'dirichlet': list_aux.append(dirichlet(west_func_bd, list_eq, 'west', n_part, xd_var, str_sp_vars)[func][0])
                        elif west_bd.lower() == 'neumann': list_aux.append(neumann(west_func_bd, list_eq, 'west', n_part, xd_var, str_sp_vars)[func][0])
                    elif i == n_part[0]-1:      
                        if east_bd.lower() == 'dirichlet': list_aux.append(dirichlet(east_func_bd, list_eq, 'east', n_part, xd_var, str_sp_vars)[func][0])
                        elif east_bd.lower() == 'neumann': list_aux.append(neumann(east_func_bd, list_eq, 'east', n_part, xd_var, str_sp_vars)[func][0])
                        elif east_bd.lower() == 'robin': list_aux.append(robin(east_func_bd, list_eq, 'east', east_alpha_bd, east_beta_bd, n_part, xd_var, str_sp_vars)[func][0])
                    else: 
                        list_aux.append(f'Ce{func}{i}0')
                list_positions.append(list_aux)
        # definir os valores dos pontos de contorno
        if len(str_sp_vars) == 2:
            for func in range(len(list_positions)):
                C = 0
                for i in range(len(list_positions[func])):
                    if 'C' in list_positions[func][i]:
                        list_positions[func][i] = list_eq[func][C]; C+=1

            if south_bd.lower() == 'dirichlet':
                list_south = dirichlet(south_func_bd, list_eq, 'south', n_part, xd_var, str_sp_vars, use_time_derivative=True)
            elif south_bd.lower() == 'neumann':
                list_south = neumann(south_func_bd, list_eq, 'south', n_part, xd_var, str_sp_vars)
            elif south_bd.lower() == 'robin':
                list_south = robin(south_func_bd, list_eq, 'south', south_alpha_bd, south_beta_bd, n_part, xd_var)

            if north_bd.lower() == 'dirichlet':
                list_north = dirichlet(north_func_bd, list_eq, 'north', n_part, xd_var, str_sp_vars, use_time_derivative=True)
            elif north_bd.lower() == 'neumann':
                list_north = neumann(north_func_bd, list_eq, 'north', n_part, xd_var, str_sp_vars)
            elif north_bd.lower() == 'robin':
                list_north = robin(north_func_bd, list_eq, 'north', north_alpha_bd, north_beta_bd, n_part, xd_var)

            if west_bd.lower() == 'dirichlet':
                list_west = dirichlet(west_func_bd, list_eq, 'west', n_part, xd_var, str_sp_vars, use_time_derivative=True)
            elif west_bd.lower() == 'neumann':
                list_west = neumann(west_func_bd, list_eq, 'west', n_part, xd_var, str_sp_vars)
            else:
                list_west = [[] for _ in range(len(list_eq))]

            list_east = [[] for i in range(len(list_eq))]
            # preencher os valores do contorno leste
            for func in range(len(list_eq)):
                list_east[func].append(list_south[func][-1])
                if east_bd.lower() == 'dirichlet':
                    centro = dirichlet(east_func_bd, list_eq, 'east', n_part, xd_var, str_sp_vars, use_time_derivative=True)[func]
                elif east_bd.lower() == 'neumann':
                    centro = neumann(east_func_bd,list_eq, 'east', n_part, xd_var, str_sp_vars)[func]
                elif east_bd.lower() == 'robin':
                    centro = robin(east_func_bd, list_eq, 'east', east_alpha_bd, east_beta_bd, n_part, xd_var)[func]
                else:
                    centro = []
                for i in range(len(centro)):
                    list_east[func].append(centro[i])
                list_east[func].append(list_north[func][-1])
        # atribuir os valores corretos aos pontos de contorno 1D
        elif len(str_sp_vars) == 1:
            for func in range(len(list_positions)):
                C = 0
                for i in range(len(list_positions[func])):
                    if 'C' in list_positions[func][i]: list_positions[func][i] = list_eq[func][C]; C+=1
                    
        # substituir os pontos de contorno pelos valores corretos
        if len(str_sp_vars) == 2:
            for func in range(len(list_positions)):
                S = 0; N = 0; E = 0; W = 0
                for len_list in range(len(list_positions[func])):
                    if   "S" in list_positions[func][len_list]: list_positions[func][len_list] = list_south[func][S]; S += 1
                    elif "N" in list_positions[func][len_list]: list_positions[func][len_list] = list_north[func][N]; N += 1
                    elif "E" in list_positions[func][len_list]: list_positions[func][len_list] = list_east[func][E];   E += 1
                    elif "W" in list_positions[func][len_list]: list_positions[func][len_list] = list_west[func][W];   W += 1

        d_vars = []
        if len(str_sp_vars) == 2:
            for func in range(len(list_positions)):
                for i in range(0,n_part[0]):
                    for j in range(0,n_part[1]):
                        if not(f'XX{func}_{i}_{j}' in d_vars):
                            d_vars.append(f'XX{func}_{i}_{j}')
        elif len(str_sp_vars) == 1:
            for func in range(len(list_positions)):
                for i in range(n_part[0]):
                    name = f'XX{func}_{i}_0'
                    if name not in d_vars:
                        d_vars.append(name)

        flat_list_positions = []
        # flatten list_positions
        for L in list_positions:
            flat_list_positions.extend(L)

       # passos corretos: 1/(n-1)
        hx_val = str(1.0 / (n_part[0] - 1))
        # Nota: O seu discretizador gera a mesma string de passo para x e y na mesma função
        
        for i in range(len(flat_list_positions)): 
            # Faz o loop por todas as variáveis (XX0, XX1, etc) para limpar todos os 'h'
            for func_name in xd_var:
                flat_list_positions[i] = flat_list_positions[i].replace(f'h{func_name}_', hx_val)

        return flat_list_positions, d_vars


def dirichlet(bd_func, list_eq, bd, n_part, xd_var, str_sp_vars = '', use_time_derivative=True):
    def maybe_dt(s: str) -> str:
        return d_dt(s) if use_time_derivative else s

    def replace_xy(expr: str, X: str, Y: str) -> str:
        out = _repl_symbol(expr, str_sp_vars[0], X)  # x
        if len(str_sp_vars) == 2:
            out = _repl_symbol(out,  str_sp_vars[1], Y)  # y
        return out
    if len(str_sp_vars) == 2:
        Nx, Ny = n_part[0], n_part[1]
        hx = f'h{xd_var[0]}_'
        hy = f'h{xd_var[0]}_'
        
            
        if bd.lower() == 'north':
            # y = (Ny-1)*hy, x = i*hx, i = 0..Nx-1
            list_north = [[] for _ in range(len(list_eq))]
            for func in range(len(list_eq)):
                for i in range(Nx):
                    expr = replace_xy(bd_func, f'{i} * {hx}', f'{Ny-1} * {hy}')
                    list_north[func].append(maybe_dt(expr))
            return list_north

        elif bd.lower() == 'south':
            # y = 0, x = i*hx
            list_south = [[] for _ in range(len(list_eq))]
            for func in range(len(list_eq)):
                for i in range(Nx):
                    expr = replace_xy(bd_func, f'{i} * {hx}', f'0 * {hy}')
                    list_south[func].append(maybe_dt(expr))
            return list_south

        elif bd.lower() == 'east':
            # x = (Nx-1)*hx, y = j*hy, j = 0..Ny-1
            list_east = [[] for _ in range(len(list_eq))]
            for func in range(len(list_eq)):
                for j in range(Ny):
                    expr = replace_xy(bd_func, f'{Nx-1} * {hx}', f'{j} * {hy}')
                    list_east[func].append(maybe_dt(expr))
            return list_east

        elif bd.lower() == 'west':
            # x = 0, y = j*hy
            list_west = [[] for _ in range(len(list_eq))]
            for func in range(len(list_eq)):
                for j in range(Ny):
                    expr = replace_xy(bd_func, f'0 * {hx}', f'{j} * {hy}')
                    list_west[func].append(maybe_dt(expr))
            return list_west

        else:
            print("Invalid boundary. Try 'east', 'north' or 'south' ")
            return []
    elif len(str_sp_vars) == 1:
        Nx = n_part[0]
        hx = f'h{xd_var[0]}_'
        if bd.lower() == 'west':
            list_west = [[] for _ in range(len(list_eq))]
            for func in range(len(list_eq)):
                for i in range(Nx):
                    expr = replace_xy(bd_func, f'{0} * {hx}', '')
                    list_west[func].append(maybe_dt(expr))
            return list_west

        elif bd.lower() == 'east':
            list_east = [[] for _ in range(len(list_eq))]
            for func in range(len(list_eq)):
                for i in range(Nx):
                    expr = replace_xy(bd_func, f'{Nx-1} * {hx}', '')
                    list_east[func].append(maybe_dt(expr))
            return list_east

        else:
            print("Invalid boundary. Try 'east' or 'west' for 1D problems (without south/north). ")
            return []

def robin(bd_func, list_eq, bd, alpha, beta, n_part, xd_var, str_sp_vars = ''):
    if len(str_sp_vars) == 2:
        # Deixado como antes (strings), pois Robin não foi o caso que disparou o erro.
        if bd.lower() == 'north':
            list_north = [[] for i in range(len(list_eq))]
            for func in range(len(list_eq)):
                for j in range(0, len(list_eq[func])):
                    if j % (n_part[1]-2) == 0:
                        list_north[func].append(f'(h{xd_var[func]}_*{bd_func}-{list_eq[func][j+n_part[1]-3]}*({alpha}*h{xd_var[func]}_-{beta}))/{beta}')
            return list_north

        elif bd.lower() == 'south':
            list_south = [[] for i in range(len(list_eq))]
            for func in range(len(list_eq)):
                for j in range(0, len(list_eq[func])):
                    if j % (n_part[1]-2) == 0:
                        list_south[func].append(f'(h{xd_var[func]}_*{bd_func}-{list_eq[func][j]}*({alpha}*h{xd_var[func]}_-{beta}))/{beta}')
            return list_south

        elif bd.lower() == 'east':
            list_east = [[] for i in range(len(list_eq))]
            for func in range(len(list_eq)):
                for i in range((n_part[1]-2)*(n_part[0]-2)-(n_part[1]-2), (n_part[1]-2)*(n_part[0]-2)):
                    list_east[func].append(f'(h{xd_var[func]}_*{bd_func}-{list_eq[func][i]}*({alpha}*h{xd_var[func]}_-{beta}))/{beta}')
            return list_east

        else:
            print("Invalid boundary. Try 'east', 'north' or 'south' ")
            return []
    elif len(str_sp_vars) == 1:
        if bd.lower() == 'west':
            list_west = [[] for i in range(len(list_eq))]
            for func in range(len(list_eq)):
                for i in range(0, len(list_eq[func])):
                    list_west[func].append(f'(h{xd_var[func]}_*{bd_func}-{list_eq[func][i]}*({alpha}*h{xd_var[func]}_-{beta}))/{beta}')
            return list_west

        elif bd.lower() == 'east':
            list_east = [[] for i in range(len(list_eq))]
            for func in range(len(list_eq)):
                for i in range(0, len(list_eq[func])):
                    list_east[func].append(f'(h{xd_var[func]}_*{bd_func}-{list_eq[func][i]}*({alpha}*h{xd_var[func]}_-{beta}))/{beta}')
            return list_east

        else:
            print("Invalid boundary. Try 'east' or 'west' for 1D problems (without south/north). ")
            return []

def neumann(bd_func, list_eq, bd, n_part, xd_var, str_sp_vars = ''):
    if len(str_sp_vars) == 2:
        if bd.lower() == 'north':
            list_north = [[] for i in range(len(list_eq))]
            for func in range(len(list_eq)):
                for j in range(0, len(list_eq[func])):
                    if j % (n_part[1]-2) == 0:
                        list_north[func].append(f'h{xd_var[0]}_*{bd_func}+{list_eq[func][j+n_part[1]-3]}')
            return list_north

        elif bd.lower() == 'south':
            list_south = [[] for i in range(len(list_eq))]
            for func in range(len(list_eq)):
                for j in range(0, len(list_eq[func])):
                    if j % (n_part[1]-2) == 0:
                        list_south[func].append(f'h{xd_var[func]}_*{bd_func}+{list_eq[func][j]}')
            return list_south

        elif bd.lower() == 'east':
            list_east = [[] for i in range(len(list_eq))]
            for func in range(len(list_eq)):
                for i in range((n_part[1]-2)*(n_part[0]-2)-(n_part[1]-2), (n_part[1]-2)*(n_part[0]-2)):
                    list_east[func].append(f'h{xd_var[func]}_*{bd_func}+{list_eq[func][i]}')
            return list_east

        else:
            print("Invalid boundary. Try 'east', 'north' or 'south' ")
            return []
    elif len(str_sp_vars) == 1:
        if bd.lower() == 'west':
            list_west = [[] for i in range(len(list_eq))]
            for func in range(len(list_eq)):
                for i in range(0, len(list_eq[func])):
                    list_west[func].append(f'h{xd_var[func]}_*{bd_func}+{list_eq[func][i]}')
            return list_west

        elif bd.lower() == 'east':
            list_east = [[] for i in range(len(list_eq))]
            for func in range(len(list_eq)):
                for i in range(0, len(list_eq[func])):
                    list_east[func].append(f'h{xd_var[func]}_*{bd_func}+{list_eq[func][i]}')
            return list_east

        else:
            print("Invalid boundary. Try 'east' or 'west' for 1D problems (without south/north). ")
            return []
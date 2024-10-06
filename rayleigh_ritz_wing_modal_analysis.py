# -*- coding: utf-8 -*-
"""
Análise modal de uma asa simplificada pelo método de Rayleigh-Ritz.
"""

import numpy as np
import scipy
from sympy import symbols, diff, integrate, lambdify
from sympy.solvers.solveset import linear_eq_to_matrix
import matplotlib.pyplot as plt

# %% INPUTS
# geometria da asa
b = 0.325  # semi-envergadura da asa [m]
c = 44e-3  # corda da asa [m]
t = 0.81e-3  # espessura da asa [m]
Ar = c * t  # área da seção da asa [m²]
I = c * t**3 / 12  # segundo momento de área da seção da asa [m^4]

# propriedades do material
E = 68.2e9  # módulo de elasticidade [Pa]
rho = 2800  # densidade [kg/m^3]

# propriedades do lastro
# m = 0  # massa [kg]
m = 0.03458  # massa [kg]
s = b  # posição da massa na envergadura [m]

# %% variáveis simbólicas
n_eq = 11  # número de equações
y = symbols('y')
C = symbols(f'c1:{n_eq + 1}')  # c1, c2, ..., c11

# Criando as equações polinomiais
pol = [y**(i + 2) for i in range(n_eq)]

# Definindo W e Wxx
W = sum(C[i] * pol[i] for i in range(n_eq))
Wxx = diff(diff(W, y), y)

# %% Calculando X
X = integrate(E * I * (Wxx**2), (y, 0, b))

# Criando a matriz A a partir das derivadas de X em relação a cada C[i]
Xeqns = [diff(X, C[i]) for i in range(n_eq)]

# Utilizando a função linear_eq_to_matrix para obter a matriz A
A_sym, _ = linear_eq_to_matrix(Xeqns, C)

# Convertendo A para valores numéricos
A = np.array(A_sym).astype(np.float64)

# %% Calculando Y
Y = integrate(rho * Ar * (W**2), (y, 0, b)) + 1 / 2 * m * W.subs(y, s)**2

# Criando a matriz B a partir das derivadas de Y em relação a cada C[i]
Yeqns = [diff(Y, C[i]) for i in range(n_eq)]

# Utilizando linear_eq_to_matrix para obter a matriz B
B_sym, _ = linear_eq_to_matrix(Yeqns, C)

# Convertendo B para valores numéricos
B = np.array(B_sym).astype(np.float64)

# Calculando os autovalores e autovetores
w2, auto_vet = scipy.linalg.eig(A, B)

# %% frequências naturais
w = np.sqrt(np.abs(w2))  # [rad/s]
w_hz = w / (2 * np.pi)  # [Hz]

# reordenando os autovalores e autovetores
w_hz = np.flip(w_hz, axis=None)
auto_vet = auto_vet[:, ::-1]

# removendo valores inf no caso com lastro
if m != 0:
    w_hz = w_hz[2:-1]
    auto_vet = auto_vet[:, 2:-1]

# print das frequências naturais
print(w_hz)

# %% PLOTS

x = np.linspace(0, b, 100)

for i in range(4):
    # Substituindo y por x nas funções polinomiais multiplicadas pelos autovetores
    vib_mode_expr = sum(pol[j] * auto_vet[j, i] for j in range(n_eq))

    # Convertendo a expressão simbólica em uma função numérica (lambdify)
    vib_mode_func = lambdify(y, vib_mode_expr, 'numpy')

    # Avaliando a função em x
    vib_mode = vib_mode_func(x)

    # Normalizando os modos de vibração
    vib_mode = vib_mode / np.max(np.abs(vib_mode))

    plt.subplot(4, 1, i + 1)
    plt.plot(x, vib_mode, color='k')
    plt.xlim(0, b)
    plt.ylim(-1, 1)
    plt.title(f'Modo {i + 1}: {w_hz[i]:.2f} Hz')

    plt.ylabel('z')
    plt.grid(True)

plt.xlabel('y [m]')
plt.tight_layout()
plt.show()

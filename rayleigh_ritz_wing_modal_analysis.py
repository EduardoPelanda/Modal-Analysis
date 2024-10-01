import numpy as np
import scipy
from sympy import symbols, diff, integrate, Matrix
from sympy.solvers.solveset import linear_eq_to_matrix
import matplotlib.pyplot as plt
from sympy import lambdify

# Propriedades
b = 44e-3  # base [m]
h = 0.81e-3  # altura [m]
E = 68.2e9  # Módulo de Elasticidade [Pa]
I = b * h**3 / 12  # Momento de Inércia de Área [m^4]
l = 0.325
m = 0.03458  # massa
s = 0.325  # posição da massa m
p = 2800  # densidade [kg/m^3]
Ar = b * h  # Área da seção transversal

# Definindo as variáveis simbólicas
n_equacoes = 11
y = symbols('y')
c = symbols(f'c1:{n_equacoes + 1}')  # Variáveis simbólicas c1, c2, ..., c11

# Criando as equações polinomiais (ignorando graus 0 e 1, assim começa em y^2)
pol = [y**(i + 2) for i in range(n_equacoes)]  # No MATLAB i=1:n, aqui i=0:n-1

# Definindo W e Wxx
W = sum(c[i] * pol[i] for i in range(n_equacoes))  # Indexação de 0 a n_equacoes-1
Wxx = diff(diff(W, y), y)

# Achando X (integral de E*I*(Wxx^2) em y de 0 a l)
X = integrate(E * I * (Wxx**2), (y, 0, l))

# Criando a matriz A a partir das derivadas de X em relação a cada c[i]
Xeqns = [diff(X, c[i]) for i in range(n_equacoes)]  # No MATLAB i=1:n, aqui i=0:n-1

# Utilizando a função linear_eq_to_matrix para obter a matriz A
A_sym, _ = linear_eq_to_matrix(Xeqns, c)

# Convertendo A para valores numéricos (como exemplo)
A = np.array(A_sym).astype(np.float64)

# Achando Y (integral de p*Ar*(W^2) em y de 0 a l e m*W^2/2 em y = s)
Y = integrate(p * Ar * (W**2), (y, 0, l)) + (1 / 2) * m * W.subs(y, s)**2

# Criando a matriz B a partir das derivadas de Y em relação a cada c[i]
Yeqns = [diff(Y, c[i]) for i in range(n_equacoes)]  # No MATLAB i=1:n, aqui i=0:n-1

# Utilizando linear_eq_to_matrix para obter a matriz B
B_sym, _ = linear_eq_to_matrix(Yeqns, c)

# Convertendo B para valores numéricos
B = np.array(B_sym).astype(np.float64)

# Encontrando os autovalores e autovetores
w2, auto_vet = scipy.linalg.eig(A, B)
# w2, auto_vet = np.linalg.eig(np.linalg.inv(B) @ A)

# Calculando as frequências naturais (rad/s) e convertendo para Hz
w = np.sqrt(np.abs(w2))
w_hz = w / (2 * np.pi)
w_hz = np.flip(w_hz, axis=None)
auto_vet = auto_vet[:, ::-1]

if m != 0:
    w_hz = w_hz[2:-1]
    auto_vet = auto_vet[:, 2:-1]

# Resultado das frequências naturais e autovetores
print(w_hz)

x = np.linspace(0, l, 100)

# Loop para plotar os 4 primeiros modos de vibração
for i in range(4):
    # Substituindo y por x nas funções polinomiais multiplicadas pelos autovetores
    vib_mode_expr = sum(pol[j] * auto_vet[j, i] for j in range(n_equacoes))

    # Convertendo a expressão simbólica em uma função numérica (lambdify)
    vib_mode_func = lambdify(y, vib_mode_expr, 'numpy')

    # Avaliando a função em x
    vib_mode = vib_mode_func(x)

    # Normalizando os modos de vibração
    vib_mode = vib_mode / np.max(np.abs(vib_mode))

    # Criando o gráfico
    plt.subplot(4, 1, i + 1)
    plt.plot(x, vib_mode, color='k')
    plt.ylim(-1, 1)
    plt.title(f'Modo {i + 1}: {w_hz[i]:.2f} Hz')

    plt.ylabel('z')
    plt.grid(True)

# Exibindo todos os gráficos
plt.xlabel('y [m]')
plt.tight_layout()
plt.show()

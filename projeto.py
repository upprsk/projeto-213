from scipy.io import loadmat
from numpy.typing import NDArray
import numpy as np
import control as cnt
import matplotlib.pyplot as plt

f64 = np.float64


def sundaresan(
    step: f64,
    time: NDArray[f64],
    output: NDArray[f64],
) -> tuple[f64, f64, f64]:
    """Calculate k, tau and theta for the given data using sundaresan."""
    valor_inicial: f64 = output[0]
    output = output - valor_inicial

    valor_final: f64 = output[-1]
    k = valor_final / step

    t1 = f64(0)
    t2 = f64(0)
    for i in range(1, len(output)):
        if t1 == 0 and output[i] >= 0.353 * valor_final:
            t1 = time[i]

        if output[i] >= 0.853 * valor_final:
            t2 = time[i]
            break

    tau = 2.0 / 3 * (t2 - t1)
    theta = 1.3 * t1 - 0.29 * t2

    return k, tau, theta


def smith(step: f64, time: NDArray[f64], output: NDArray[f64]) -> tuple[f64, f64, f64]:
    """Calculate k, tau and theta for the given data using smith."""
    valor_inicial: f64 = output[0]
    output = output - valor_inicial

    valor_final: f64 = output[-1]
    k = valor_final / step

    t1 = time[output >= 0.283 * valor_final][0]
    t2 = time[output >= 0.6321 * valor_final][0]

    tau = 1.5 * (t2 - t1)
    theta = t2 - tau

    return k, tau, theta


def calc_mse(output: NDArray[f64], expected: NDArray[f64]) -> f64:
    """Calculate mean-square-error"""
    return ((expected - output) ** 2).sum() / len(saida)


def tf(k: f64, tau: f64, theta: f64, n_pade=20) -> cnt.TransferFunction:
    sys_pade = cnt.tf(*cnt.pade(theta, n_pade))
    sys = cnt.tf([k], [tau, 1])

    return cnt.series(sys, sys_pade)


def PID(kp: f64, ti: f64, td: f64):
    h_kp = cnt.tf([kp], [1])
    h_ki = cnt.tf([kp], [ti, 0])
    h_kd = cnt.tf([kp * td, 0], [1])

    return cnt.parallel(h_kp, h_ki, h_kd)


# 1. Identifique o conjunto de dados para seu grupo, nomeado por 'Dataset GrupoX'.
dataset = loadmat("./datasets/Dataset_Grupo5.mat")

tempo = dataset["TARGET_DATA____ProjetoC213_Degrau"][:, 0]
degrau = dataset["TARGET_DATA____ProjetoC213_Degrau"][:, 1]
saida = dataset["TARGET_DATA____ProjetoC213_Saida"][:, 1]

plt.plot(tempo, degrau, tempo, saida)
plt.grid()
plt.legend(["degrau", "saida"])
plt.title("Dataset")
plt.figure()

# 2. Escolha o Método de Identificação da Planta - Smith ou Sundaresan, e
#    determine os valores de k, θ e τ para levantar a Função de Transferência do
#    modelo de acordo com a resposta tı́pica. Justifique a escolha do método e do
#    modelo.
amplitude_degrau: f64 = degrau[-1]
valor_inicial: f64 = saida[0]
k, tau, theta = sundaresan(amplitude_degrau, tempo, saida)
print(f"{k=}, {tau=}, {theta=}")

sys = tf(k, tau, theta)
t, y = cnt.step_response(sys, np.linspace(0, tempo[-1], len(tempo)))
assert t is not None
assert y is not None

output = y * amplitude_degrau + valor_inicial
mse = ((saida - output) ** 2).sum() / len(saida)
print(f"mse sundaresan={mse}")

plt.plot(t, output, tempo, degrau, tempo, saida)
plt.legend(["output", "degrau", "saida"])
plt.grid()
plt.title(f"Sundaresan ({mse=})")
plt.figure()

# ---

amplitude_degrau: f64 = degrau[-1]
valor_inicial: f64 = saida[0]
k, tau, theta = smith(amplitude_degrau, tempo, saida)
print(f"{k=}, {tau=}, {theta=}")

sys = tf(k, tau, theta)
t, y = cnt.step_response(sys, np.linspace(0, tempo[-1], len(tempo)))
assert t is not None
assert y is not None

output = y * amplitude_degrau + valor_inicial
mse = ((saida - output) ** 2).sum() / len(saida)
print(f"mse smith={mse}")

plt.plot(t, output, tempo, degrau, tempo, saida)
plt.legend(["output", "degrau", "saida"])
plt.grid()
plt.title(f"Smith ({mse=})")
plt.figure()

# 3. Compare a resposta original em relação à estimada e verifique se a
#    aproximação foi satisfatória. Se necessário, realize o ajuste fino dos
#    parâmetros, expondo o reflexo das alterações na resposta do sistema.
#
# > Será usado o método de smith, ja que visualmente aparenta ser mais próximo e
# > também seu MSE é menor. Erro de 2.2389e-06 (smith) vs 8.35297e-05 (sundaresan).

# 4. Plote as respostas do Sistema em Malha Aberta e Fechada e comente sobre as
#    diferenças nos valores do Erro em regime permanente e no Tempo de Acomodação;

sys_closed = cnt.feedback(sys)
tc, yc = cnt.step_response(sys_closed, np.linspace(0, tempo[-1], len(tempo)))
assert tc is not None
assert yc is not None

outputc = yc * amplitude_degrau + valor_inicial
plt.plot(t, output, tc, outputc)
plt.legend(["open", "closed"])
plt.grid()
plt.title("open/closed")
plt.figure()

# 5. Sintonize um Controlador PID de acordo com os métodos especificados e
#    verifique o comportamento do sistema controlado. Para a Sintonia IMC, a
#    escolha de λ é livre de acordo com o critério de desempenho.

kp = (0.6 * tau) / (k * theta)
ti = tau
td = theta / 2

pid = PID(kp, ti, td)
sys_pid = cnt.feedback(cnt.series(sys, pid))

tc, yc = cnt.step_response(sys_pid, np.linspace(0, tempo[-1], len(tempo)))
assert tc is not None
assert yc is not None

outputc = yc * amplitude_degrau + valor_inicial
plt.plot(tc, outputc)
plt.legend(["response"])
plt.grid()
plt.title(f"PID {kp:.4f}, {ti:.4f}, {td:.4f}")
plt.figure()


# ---

plt.show()

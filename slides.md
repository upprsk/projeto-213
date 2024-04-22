[comment]: <> "LTeX: language=pt-BR"

Projeto PID
===

<!-- column_layout: [1, 3, 1] -->
<!-- column: 1 -->
<!-- jump_to_middle -->
Grupo 5
Lucas Ross - GEC - 1718
<!-- reset_layout -->
<!-- end_slide -->

0 - Bibliotecas
====

```python
import control as cnt
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat

f64 = np.float64
```

<!-- end_slide -->

1 - Carregar os dados
====

> Identifique o conjunto de dados para seu grupo, nomeado por 'Dataset GrupoX'.

```python
dataset = loadmat("./datasets/Dataset_Grupo5.mat")
tempo = dataset["TARGET_DATA____ProjetoC213_Degrau"][:, 0]
degrau = dataset["TARGET_DATA____ProjetoC213_Degrau"][:, 1]
saida = dataset["TARGET_DATA____ProjetoC213_Saida"][:, 1]
```

![Dataset](./dataset.png)

<!-- end_slide -->

2 - Identificação
====

> Escolha o Método de Identificação da Planta - Smith ou Sundaresan, e
> determine os valores de k, θ e τ para levantar a Função de Transferência do
> modelo de acordo com a resposta tı́pica. Justifique a escolha do método e do
> modelo.

Pare decidir qual possui a melhor aproximação, vai ser usado o RMSE (_Rising Mean Square Error_):

```python
def calc_rmse(output: NDArray[f64], expected: NDArray[f64]) -> f64:
    """Calculate mean-square-error"""
    return np.sqrt(((expected - output) ** 2).sum() / len(saida))
```

<!-- pause -->

---

```python
def tf(k: f64, tau: f64, theta: f64, n_pade=20) -> cnt.TransferFunction:
    sys_pade = cnt.tf(*cnt.pade(theta, n_pade))
    sys = cnt.tf([k], [tau, 1])

    return cnt.series(sys, sys_pade)
```

<!-- end_slide -->

Sundaresan
===
```python
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

    t1 = time[output >= 0.353 * valor_final][0]
    t2 = time[output >= 0.853 * valor_final][0]

    tau = 2.0 / 3 * (t2 - t1)
    theta = 1.3 * t1 - 0.29 * t2

    return k, tau, theta
```

<!-- end_slide -->

```python
amplitude_degrau: f64 = degrau[-1]
valor_inicial: f64 = saida[0]
k, tau, theta = sundaresan(amplitude_degrau, tempo, saida)

sys = tf(k, tau, theta)
t, y = cnt.step_response(sys, np.linspace(0, tempo[-1], len(tempo)))

output = y * amplitude_degrau + valor_inicial
mse = calc_rmse(output, saida)
```

![sundaresan](./sundaresan.png)

<!-- end_slide -->

Smith
===

```python
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
```

<!-- end_slide -->

```python
amplitude_degrau: f64 = degrau[-1]
valor_inicial: f64 = saida[0]
k, tau, theta = smith(amplitude_degrau, tempo, saida)

sys = tf(k, tau, theta)
t, y = cnt.step_response(sys, np.linspace(0, tempo[-1], len(tempo)))

output = y * amplitude_degrau + valor_inicial
mse = calc_rmse(output, saida)
```

![smith](./smith.png)

<!-- end_slide -->

Resultados Smith (🎉)
====

- **k**: 0.9572
- **tau**: 35.1
- **theta**: 13.1
- **RMSE**: 0.0015

![smith](./smith.png)

<!-- end_slide -->

3 - Conclusão
===

> Compare a resposta original em relação à estimada e verifique se a
> aproximação foi satisfatória. Se necessário, realize o ajuste fino dos
> parâmetros, expondo o reflexo das alterações na resposta do sistema.

Será usado o método de smith, ja que visualmente aparenta ser mais próximo e
também seu MSE é menor. Erro de 0.0015 (smith) vs 0.0091 (sundaresan).

<!-- end_slide -->

4 - Malha Aberta/Fechada
===

> Plote as respostas do Sistema em Malha Aberta e Fechada e comente sobre as
> diferenças nos valores do Erro em regime permanente e no Tempo de Acomodação;

```python
sys_closed = cnt.feedback(sys)
tc, yc = cnt.step_response(sys_closed, np.linspace(0, tempo[-1], len(tempo)))

outputc = yc * amplitude_degrau + valor_inicial
```

![Closed](./closed.png)

<!-- end_slide -->

4 - Sintonizar
===

> Sintonize um Controlador PID de acordo com os métodos especificados e
> verifique o comportamento do sistema controlado. Para a Sintonia IMC, a
> escolha de λ é livre de acordo com o critério de desempenho.

<!-- column_layout: [1, 3, 1] -->
<!-- column: 1 -->

|        | Métodos        | Conclusão              |
| ------ | -------------- | ---------------------- |
| Grupo 1| Técnicas 1 e 5 | (a) Tempo de resposta  |
| Grupo 2| Técnicas 2 e 3 | (b) Overshoot          |
| Grupo 3| Técnicas 2 e 6 | (a) Tempo de resposta  |
| Grupo 4| Técnicas 4 e 5 | (b) Overshoot          |
| Grupo 5| Técnicas 3 e 6 | (a) Tempo de resposta  |

<!-- reset_layout -->
<!-- pause -->

- CHR - Sem sobrevalor: O Método CHR estabelece dois critérios de desempenho para
  o sistema considerando o problema servo (mudança de valor do Setpoint):
  resposta mais rápida sem overshoot ou resposta mais rápida com 20% de overshoot.
- ITAE: O ITAE é um Índice de Desempenho baseado na minimização do erro. Um
  Sistema com Controle ITAE reduz grandes erros iniciais, assim como erros
  posteriores ocorridos na resposta transitória, tornando as oscilações bem
  amortecidas e reduzindo consideravelmente o overshoot

<!-- end_slide -->

Helpers
====

```python
def PID(kp: f64, ti: f64, td: f64):
    h_kp = cnt.tf([kp], [1])
    h_ki = cnt.tf([kp], [ti, 0])
    h_kd = cnt.tf([kp * td, 0], [1])

    return cnt.parallel(h_kp, h_ki, h_kd)

def calc_settling_time(t: NDArray[f64], y: NDArray[f64]):
    """Segundo o criterio dos 2%"""
    vf = y[-1]
    yr = y[::-1]

    return t[::-1][np.argmax((yr <= vf * 0.98) | (yr >= vf * 1.02))]

def calc_response_time(t: NDArray[f64], y: NDArray[f64], theta: f64):
    vf = y[-1]

    p90_idx = np.argmax(y > vf * 0.90)
    return t[p90_idx] - theta
```

<!-- end_slide -->

CHR - Sem sobrevalor
===

```python
kp = (0.6 * tau) / (k * theta)
ti = tau
td = theta / 2

pid = PID(kp, ti, td)
sys_pid = cnt.feedback(cnt.series(sys, pid))

tc, yc = cnt.step_response(sys_pid, np.linspace(0, tempo[-1], len(tempo)))

overshoot = (np.max(yc) - yc[-1]) / yc[-1]
settling_time = calc_settling_time(tc, yc)
response_time = calc_response_time(tc, yc, theta)

outputc = yc * amplitude_degrau + valor_inicial
```

![CHR](chr.png)

<!-- end_slide -->

ITAE
===

```python
A = 0.965; B = -0.85; C = 0.796; D = -0.147; E = 0.308; F = 0.929

kp = (A / k) * (theta / tau) ** B
ti = tau / (C + D * (theta / tau))
td = tau * E * (theta / tau) ** F

pid = PID(kp, ti, td)
sys_pid = cnt.feedback(cnt.series(sys, pid))

tc, yc = cnt.step_response(sys_pid, np.linspace(0, tempo[-1], len(tempo)))

overshoot = (np.max(yc) - yc[-1]) / yc[-1]
settling_time = calc_settling_time(tc, yc)
response_time = calc_response_time(tc, yc, theta)

outputc = yc * amplitude_degrau + valor_inicial
```

![ITAE](itae.png)

<!-- end_slide -->

6 - Explicação
===

> De acordo com a conclusão especificada, explique, entre os sistemas
> controlados, qual apresenta: (a) tempo de resposta mais rápido ou (b)
> menores ı́ndices de overshoot.

Nesse caso, o ITAE possui o tempo de resposta mais rápido e também o menor
índice de overshoot.

<!-- column_layout: [1, 1] -->
<!-- column: 0 -->
![CHR](chr.png)
<!-- column: 1 -->
![ITAE](itae.png)
<!-- reset_layout -->

<!-- end_slide -->
7 - UI
===

```python
def ui(kp: f64, ti: f64, td: f64) -> None:
    from rich.console import Console

    c = Console()

    def get_float(prompt: str) -> f64 | None:
        while True:
            try:
                return f64(input(prompt))
            except ValueError as e:
                c.print("error:", e)
            except EOFError:
                return

    while True:
        c.print(f"PID: ({kp:.4f}, {ti:.4f}, {td:.4f})")
        c.print("tweeker:")
        c.print("   1. Change P")
        c.print("   2. Change I")
        c.print("   3. Change D")
        c.print("   5. Run")
        c.print("[bright_black]Ctr+D to exit")
        ...
```

<!-- end_slide -->

```python
        opt = get_float("> ")
        if opt is None:
            break

        if opt not in (1, 2, 3, 4, 5):
            c.print(f"[red]Invalid option: {opt}")
            continue

        if opt == 1:
            val = get_float("P> ")
            if val is None:
                break

            kp = val
        elif opt == 2:
            val = get_float("I> ")
            if val is None:
                break

            ti = val
        elif opt == 3:
            val = get_float("D> ")
            if val is None:
                break

        elif opt == 5:
            ...
```

<!-- end_slide -->

```python
pid = PID(kp, ti, td)
sys_pid = cnt.feedback(cnt.series(sys, pid))

tc, yc = cnt.step_response(sys_pid, np.linspace(0, tempo[-1], len(tempo)))
assert tc is not None
assert yc is not None

overshoot = (np.max(yc) - yc[-1]) / yc[-1]
print(f"overshoot: {overshoot:.2%}")

settling_time = calc_settling_time(tc, yc)
print(f"settling time: {settling_time:.2f}s")

response_time = calc_response_time(tc, yc, theta)
print(f"response time: {response_time:.2f}s")

outputc = yc * amplitude_degrau + valor_inicial
plt.plot(tc, outputc)
plt.legend(["response"])
plt.grid()
plt.suptitle(f"PID P={kp:.4f} I={ti:.4f} D={td:.4f}")
plt.title(
    f"overshoot={overshoot*100:.2f}% settling={settling_time:.2f}s response={response_time:.2f}s"
)
plt.show()
```

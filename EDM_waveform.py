
from matplotlib import pyplot as plt
import random
from math import exp
import pandas as pd
from glob import glob


def gen_wave(h: float = 1.0, t_max: float = 2**10, t_on: float = 30.0, t_off: float = 40.0, v_n: float = 100.0,
             r: float = 1e6, c: float = 3e-3, tp: str = "t"):
    # h: step size (µs)
    # t_max: simulation time (µs)
    # t_on: discharge time (µs)
    # t_off: discharge interval (µs)
    # v_n: nominal voltage (V)
    # r: resistance of the RC circuit (Ω)
    # c: capacitance of the RC circuit (F)
    # tp: type of circuit ("t" for transistor, "rc" for RC circuit)
    n = int(t_max / h)
    time, signal = [0], [0]
    if tp.lower() == "t":
        i = 1
        while t_max - i*h >= t_on + 0.4*t_off:
            # ignition time follows gamma distribution (!)
            t_ig = random.gammavariate(5, 2)
            # 10% chance of short circuit (!)
            if random.random() > 0.98:
                v = 0
            else:
                v = v_n
            if random.random() > 0.2:
                v_var = random.gammavariate(0.5, 0.2)
                for j in range(int(t_ig / h)):
                    if i == n:
                        break
                    time.append(i*h)
                    signal.append((1 - v_var)*v + random.gammavariate(1, 0.2))
                    i += 1
                for j in range(int((t_on - t_ig) / h)):
                    if i == n:
                        break
                    time.append(i * h)
                    signal.append(0.4*v + random.gammavariate(1, 0.2))
                    i += 1
            else:
                for j in range(int(t_on / h)):
                    if i == n:
                        break
                    time.append(i * h)
                    signal.append(v + random.gammavariate(1, 0.2))
                    i += 1
            for j in range(int(t_off / h)):
                if i == n:
                    break
                time.append(i * h)
                signal.append(0)
                i += 1
    else:
        i = 1
        tau = 1 / (r*c)
        while i < n:
            if random.random() > 0.98:
                v = 0
            else:
                v = v_n
            v_var = random.gauss(0.79, .05)
            if v_var > 0.8:
                v_var = 0.8
            tau_var = random.gammavariate(4, 0.35)
            time_var = random.gauss(12, 0.05)
            for j in range(int(time_var*tau_var*tau*1e6 / h)):
                if i == n:
                    break
                time.append(i * h)
                signal.append(v_var*v*(1 - exp(-j*h*1e-6/(tau_var*tau))) + random.gammavariate(1, 0.3))
                i += 1
    return time, signal


tp = "rc"
# time, signal = gen_wave(tp=tp, c=2e-1)
# plt.plot(time, signal)
# plt.xlabel("Time (µs)")
# plt.ylabel("Voltage (V)")
# plt.title("Simulated waveform")
# plt.show()
#
# if input("Do you want to save this signal? (y/n): ").lower() == "y":
#     c = len(glob("./Data/Simulated data/"+tp+"*.csv"))
#     data = pd.Series(data=signal, index=time)
#     data.to_csv("./Data/Simulated data/"+tp+str(c+1)+".csv")

data_iter = 10
for i in range(data_iter):
    time, signal = gen_wave(tp=tp, c=2e-1)
    c = len(glob("./Data/Simulated data/" + tp + "*.csv"))
    data = pd.DataFrame(data={"timestamp": time, "target": signal})
    data.to_csv("./Data/Simulated data/" + tp + str(c + 1) + ".csv", index=False)

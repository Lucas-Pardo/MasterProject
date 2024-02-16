
import pywt
import pandas as pd
import matplotlib.pyplot as plt

signal = pd.read_csv("./Data/Simulated data/rc1.csv")
# print(len(signal["target"]))
level = 4
wlist = pywt.wavelist("sym", kind="discrete")
print(wlist)
for w in wlist:
    coef = pywt.wavedec(signal["target"], w, level=level, mode="constant")

    fig = plt.figure()
    fig.suptitle("Wavelet: " + w)
    ax = fig.add_subplot(3, 1, 1)
    ax.plot(signal["timestamp"], signal["target"])
    ax.set_title("Original signal")
    ax = fig.add_subplot(3, 1, 2)
    # print(len(coef[0][-len(signal["target"])//2**level:]), len(coef[1][-len(signal["target"])//2**level:]))
    ax.plot(coef[0][-len(signal["target"])//2**level:])
    ax.set_title("cA"+str(level))
    ax = fig.add_subplot(3, 1, 3)
    ax.plot(coef[0][-len(signal["target"])//2**level:] - coef[1][-len(signal["target"])//2**level:])
    ax.set_title("cA" + str(level) + " - cD" + str(level))


    # for i, a in enumerate(coef):
    #     ax = fig.add_subplot(len(coef), 1, i + 2)
    #     ax.plot(a)
    #     ax.set_title("Coefficient "+str(i+1))

    fig.tight_layout()
    plt.show()
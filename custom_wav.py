import pywt
import pandas as pd
import matplotlib.pyplot as plt

# signal = pd.read_csv("./Data/Simulated data/rc1.csv")
wlist = pywt.wavelist(kind="discrete")
for w in wlist:
    print(w)
    print(pywt.Wavelet(w).filter_bank)
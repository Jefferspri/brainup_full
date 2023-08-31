import pandas as pd
import datetime
from scipy import signal
import matplotlib.pyplot as plt

df_eeg = pd.read_csv("exports/eeg_data_fabrizio.csv")

df_eeg["time"] = df_eeg["time"].map(lambda x: datetime.datetime.fromtimestamp(x))
df_eeg = df_eeg.iloc[:, :-1]

fs = 256
(f, eeg) = signal.welch(df_eeg["AF8"], fs, nperseg = df_eeg.shape[0])

plt.semilogy(f, eeg)
plt.xlabel("frecuency")
plt.ylabel("PSD")
plt.show()
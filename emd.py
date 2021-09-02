from __future__ import division, print_function
import pandas as pd
import numpy  as np
import pylab as plt
from PyEMD import EMD



df = pd.read_excel('./data/ali.xlsx')
s=df['Open2'].values

# Define signal
t = np.linspace(1, 8, 6586)          #三个参数表示间隔从1到8，其中有6586个点。如果省略第三个参数，则第三个参数默认为50

# Execute EMD on signal
IMF = EMD().emd(s,t)
N = IMF.shape[0] + 1

# Plot results
plt.subplot(N, 1, 1)
plt.plot(t, s, 'r')
plt.title("result")
plt.xlabel("Time [s]")

for n, imf in enumerate(IMF):
    plt.subplot(N, 1, n + 2)
    plt.plot(t, imf, 'g')
    plt.title("IMF " + str(n + 1))
    plt.xlabel("Time [s]")

plt.tight_layout()
plt.savefig('simple_example')
plt.show()


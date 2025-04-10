import numpy as np
import matplotlib.pyplot as plt

def Senoidal():
    x = np.linspace(-np.pi/2, 3*np.pi/2, 48, endpoint=True)
    x = 10*(np.sin(x)+1)/2
    x = np.repeat(x, 3)
    y = np.linspace(-np.pi/2, 3*np.pi/2, 12, endpoint=True)
    y = 10*(np.sin(y)+1)/2
    y = np.repeat(y, 12)
    x = np.concatenate((x, y))
    x = np.tile(x, 2)

    return x

def Flatten(t):
    return [item for sublist in t for item in sublist]

plt.figure()

plt.subplot(311)
plt.title("PLANTA")
plt.grid()
plt.legend()

plt.subplot(323)
plt.title("ANFIS Python")
plt.grid()
plt.legend()

plt.subplot(324)
plt.title("ANFIS Matlab")
plt.grid()
plt.legend()

plt.subplot(325)
plt.title("Input")
plt.grid()
plt.legend()

plt.subplot(326)
plt.title("Erro")
plt.grid()
plt.legend()

plt.show()
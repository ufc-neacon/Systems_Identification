import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#ensaios_label = ["mpc(nh=15, nr=0, ip=10)", "mpc(nh=5, nr=0, ip=10)", "mpc(nh=25, nr=0, ip=10)", "mpc(nh=50, nr=0, ip=10)", "mpc(nh=15, nr=1, ip=10)", "mpc(nh=5, nr=1, ip=10)", "mpc(nh=25, nr=1, ip=10)", "mpc(nh=50, nr=1, ip=10)", "mpc(nh=15, nr=0, ip=10)", "mpc(nh=5, nr=0, ip=100)", "mpc(nh=25, nr=0, ip=1e-1)", "mpc(nh=50, nr=0, ip=1e-2)", "mpc(nh=31, nr=1, ip=1)", "mpc(nh=9, nr=0, ip=1e-2)", "mpc(nh=43, nr=0, ip=1e3)", "mpc(nh=10, nr=1, ip=1e-3)"]

#for i in range(len(ensaios_label)):

input = np.loadtxt('INPUT_Planta_' + "Res_1" + '.txt')
output = np.loadtxt('OUTPUT_Planta_' + "Res_1" + '.txt')
anfis = np.loadtxt('ANFIS_Planta_' + "Res_1" + '.txt')
erro = np.loadtxt('ERRO_Planta_' + "Res_1" + '.txt')

#print(f'{i+1}: {max(output)}')

plt.figure()

plt.subplot(311)
plt.plot(output, label="Saída Planta")
plt.grid()
plt.legend()
#plt.title(ensaios_label[i])

plt.subplot(312)
plt.plot(input, label="Entrada Controlada")
plt.grid()
plt.legend()

plt.subplot(325)
plt.plot(anfis, label="ANFIS Python")
plt.grid()
plt.legend()

plt.subplot(326)
plt.plot(erro, label="Erro")
plt.grid()
plt.legend()

plt.show()
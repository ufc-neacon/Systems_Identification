import matlab.engine as me
import numpy as np
import matplotlib.pyplot as plt
from senoide import Senoidal, Flatten

eng = me.start_matlab()

#x = Senoidal().tolist()
#x = [10, 10, 10, 10, 10, 10]
x = np.repeat(10, 20)

y_nivel = [float(0)]
y_volume = [float(0)]
y_ = [float(0)]

for i in range(len(x)):

    print(y_volume[i])
    y = eng.simulacaopy(float(x[i]), y_volume[i])
    print(y)
    y = Flatten(y)
    y_nivel.append(y[0])
    y_volume.append(y[1])
    y_.append(y[2])

np.savetxt('x1.txt',x)
np.savetxt('y_1.txt',y_)
np.savetxt('y_1nivel.txt',y_nivel)
np.savetxt('y_1volume.txt',y_volume)

#plt.subplot(x, label="Entrada")
#plt.grid()
#plt.legend()
plt.plot(y_, label="Saida")
plt.grid()
plt.legend()

plt.show()

eng.quit()
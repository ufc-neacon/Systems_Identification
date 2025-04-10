# -*- coding: utf-8 -*-

"""
Código ensaio MPC com ANFIS 
"""
try:
    import numpy as np
    from casadi import *
    import do_mpc
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import time
    import nidaqmx
    from nidaqmx.constants import TerminalConfiguration
    from matplotlib.animation import FuncAnimation, FFMpegWriter, ImageMagickWriter
    import sklearn
    from sklearn.utils import shuffle
    from sklearn.neighbors import KNeighborsRegressor
    import pandas as pd
    import matlab.engine as me
except:
    print ("Erro de import")

DATATIME = str(time.localtime().tm_mday) + '_' + str(time.localtime().tm_mon) + '_' + str(time.localtime().tm_mday) + '_' + str(time.localtime().tm_hour)

def Flatten(t):
    return [item for sublist in t for item in sublist]

class daq:
    DATATIME = str(time.localtime().tm_mday) + '_' + str(time.localtime().tm_mon) + '_' + str(time.localtime().tm_mday) + '_' + str(time.localtime().tm_hour)

    def __init__(self, ts, numAmostras): #ts = tempo de amostragem; numAmostras = numero de amostras para filtro passa baixa
        self.sinalEntrada = np.array([])
        self.sinalSaida = np.array([])
        self.ts = ts
        self.numAmostras = numAmostras
        self.readData = None
        self.writeData = 0
        self.flag = False

    def setWriteData(self, sinal):
        self.writeData = sinal
    
    def getReadData(self):
        return self.readData

    def ensaio(self):
        #Inicia a task de leitura e geração; Se a flag = True -> O ensaio acaba
        if self.flag:
            #Inicia a task de leitura e geração;
            with nidaqmx.Task() as writeTask:
                #Configuração das portas do daq
                writeTask.ao_channels.add_ao_voltage_chan("Dev2/ao0")

                #Gera o sinal:
                writeTask.write(0)

            #Salva os dados de geração e de leitura:
            np.savetxt('sinalEntrada' + self.DATATIME + '.txt', self.sinalEntrada)
            np.savetxt('sinalSaida' + self.DATATIME + '.txt', self.sinalSaida)
        else:
            with nidaqmx.Task() as writeTask, nidaqmx.Task() as readTask:
                #Configuração das portas do daq
                readTask.ai_channels.add_ai_voltage_chan("Dev2/ai0",  terminal_config=TerminalConfiguration.RSE)
                writeTask.ao_channels.add_ao_voltage_chan("Dev2/ao0")

                #Gera o sinal:
                writeTask.write(self.writeData/6)

                #Ler o sinal filtrado:
                vetorAmostras = np.array([])
                for i in range(0, self.numAmostras):
                    dadoInst = readTask.read()
                    vetorAmostras = np.append(vetorAmostras, dadoInst)
                    time.sleep(self.ts)
                self.readData = np.sum(vetorAmostras)/len(vetorAmostras)

            #Salva os dados de geração e de leitura:
            self.sinalEntrada = np.append(self.sinalEntrada, self.writeData)
            np.savetxt('sinalEntrada' + self.DATATIME + '.txt', self.sinalEntrada)
            self.sinalSaida = np.append(self.sinalSaida, self.readData)
            np.savetxt('sinalSaida' + self.DATATIME + '.txt', self.sinalSaida)

    def fimEnsaio(self):
        self.flag = True

def modeloEdo(modelType, tstep):

    model = do_mpc.model.Model(modelType)

    # Parameters
    R_ = 1.27/2
    R = 1.46/2
    H = 1.38
    GAMMA = 9810
    ki = 0.632208327437036
    ko = 0.055762231929563

    # States struct (optimization variables):
    h_fixo = model.set_variable(var_type='_x', var_name='h_fixo', shape=(1,1))

    # Input struct (optimization variables):
    w_fixo = model.set_variable(var_type='_u', var_name='w_fixo')

    # Right-hand-side equation
    a=ki*w_fixo-ko*np.sqrt(GAMMA*h_fixo)
    b = (np.pi*R_**2+(2*np.pi*R_*h_fixo*(R-R_))/H+(np.pi*h_fixo**2*(R-R_)**2)/H**2)
    model.set_rhs('h_fixo', a/b)

    # Build the model
    model.setup()

    # Build simulador
    simulador = do_mpc.simulator.Simulator(model)

    params_simulator = {
        'integration_tool': 'cvodes',
        'abstol': 1e-10,
        'reltol': 1e-10,
        't_step': tstep
    }

    simulador.set_param(**params_simulator)

    simulador.setup()

    return model, simulador

def modeloAnfis(modelType, tstep):
    model = do_mpc.model.Model(modelType)

    # Parameters
    MI          = 3
    NUMINVARS   = MI
    NUMINTERMS  = 5
    NUMRULES    = NUMINTERMS

    b1 = np.array([[-15.4759654931256,12.2209704903274,0.289123398113882,-71.9359064235566,21.1029827274901],[53.1345392743568,59.6156777644246,24.2261619397451,-111.016028737690,-79.7593631490980],[-27.2309348684618,-11.1514175692144,-0.889195034461636,12.4540272772075,30.5675786092304]])
    mean1 = np.array([[-103.480659859437,0.160225765345676,47.1901134705417,55.2789314313193,493.802868016107],[-101.143381316956,-86.2525384831843,-60.0254473600103,-49.3050420141232,61.7302726165688],[-34.1633413139230,-17.2408154290438,-16.7182517919470,33.7842984316230,60.0187522106859]])
    sigma1 = np.array([[75.1884591042617,-8.81165137905587,65.4355998025384,154.149488192979,19.3129732237862],[-28.6583733437366,177.263377304061,-4.58534607185469,118.220932608957,-66.0488427183459],[-104.852779993955,4.48200824012751,-18.0814276368059,51.2594822477457,13.0146519553683]])
    ThetaL4 = np.array([[442.289967536936],[-230.632389559791],[-319.333734336911],[-69.6453354552407],[-0.0129190219528040],[0.0350950739573748],[0.977793760104156],[-0.000167235344897465],[-6655.46642722836],[6554.95122169587],[-448.603649444125],[35.1420162336694],[-1075.92572429594],[1140.71744166750],[-2455.19434546434],[156.966414853777],[109.045398229963],[-80.6689421807745],[9573.16412265227],[-584.839849378831]])

    # States struct (optimization variables):
    h_anfis = model.set_variable(var_type='_x', var_name='h_anfis', shape=(1,1))
    uo_anfis = model.set_variable(var_type='_x', var_name='uo_anfis', shape=(1,1)) #u(k-1)
    # ho_anfis = model.set_variable(var_type='_x', var_name='ho_anfis', shape=(1,1)) #h(k-1)
        
    # Input struct (optimization variables):
    u_anfis = model.set_variable(var_type='_u', var_name='u_anfis') #u(k)

    # LAYER 1 - INPUT TERM NODES:
    In1 = SX(3,1)
    In1[0,0] = uo_anfis
    In1[1,0] = u_anfis
    In1[2,0] = (h_anfis)
    In2 = In1@SX.ones(1,NUMINTERMS)
    Out1 = 1/(1 + (fabs((In2-mean1)/sigma1))**(2*b1))

    # LAYER 2 - PRODUCT NODES:
    Out2 = SX.ones(NUMINTERMS,1)
    for i in range(NUMINTERMS):
        for j in range(NUMINVARS ):
            Out2[i] = Out2[i]*Out1.T[i,j]
    S_2 = SX(0)
    for i in range(NUMINTERMS):
        S_2 = S_2+Out2[i]

    # LAYER 3 - NORMALIZATION NODES:
    Out3 = Out2.T/S_2

    # LAYERS 4 - 5: CONSEQUENT NODES - SUMMING NODE:
    In1_ = SX(4,1)
    In1_[0,0] = In1[0,0]
    In1_[1,0] = In1[1,0]
    In1_[2,0] = In1[2,0]
    In1_[3,0] = 1
    Aux1 = In1_@Out3

    # New Input Training Data shaped as a column vector:
    a = reshape(Aux1,((NUMINVARS+1)*NUMRULES,1))
    hnext_aux = a.T@ThetaL4
    hnext = hnext_aux

    model.set_rhs('h_anfis', hnext) 
    model.set_rhs('uo_anfis', u_anfis) 

    # Build the model
    model.setup()

    # Build simulador
    simulador = do_mpc.simulator.Simulator(model)

    params_simulator_ANFIS = {
        't_step': tstep
    }

    simulador.set_param(**params_simulator_ANFIS)

    #Simulator Setup
    simulador.setup()

    return model, simulador

def controladorAnfis(model, tstep, SETPOINT): 

    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
    'n_horizon': 15,
    'n_robust': 0,
    't_step': tstep,
    'state_discretization': 'discrete',
    'store_full_solution': True,
    }
    mpc.set_param(**setup_mpc)

    mterm = 10*((model.x['h_anfis']-SETPOINT)**2) # terminal cost
    lterm =  100*((model.x['h_anfis']-SETPOINT)**2) # stage cost
    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(u_anfis = 10) # input penalty

    ### Constraints:

    # # lower bounds of the states
    mpc.bounds['lower', '_x', 'h_anfis'] = 0
    mpc.bounds['lower', '_x', 'uo_anfis'] = 0

    # # upper bounds of the states
    mpc.bounds['upper', '_x', 'h_anfis'] = 10
    mpc.bounds['upper', '_x', 'uo_anfis'] = 10

    # # lower bounds of the inputs
    mpc.bounds['lower', '_u', 'u_anfis'] = 0

    # # upper bounds of the inputs
    mpc.bounds['upper', '_u', 'u_anfis'] = 10

    mpc.setup()

    return mpc


def plantaSimu(input, volumeA):
    y = eng.simulacaopy(input, volumeA)
    y = Flatten(y)
    return y

#%% Closed-loop simulation config

eng = me.start_matlab()

tstep = 25

modelAnfis, simuladorAnfis = modeloAnfis('discrete', tstep)
modelEdo, simuladorEdo = modeloEdo('continuous', tstep)
mpc = controladorAnfis(modelAnfis, tstep, 3)

init = 0.1
Uinit = 0.1
Einit = 0
X0 = np.array([init, Uinit])

mpc.x0 = X0
mpc.set_initial_guess()

simuladorEdo.reset_history()
simuladorEdo.x0 = np.array([[init]])

simuladorAnfis.reset_history()
simuladorAnfis.x0 = X0

mpc.reset_history()

#%% Closed-loop simulation ensaio

tsimu = 600
HsimuAnfis_h = np.zeros([1,tsimu])
HsimuEDO_h = np.zeros([1,tsimu])
HsimuPlanta = np.zeros([1,tsimu], dtype=object)
u_mpc = np.zeros([1,tsimu])
Einit = np.zeros([1,tsimu])
#planta = daq(0.1, 100) #Planta com 10s de ensaio(geração + leitura)

volumeA = 0

##Gerar os gráficos
plt.figure()
plt.subplot(211)
plt.grid()
plt.subplot(212)
plt.grid()

count = 0

for k in range(tsimu):
    u0 = mpc.make_step(X0)
    u_mpc[0][k] = u0
    np.savetxt('sinalSaida_INPUT_Planta' + DATATIME + '.txt', u_mpc[0])

    print(u0[0][0])
    y = plantaSimu(float(u0[0][0]), float(volumeA))
    HsimuPlanta[0][k] = y[2]
    volumeA = y[1]
    np.savetxt('sinalSaida_MPC_Planta' + DATATIME + '.txt', HsimuPlanta[0])

    HsimuEDO = simuladorEdo.make_step(u0)
    HsimuEDO_h[0][k] = HsimuEDO[0][0]
    np.savetxt('sinalSaida_EDO_Planta' + DATATIME + '.txt', HsimuEDO_h[0])

    HsimuAnfis = simuladorAnfis.make_step(u0)
    HsimuAnfis_h[0][k] = HsimuAnfis[0][0]
    np.savetxt('sinalSaida_ANFIS_Planta' + DATATIME + '.txt', HsimuAnfis_h[0])

    Einit[0][k] = HsimuPlanta[0][k]-HsimuAnfis[0][0]
    np.savetxt('sinalSaida_ERRO_Planta' + DATATIME + '.txt', Einit[0])

    X0 = np.array([HsimuAnfis[0][0]+Einit[0][k], u0[0][0]])

    if k == 200:
        controladorAnfis(modelAnfis, tstep, 5)
        mpc.x0 = X0
        mpc.set_initial_guess()
    elif k == 400:
        controladorAnfis(modelAnfis, tstep, 1)
        mpc.x0 = X0
        mpc.set_initial_guess()

    time.sleep(1)

    plt.clf()
    plt.subplot(211)
    plt.plot(u_mpc[0][0:k], label="MPC input")
    plt.legend()
    plt.grid()
    plt.subplot(212)
    plt.plot(HsimuAnfis_h[0][0:k], label="ANFIS")
    plt.plot(HsimuEDO_h[0][0:k], label="EDO")
    plt.plot(HsimuPlanta[0][0:k], label="PLANTA")
    plt.grid()
    plt.legend()
    plt.show(block=False)
    plt.pause(0.01)

    #planta.setWriteData(u0[0][0])

#%% Plot results
plt.figure()
plt.plot(HsimuAnfis_h[0], label="ANFIS")
plt.plot(HsimuEDO_h[0], label="EDO")
#plt.plot(HsimuPlanta[0][k], label="PLANTA")
plt.plot(Einit[0], label="Erro")
plt.plot(u_mpc[0], label="MPC input")
plt.grid()
plt.legend()
plt.savefig("test")
plt.show()
#planta.fimEnsaio()

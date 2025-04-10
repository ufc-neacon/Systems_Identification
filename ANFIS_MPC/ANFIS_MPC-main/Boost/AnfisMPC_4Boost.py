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

def modeloAnfis(modelType, tstep):
    model = do_mpc.model.Model(modelType)

    # Parameters
    MI          = 7
    NUMINVARS   = MI
    NUMINTERMS  = 7
    NUMRULES    = NUMINTERMS

    b1 = np.array([[-44.0573249198801,-3.01318228051362,-13.6834056692177,5.39229512944622,8.65943640411405,5.43090099217590,-11.7574885245845],[-34.3161545180684,110.735628861312,3.07245327629263,-33.0859943643098,-8.81749579181387,-128.488946681374,22.4316392854820],[-123.108169235995,3.34528448319628,-14.7383081893959,15.7335876926653,10.1801527697701,65.9173989576329,-34.7500391619463],[-6.70405982410581,-29.6048184503747,-21.3043228534210,-8.56967092616348,-14.4044713928412,52.0065608219453,63.5990220338469],[22.8005873682614,10.4961534227908,-14.5037878890356,10.3376165350608,-4.80421733066544,-49.1136959446346,-137.006217332172],[69.0794273784091,7.20203931365437,-19.0034235694986,7.62958843330971,-10.2178561921036,16.1178375209470,4.41059662976930],[-65.4743326214945,47.3618856245921,-27.5599232595838,0.941389117178850,-11.5505637695882,-13.7922811626365,-263.606950367280]])
    mean1 = np.array([[-353.558410197683,-38.8880222148552,5.40142914619578,6.80853915779209,9.48449270370711,20.2424691642285,34.2647687468822],[-222.225262763012,-36.7798595978532,3.94923031601152,8.66615507724804,32.6731414958159,93.7841352029563,502.990966979516],[-168.273615611018,-0.399546403788738,24.8213308495999,27.6770070878262,30.1875214694559,45.2307556648521,124.641408677142],[3.41273273593276,11.1765159683727,21.5977530907096,26.1026649090919,29.3642358964460,36.8977650733427,111.063342031655],[-13.5966405704134,4.15076976320543,5.51669315268063,6.59974296281747,20.3392757558742,63.6275550199403,1217.87059690949],[-46.4150986304193,-14.1696201346101,0.374268520037133,10.7766774391450,33.4573043016600,74.9419370136020,95.6169180136497],[-123.600518262858,-19.9535936636326,-6.90175211385527,2.49199762884177,4.75861316945817,8.93053730527283,405.541790177747]])
    sigma1 = np.array([[-232.089887129330,81.2571209681163,4.53961203966865,-3.84929711760597,5.50035084162194,-15.1702256916638,-14.1113843532041],[150.397662050080,176.656320508728,1.08286037884481,3.18378146988815,47.1906102979283,120.635680227603,-8.11517844311309],[-80.9139823674148,-8.59199912614853,48.4233928427262,4.63210725410515,-37.5392641668837,-81.6468481741995,51.7976055278881],[-1.55017614734290,65.4472334927356,18.3353485688629,8.87180573961267,-12.6148857235093,66.0053247804628,80.7361205456110],[36.6025944971823,14.2740916422147,3.89839744610439,10.7401486787670,6.85239853705069,27.5763031280418,109.089420798546],[91.1749243364195,24.2483774104017,-30.8412690351411,-4.63599182139312,21.7248364778519,-88.4261250755028,-34.1716571821792],[19.5193979090080,-94.0672107010295,-5.32139343457962,0.971146841504750,7.01543226090493,-13.0271777374607,171.284302806004]])
    ThetaL4 = np.array([[-0.153555572809097],[0.149746915930402],[0.410621015454787],[-0.0129790885509411],[-0.0196556638796432],[0.487412969953205],[0.170476444169948],[-0.00190356205581022],[5814.63551577469],[5814.77854667134],[-36574.6337839047],[-146262.662472782],[35697.0539481796],[-195036.751813150],[436562.236910930],[-424366.148134832],[-19043.2582750946],[-19043.1166400735],[78824.0382060923],[10572.4701027861],[-115124.981889694],[55640.4858281335],[113997.196968512],[-165606.417692860],[-62442.3675632557],[-62442.3489457048],[-37872.3617710610],[42711.0311518595],[-346089.702802060],[-20368.5553197567],[-73178.2422722943],[96116.6410661992],[-52010.5743668421],[-52011.0005567096],[23034.1074440663],[44915.9111243212],[-370206.668828496],[3569.32188381144],[70016.7787876976],[119330.251234137],[188956.038150015],[188955.796528095],[-77573.9535202018],[84401.9464049275],[-685456.909928298],[-10468.2582348613],[-240290.639127713],[-58134.8468008718],[-91201.8288621800],[-91201.9885440665],[-31748.2338626064],[1143.20159346931],[-11956.7876970884],[5170.18012366402],[-125687.999656450],[-150849.596538010]])

    # States struct (optimization variables):
    vo = model.set_variable(var_type='_x', var_name='vo', shape=(1,1))
    uo = model.set_variable(var_type='_x', var_name='uo', shape=(1,1)) #u(k-1)
    # ho_anfis = model.set_variable(var_type='_x', var_name='ho_anfis', shape=(1,1)) #h(k-1)
    il1 = model.set_variable(var_type='_x', var_name='il1') 
    il2 = model.set_variable(var_type='_x', var_name='il2') 
    vc1 = model.set_variable(var_type='_x', var_name='vc1') 
    vc2 = model.set_variable(var_type='_x', var_name='vc2')
        
    # Input struct (optimization variables):
    u = model.set_variable(var_type='_u', var_name='u') 

    # LAYER 1 - INPUT TERM NODES:
    In1 = SX(7,1)
    In1[0,0] = uo
    In1[1,0] = u
    In1[2,0] = vo
    """ In1[3,0] = il1
    In1[4,0] = il2
    In1[5,0] = vc1
    In1[6,0] = vc2 """
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
    In1_ = SX(8,1)
    In1_[0,0] = In1[0,0]
    In1_[1,0] = In1[1,0]
    In1_[2,0] = In1[2,0]
    In1_[3,0] = In1[3,0]
    In1_[4,0] = In1[4,0]
    In1_[5,0] = In1[5,0]
    In1_[6,0] = In1[6,0]
    In1_[3,0] = 1
    Aux1 = In1_@Out3

    # New Input Training Data shaped as a column vector:
    a = reshape(Aux1,((NUMINVARS+1)*NUMRULES,1))
    vo_next = a.T@ThetaL4

    model.set_rhs('vo', vo_next) 
    model.set_rhs('uo', u) 
    model.set_rhs('il1', il1) 
    model.set_rhs('il2', il2) 
    model.set_rhs('vc1', vc1) 
    model.set_rhs('vc2', vc2) 


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

def controladorAnfis(model, tstep, SETPOINT, n_horizon, n_robust, inputPenalty): 

    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
    'n_horizon': n_horizon,
    'n_robust': n_robust,
    't_step': tstep,
    'state_discretization': 'discrete',
    'store_full_solution': True,
    }
    mpc.set_param(**setup_mpc)

    mterm = 10*((model.x['vo']-SETPOINT)**2) # terminal cost
    lterm =  100*((model.x['vo']-SETPOINT)**2) # stage cost
    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(u = inputPenalty) # input penalty

    ### Constraints:

    # # lower bounds of the states
    mpc.bounds['lower', '_x', 'vo'] = 0
    mpc.bounds['lower', '_x', 'uo'] = 0
    mpc.bounds['lower', '_x', 'il1'] = 0
    mpc.bounds['lower', '_x', 'il2'] = 0
    mpc.bounds['lower', '_x', 'vc1'] = 0
    mpc.bounds['lower', '_x', 'vc2'] = 0

    # # upper bounds of the states
    mpc.bounds['upper', '_x', 'vo'] = 10
    mpc.bounds['upper', '_x', 'uo'] = 10

    # # lower bounds of the inputs
    mpc.bounds['lower', '_u', 'u'] = 0.1

    # # upper bounds of the inputs
    mpc.bounds['upper', '_u', 'u'] = 0.9

    mpc.setup()

    return mpc

def plantaSimu(input, il1, il2, vc1, vc2):
    y = eng.simulacao_boost(input, il1, il2, vc1, vc2)
    y = Flatten(y)
    return y

#%% Closed-loop simulation config

ensaios_label = ["mpc(nh=15, nr=0, ip=10)", "mpc(nh=5, nr=0, ip=10)", "mpc(nh=25, nr=0, ip=10)", "mpc(nh=50, nr=0, ip=10)", "mpc(nh=15, nr=1, ip=10)", "mpc(nh=5, nr=1, ip=10)", "mpc(nh=25, nr=1, ip=10)", "mpc(nh=50, nr=1, ip=10)", "mpc(nh=15, nr=0, ip=10)", "mpc(nh=5, nr=0, ip=100)", "mpc(nh=25, nr=0, ip=1e-1)", "mpc(nh=50, nr=0, ip=1e-2)", "mpc(nh=31, nr=1, ip=1)", "mpc(nh=9, nr=0, ip=1e-2)", "mpc(nh=43, nr=0, ip=1e3)", "mpc(nh=10, nr=1, ip=1e-3)"]

n_horizon = [15, 5, 25, 50, 15, 5, 25, 50, 15, 5, 25, 50, 31, 9, 43, 10]
n_robust = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1]
input_penalty = [10, 10, 10, 10, 10, 10, 10, 10, 10, 100, 25, 1e-1, 1e-2, 1, 1e-2, 1e3, 1e-3]

eng = me.start_matlab()

tstep = 25

modelAnfis, simuladorAnfis = modeloAnfis('discrete', tstep)

for j in range(1):

    i = -1

    mpc = controladorAnfis(modelAnfis, tstep, 3, n_horizon[i], n_robust[i], input_penalty[i])

    init = 0.1
    Uinit = 0.1
    il1 = 0.1
    il2 = 0.1
    vc1 = 0.1
    vc2 = 0.1
    Einit = 0
    X0 = np.array([init, Uinit, il1, il2, vc1, vc2])

    mpc.x0 = X0
    mpc.set_initial_guess()

    simuladorAnfis.reset_history()
    simuladorAnfis.x0 = X0

    mpc.reset_history()

    #%% Closed-loop simulation ensaio

    tsimu = 1500
    HsimuAnfis_h = np.zeros([1,tsimu])
    """ HsimuEDO_h = np.zeros([1,tsimu]) """
    HsimuPlanta = np.zeros([1,tsimu], dtype=object)
    u_mpc = np.zeros([1,tsimu])
    Einit = np.zeros([1,tsimu])

    """ ##Gerar os gráficos
    plt.figure()
    plt.subplot(211)
    plt.grid()
    plt.subplot(212)
    plt.grid() """

    #count = 0

    for k in range(tsimu):
        u0 = mpc.make_step(X0)
        np.savetxt('INPUT_Planta_' + ensaios_label[i] + '.txt', u0)

        print(float(u0[0][0]), float(il1), float(il2), float(vc1), float(vc2))

        y = plantaSimu(float(u0[0][0]), float(il1), float(il2), float(vc1), float(vc2))
        print(y)
        HsimuPlanta[0][k] = y[0]
        il1 = y[1]
        il2 = y[2]
        vc1 = y[3]
        vc2 = y[4]
        np.savetxt('OUTPUT_Planta_' + ensaios_label[i] + '.txt', HsimuPlanta[0])

        """ HsimuEDO = simuladorEdo.make_step(u0)
        HsimuEDO_h[0][k] = HsimuEDO[0][0]
        np.savetxt('EDO_Planta_' + ensaios_label[i] + '.txt', HsimuEDO_h[0]) """

        HsimuAnfis = simuladorAnfis.make_step(u0/10)*6.180021026227129
        HsimuAnfis_h[0][k] = HsimuAnfis[0][0]
        np.savetxt('ANFIS_Planta_' + ensaios_label[i] + '.txt', HsimuAnfis_h[0])

        Einit[0][k] = HsimuPlanta[0][k]-HsimuAnfis[0][0]
        np.savetxt('ERRO_Planta_' + ensaios_label[i] + '.txt', Einit[0])

        X0 = np.array([HsimuAnfis[0][0]+Einit[0][k], u0[0][0], il1, il2, vc1, vc2])

        if k == 300:
            mpc = controladorAnfis(modelAnfis, tstep, 5, n_horizon[i], n_robust[i], input_penalty[i])
            mpc.x0 = X0
            mpc.set_initial_guess()
        elif k == 600:
            mpc = controladorAnfis(modelAnfis, tstep, 4, n_horizon[i], n_robust[i], input_penalty[i])
            mpc.x0 = X0
            mpc.set_initial_guess()
        elif k == 900:
            mpc = controladorAnfis(modelAnfis, tstep, 6, n_horizon[i], n_robust[i], input_penalty[i])
            mpc.x0 = X0
            mpc.set_initial_guess()
        elif k == 1200:
            mpc = controladorAnfis(modelAnfis, tstep, 2, n_horizon[i], n_robust[i], input_penalty[i])
            mpc.x0 = X0
            mpc.set_initial_guess()

        """ time.sleep(1)

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

        #planta.setWriteData(u0[0][0]) """

""" #%% Plot results
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
#planta.fimEnsaio() """

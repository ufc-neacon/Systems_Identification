clear all;
close all;
clc;

%% Converter parameters
% Quadratic Boost Data Structure
BoostQ.L1 = 1.48e-3;
BoostQ.L2 = 1.7e-3;
BoostQ.C1 = 100e-6;
BoostQ.C2 = 100e-6;
BoostQ.R = 175; %175
BoostQ.f = 20e3;
BoostQ.Vin = 30;
BoostQ.Ts = 1/(BoostQ.f);
% Parasitics
BoostQ.rL1 = 0.4;
BoostQ.rL2 = 0.8;
BoostQ.rC1 = 0.318;
BoostQ.rC2 = 0.318;
BoostQ.Rmos = 0.1;
BoostQ.Dn = 0.5; % Nominal DUTY CYCLE adjust
BoostQ.Vout = BoostQ.Vin/(1-BoostQ.Dn)^2;
save('Boost_Data','BoostQ');

%% PRBS
%TbN>/ Tassentamento
%1/(2^N-1)Tb < fprbs < 0.44/Tb
%Assentamento = 0.01
% PRBS
PRBS_N = 10;
Tb = 0.002; % 0.002
Lmax = 0.44/Tb;
Lmin = 1/((2^PRBS_N-1)*Tb);
PRBS_Ts = Tb;
% PRBS_Ts = 5;
PRBS_Amp = .05;
Tsim_min = (2^PRBS_N-1)*Tb;
Tsim = 1*Tsim_min;
Ramp_Max = 0.75;
% Tsim = 0.05; %aux1
Tsim = 0.1; %aux2

%% Data to identification

aux = linspace(0.1,0.9,20);
aux1 = linspace(0.1,0.75,10);
aux2 = linspace(0.75,0.9,10);

for i = 1:length(aux)
    Dc = aux(i);
    sim('BoostQuadSimu');
    U{i} = Din.Data;
    Y{i} = Vout.Data;
    IL1x{i} = IL1.Data;
    IL2x{i} = IL2.Data;
    VC1x{i} = VC1.Data;
    VC2x{i} = VC2.Data;
    Y_anfis{i} = y1.Data;
    plot(Y{i},'k')
    hold on
    plot(Y_anfis{i},'r');
end
grid
% axis([-1 4*10^4 0 180])
legend("Sistema","Anfis")

% ymin = Voutmin.Data;
% ymed = Vout.Data;
% ymax = Voutmax.Data;
% 
% %Normaliza saida
% y_norm = max(Voutmax.Data);
% ymin = ymin/y_norm;
% ymed = ymed/y_norm;
% ymax = ymax/y_norm;
% Gains.Ynorm = y_norm;
% 
% % Construção dos vetores
% U = [Dinmin.Data Din.Data Dinmax.Data];
% Y = [ymin ymed ymax];
Ts = BoostQ.Ts;



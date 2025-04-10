%% Initialization
%clear all
%close all
%clc

%% Model parameters 

% aux = linspace(7.52,8.45,10);
aux = linspace(0,10,10);
% aux = 4.25
% aux = 10;
Ts = 25;
Tsim = 500*Ts;

for i = 1:length(aux)
    inputSig = aux(i);
    sim('Tanques_gravidade');
    U{i} = u;
    Y{i} = y;
    Y_anfis{i} = y1;
    plot(Y{i},'k')
    hold on
    plot(Y_anfis{i},'r');
end
grid
axis([0 500 -1 7])
legend("Sistema","Anfis")
% fim do arquivo
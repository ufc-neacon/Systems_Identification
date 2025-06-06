function y = simulacaopy_tanque(x, y0)

assignin("base", "input", x);
assignin("base", "volume_inicial", y0);
sim("Tanques_gravidade.slx");
y_nivel = nivel(length(nivel));
y_volume = volume(length(volume));
y_ = yy(length(yy));
y_anfis = y1(length(y1));
y = [y_nivel, y_volume, y_, y_anfis];

end
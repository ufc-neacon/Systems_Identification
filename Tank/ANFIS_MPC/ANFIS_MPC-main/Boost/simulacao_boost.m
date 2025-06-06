function y = simulacao_boost(x, il1, il2, vc1, vc2)

assignin("base", "Dc", x);
assignin("base", "il1", il1);
assignin("base", "il2", il2);
assignin("base", "vc1", vc1);
assignin("base", "vc2", vc2);
sim("BoostQuadSimu.slx");

yout = Vout(length(Vout));
il1_ = IL1(length(IL1));
il2_ = IL2(length(IL2));
vc1_ = VC1(length(VC1));
vc2_ = VC2(length(VC2));

y = [yout, il1_, il2_, vc1_, vc2_];

end
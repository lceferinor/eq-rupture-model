%% MLE Estimator for Brownian Passage Time and Weibull Probability Distribution
N_Rupture_zones = 8;
L_rup = 620/N_Rupture_zones; % Rupture unit
% From South to North: Dorbath

t0 = [2007 2007 1974 1974 1974 1966 1966 1746]; %Time from the last rupture
Mean_T = [320 172 194 97 114 110 144 34]; % No information in the last 3 segments
Covar_T = [1.22 1.22 0.68 .73 .99 1.08 0.75 .54];

BTP_u = zeros(1, N_Rupture_zones);
BTP_a = zeros(1, N_Rupture_zones);
WPD_a = zeros(1, N_Rupture_zones);
WPD_b = zeros(1, N_Rupture_zones);

% Define ruptures
T_0 = 1500;
T_f = 2017;
Rupture_history = zeros(T_f - T_0 + 1, N_Rupture_zones);


% From South to North: Dorbath
% Segment 1: Since we have only 2 events
T = 320;
BTP_a(1) = Covar_T(1); BTP_u(1) = T;
[WPD_a(1), WPD_b(1)] = EqualMoments_WPD(T, Covar_T(1)*T);
Rupture_history((1687 - T_0 + 1),1) = 1;
Rupture_history((2007 - T_0 + 1),1) = 1;


% Segment 2
T = [320 23];
[BTP_a(2), BTP_u(2)] = MLE_BPT(T);
[WPD_a(2), WPD_b(2)] = MLE_WPD(T);
Rupture_history((1664 - T_0 + 1),2) = 1;
Rupture_history((1687 - T_0 + 1),2) = 1;
Rupture_history((2007 - T_0 + 1),2) = 1;


% Segment 3
T = [287 101];
[BTP_a(3), BTP_u(3)] = MLE_BPT(T);
[WPD_a(3), WPD_b(3)] = MLE_WPD(T);
Rupture_history((1586 - T_0 + 1),3) = 1;
Rupture_history((1687 - T_0 + 1),3) = 1;
Rupture_history((1974 - T_0 + 1),3) = 1;


% Segment 4:
T = [34 194 59 101];
[BTP_a(4), BTP_u(4)] = MLE_BPT(T);
[WPD_a(4), WPD_b(4)] = MLE_WPD(T);
Rupture_history((1586 - T_0 + 1),4) = 1;
Rupture_history((1687 - T_0 + 1),4) = 1;
Rupture_history((1746 - T_0 + 1),4) = 1;
Rupture_history((1940 - T_0 + 1),4) = 1;
Rupture_history((1974 - T_0 + 1),4) = 1;


% Segment 5: 
T = [34 194];
[BTP_a(5), BTP_u(5)] = MLE_BPT(T);
[WPD_a(5), WPD_b(5)] = MLE_WPD(T);
Rupture_history((1746 - T_0 + 1),5) = 1;
Rupture_history((1940 - T_0 + 1),5) = 1;
Rupture_history((1974 - T_0 + 1),5) = 1;


% Segment 6
T = [26 194];
[BTP_a(6), BTP_u(6)] = MLE_BPT(T);
[WPD_a(6), WPD_b(6)] = MLE_WPD(T);
Rupture_history((1746 - T_0 + 1),6) = 1;
Rupture_history((1940 - T_0 + 1),6) = 1;
Rupture_history((1966 - T_0 + 1),6) = 1;


% Segment 7
T = [220 68];
[BTP_a(7), BTP_u(7)] = MLE_BPT(T);
[WPD_a(7), WPD_b(7)] = MLE_WPD(T);
Rupture_history((1678 - T_0 + 1),7) = 1;
Rupture_history((1746 - T_0 + 1),7) = 1;
Rupture_history((1966 - T_0 + 1),7) = 1;


% Segment 8
T = [21 47];
[BTP_a(8), BTP_u(8)] = MLE_BPT(T);
[WPD_a(8), WPD_b(8)] = MLE_WPD(T);
Rupture_history((1678 - T_0 + 1),8) = 1;
Rupture_history((1725 - T_0 + 1),8) = 1;
Rupture_history((1746 - T_0 + 1),8) = 1; % Adding this for the journal paper


% PDF of Section 4:
t = 0:1:250;

% BTP
pdf_BTP = BTP_PDF(t, BTP_a(4), BTP_u(4));
f = figure;
hold on;
% Weibull
pdf_WPD = WPD_PDF(t, WPD_a(4), WPD_b(4));

plot(t, pdf_BTP, t, pdf_WPD);
xlabel('Interarrival Time (years)');
ylabel('Probability Distribution Function');  
legend({'BPT Distribution','Weibull Distribution'});
axis([0 250 0 10^-2])
grid on;
title('PDF of the Interarrval Time for Section Number 4');
saveas(f, 'PDF_Section4.png');  







% Save BTP and WPD parameters
save TD_Parameters_Segments.mat WPD_a WPD_b BTP_a BTP_u Mean_T Covar_T t0 T_0 T_f Rupture_history;

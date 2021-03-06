%% Rupture Rate Segments: Get the rates from SD_Bernulli: Update if you 



%% update SD_Bernulli
clc; clearvars; close all;

N_rup_cases = 8*9/2;
Earthquake_set = zeros(2,N_rup_cases); % Columns: first 8 (1-seg), next 7 (2-seg), 
                              % next 6 (3-seg), next 5 (4-seg), ... 
                              % Rows: 1: BPT, 2: WPD
% Gamma = 10,

%%%%%%%%%%%%%%%%%%%%%%%%%%%% SD_Bernulli %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% MC Simuation of Mc

% Load BTP and WPD parameters


tic;
addpath(fullfile(pwd));
colors = {'k', 'y', 'm', 'c', 'r', 'g', 'b'};

load TD_Parameters_Segments.mat % It has: WPD_a WPD_b BTP_a BTP_u Mean_T std_T t0
                                % From South to North: Dorbath

% Histogram Bins for Mw
Mw_bins = 7.7499:0.5:9.5;
% From Dorbath Records
%Historical_Magnitudes = [8.10 7.85 7.50 7.85 8.40 7.50 8.60 8.20 8.10 ...
%    7.90 8.10 7.50 8.00];
Historical_Magnitudes = [8.10 7.50 7.85 8.40 7.50 8.60 8.20 8.10 ...
    8.10 7.99];


% Magnitude Histogram
[h_hist, x_hist]  = hist(Historical_Magnitudes + 0.001, Mw_bins); 
%hist_norm = h_hist/(sum(h_hist)*(x_hist(2) - x_hist(1))); % Normalize to are to get empirical pdf. 
hist_norm = h_hist;
% Seed of the random generator
seed = 20;
rng(seed);
% MC simulation
N_years = 10000;
%N_years = 100000;

N_Rupture_zones = 8;
L_rup = 620/N_Rupture_zones; % Rupture unit


% Rates from Chlie for Lima from South to North (Review Excel for details)
% (8th and 9th were assumed)
%M_rates = [3.08 4.81 4.4 3.47 4.7 5.49 3.06 0.01 0.01 0.01]*10^18*3.286; %8th, 9th and 10th were assumed
%M_rates = [2.79 4.38 6.08 11.28 8.14 7.37 6.97	3.26]*10^18;
M_rates = [3.73 4.17 5.42 11.04 6.79 7.38 6.46	5.15]*10^18; % Discrete


% Segment rupture times
Rup_T = cell(N_Rupture_zones,1);
for i = 1:N_Rupture_zones
    Rup_T{i} = [];
end

% CDF_t = @(x)BTP_CDF(x, BTP_a, BTP_u);
% PDF_t = @(x)BTP_PDF(x, BTP_a, BTP_u);
n_functions = 1;
CDF_t = {@(x, BTP_a, BTP_u)BTP_CDF(x, BTP_a, BTP_u), ...
    @(x, WPD_a, WPD_b)WPD_CDF(x, WPD_a, WPD_b) };
PDF_t = {@(x, BTP_a, BTP_u)BTP_PDF(x, BTP_a, BTP_u), ...
    @(x, WPD_a, WPD_b)WPD_PDF(x, WPD_a, WPD_b) };
Labels = {'BPT', 'Weibull'};
Parameters = cat(3, [BTP_a; BTP_u], [WPD_a; WPD_b]);


% Set uncorrelated Normal Random Variables
mu = 0;
sigma = 1;
Occurrance_Sample = normrnd(mu, sigma , N_years, N_Rupture_zones);
% Spatially Correlated Bernoulli


% Sensitivity
% Gamma = [5, 7.5, 10, 12.5, 15];
Gamma = [150,200,250,300,350];

U = [];
% f = figure;
% hold on;
% k = 1;
% legend_CorrM = cell(1,5);
for gamma = Gamma
    % Set covariance matrix
    Cov = ones(N_Rupture_zones, N_Rupture_zones);
    for i = 1:N_Rupture_zones
       for j = 1:i 
        Cov(i,j) = exp(-((j-i)*L_rup)^2/gamma^2);
%         Cov(i,j) = 1 - abs(j-i)/gamma;
        Cov(j,i) = Cov(i,j);
       end
    end
    % Cholesky decomposition
    U = cat(3, U, chol(Cov, 'upper'));
%     legend_CorrM{1, k} = ['\phi = ', round(num2str(gamma*L_rup)), ' km'];
%     h = plot((0:9), Cov(1,:), '-o');
%     set(h, 'Color', colors{k});
%     k = k + 1;
end
% 
% legend(legend_CorrM, 'Location', 'Southwest');
% legend(legend_CorrM);
% xlabel('Number of sections away from one to another');
% ylabel('Correlation between two sections');
% axis([0 10 0 1]);
% grid on;
% title('Correlation between separated pair of sections');
% saveas(f, 'CorrelationBSections.png');    

% Simulation
for k = 1:n_functions
    Yearly_Mom_Release = [];
    h_norm = [];
    % Set legends for graphs
    legend_moment = cell(1,length(Gamma));
    legend_hist = cell(1,length(Gamma));
    legend_moment{1,1} = 'Accumulated Moment (Historic Catalog)';
    legend_hist{1,1} = 'Historical Records (Historic Catalog)';
    
    for p = 1:length(Gamma)
        t = 2015 - t0;
        Simulation_Events = zeros(N_Rupture_zones, N_years);
        Moment_Release = zeros(N_Rupture_zones, N_years);
        Magnitude = [];
        for i = 1:N_years
            t = t + 1;
            CDF = CDF_t{1, k};
            p_equ = zeros(1, N_Rupture_zones);
            for j = 1:N_Rupture_zones
                CDF_temp = @(t) CDF(t, Parameters(1, j, k), ...
                    Parameters(2, j, k));
                p_equ(j) = (CDF_temp(t(j) + 1) - CDF_temp(t(j)))./...
                    (1 - CDF_temp(t(j)));
            end
            NormRV_corr = Occurrance_Sample(i,:)*U(:,:,p);
            unifcorr = normcdf(NormRV_corr);
            % Simulation threshold for earthquake occurrence
            % 1 means earthquake occurrence (True)
            Simulation_Events(:, i) = unifcorr < p_equ;
            n_seg = sum(Simulation_Events(:, i));
            if(n_seg > 0)
                rup_index = 1;
                % More than one rupture can happen in a given year
                while true
                    start = rup_index;
                    % Check rupture in the segment
                    while Simulation_Events(rup_index, i) == 1
                        rup_index = rup_index + 1;
                        if rup_index > N_Rupture_zones
                           break; 
                        end
                    end
                    % If no rupture in the segment , do not enter here
                    if(start ~= rup_index)
                        rup_sec = rup_index - start;
                        Mw = 1.62*log10(rup_sec*L_rup) + 4.44;
                        M0 = 10^((Mw +10.7)*3/2)*10^-7; %M0 in Nm
                        Moment_Release(start:(rup_index - 1), i) = M0/rup_sec;
                        Magnitude = [Magnitude; Mw];
                        %%% Count the Earthquake Event
                        set_pos = ((8+9)*(rup_sec-1) - (rup_sec-1)^2)/2; % 8 + 7 + ... (9 - set_pos)
                        Earthquake_set(k, set_pos + start) = ...
                            Earthquake_set(k, set_pos + start) + 1;
                        %%%
                        for i_seg = start:(rup_index - 1)
                            Rup_T{i_seg} = [Rup_T{i_seg}, t(i_seg)];
                        end
                    end
                    rup_index = rup_index + 1;
                    if rup_index > N_Rupture_zones
                       break; 
                    end
                end
            end
            % Time since the last earthquake goes to 0 
            t(unifcorr < p_equ) = 0;
        end
        Yearly_Mom_Release = [Yearly_Mom_Release, ...
            sum(Moment_Release,2)/N_years];
        legend_moment{1, p+1} = ['Released by Simulation: \phi = ', num2str(Gamma(p)), ' km'];
        [h, x] = hist(round(Magnitude*10)/10 + 0.001 , Mw_bins); % Adding 0.01 just to visualize Mw 7.5 as part of the interval 7.5-8 rather than 7 - 7.5
        h_norm = [h_norm; h];
        %h_norm = [h_norm; h/(sum(h)*(x(2) - x(1)))];
        legend_hist{1, p+1} = ['Simulated EQs: \phi = ', num2str(Gamma(p)), ' km'];
    end
    
% %     Plots
%     f = figure;
%     imagesc([2016, 2015 + N_years], N_Rupture_zones:-1:1, ...
%         ~fliplr(Simulation_Events')');
%     
%     colormap(gray);
%     set(gca,'YDir','normal')
%     xlabel('Time(Years)');
%     ylabel('Section Label');
%     title(['Earthquake Process Simulation: ', Labels{1,k}, ...
%         ' Distribution. Phi = ', num2str(round(Gamma(p))), ' km']);
%     saveas(f, [Labels{1,k}, num2str(round(Gamma(p))), 'EQ_Process.png']);     
% 
%     % Plots: Graph of consistency for segment 4
%     t_points = 0:2:400;
%     PDF = PDF_t{1, k};
%     PDF_temp = @(t) PDF(t, Parameters(1, 4, k), ...
%                     Parameters(2, 4, k));
%     pdf = PDF_temp(t_points);
%     T_bins = 0:20:400;
%     T_Rup = Rup_T{4};
%     [h, t_hist] = hist(T_Rup, T_bins); % Adding 0.01 just to visualize Mw 7.5 as part of the interval 7.5-8 rather than 7 - 7.5
%     h_norm = h./(sum(h)*(t_hist(2) - t_hist(1)));
%     f = figure;
%     hold on
%     bar(t_hist, h_norm);
%     plot(t_points, pdf, 'k','LineWidth', 2);
%     xlabel('Time(Years)');
%     ylabel('PDF');
%     title(['Probability Density Function ', Labels{1,k}, ...
%         ' Distribution. Phi = ', num2str(round(Gamma(end)*L_rup))]);
%     legend({'Simulation', 'BTP Distribution'});
%     saveas(f, [Labels{1,k}, 'Consistency.png']);     
%     
%     %Plots
%     
    f = figure;
    hold on;
    plot(1:8, M_rates(1:8), 'k', 'linewidth',2);
    pl = plot(1:8, Yearly_Mom_Release');
    % Set Color
    for p = 1:length(Gamma)
        set(pl(p), 'Color', colors{p+1});
    end
    xlabel('Segment Portion');
    ylabel('Average Moment Release per Year (Nm/year)');  
    legend(legend_moment);
    axis([1 8 0 2.5*10^19])
    title(['Accumulated vs Released Yearly M0 in Simulation using ', Labels{1,k}]);
    saveas(f, [Labels{1,k}, 'Moment_Release.png']);     

    f = figure;
    ba = bar(x_hist, [hist_norm/450; h_norm/N_years]', 'grouped');  % normalize to are to get empirical pdf.
   % Set Color
    set(ba(1), 'FaceColor', colors{1});
    for p = 1:(length(Gamma))
        set(ba(p+1), 'FaceColor', colors{p+1});
    end
    xlabel('Mw');
    ylabel('Normalized Frequency');
    title(['Empirical PDF of Mw: Historical vs. Simulated using ', Labels{1,k}]);
    legend(legend_hist);
    axis([7.5 9.5 0 30/1000]);
    saveas(f, [Labels{1,k}, 'Magnitude_Hist.png']);
    1./sum(Simulation_Events')*N_years
end



%%%%%%%%%%%%%%%%%%%%%%%%% End of SC_Bernulli %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Earthquake_rates = Earthquake_set/N_years;
save EQ_rates.mat Earthquake_rates

Parameters_BPT = [BTP_u;BTP_a;t0;Mean_T;Covar_T];
filename = 'Parameters_TD.csv';
csvwrite(filename,Parameters_BPT);

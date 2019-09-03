%% Bayesian non-parametric clustering
clear all
close all
clc
% Dirichlet Process Gaussian Mixture Model

% Load and plot the simulated data
% load ./data/simudata
load ./data/genedata
figure('name', 'simulated data')
plot(y(1,:), y(2,:), '.')
xlabel('First HKS')
ylabel('Second HKS')

% Parameters of the base distribution G0
hyperG0.mu = [0;0];
hyperG0.kappa = 1;
hyperG0.nu = 4;
hyperG0.lambda = eye(2);
% Scale parameter of DPM
alpha = 0.2;
% Number of iterations
niter = 20; 
% do some plots
doPlot = 2; 
% type_algo = 'BMQ'; % other algorithms: 'CRP', 'collapsedCRP', 'slicesampler'
type_algo = 'collapsedCRP';
%%
% # Look at the Matlab function gibbsDPM_algo1.
% # Run the Gibbs sampler for 20 iterations, with graphical output.
figure
[c_st, c_est, similarity] = gibbsDPM(y, hyperG0, alpha, niter, type_algo, doPlot);

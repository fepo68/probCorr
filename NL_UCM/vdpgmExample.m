%% VDMP toolbox

clear all
close all
clc

addpath('E:\Clases Doctorado\Applied Bayesian Nonparametrics\vdpgm\vdpgm');

% Load and plot the simulated data
load ./data/simudata
% load ./data/genedata
figure('name', 'simulated data');
plot(y(1,:), y(2,:), '.');
xlabel('X');
ylabel('Y');
X=y;
opts.algorithm = 'csb'; % csb : collapsed stick-breaking
result = vdpgm(X, opts);
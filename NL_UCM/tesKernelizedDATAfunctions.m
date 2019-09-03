%% Test Kernelized Forms Kxx Kww Kwx

clear all
clc 
close all

Md = 5, % number of features
K = 3; % number of observations

A = rand(Md,3);
B = rand(Md,1);

%% kernel parameters

kern.type = 'rbf';
kern.variance = 1;
kern.inverseWidth = 1;
tic
k = mykernCompute(kern,A)
toc
tic
G = gram(A',A','gauss')
toc
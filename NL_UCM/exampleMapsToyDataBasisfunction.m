%% MAPs Toy Dataset by using Polynomial Basis Functions

clear all
close all
clc

load synth3exp1D2 % synth3

x = X{1}(1,:);
kern.weightVariance = 1;
kern.biasVariance = 0;
kern.variance = 1;
kern.degree = 5;
fBasis = 'polynomial';
PHIX = polyToyMapBasis(X,kern,D,Nd,fBasis)

% Phi = designMatrix(X{1},'polynomial',10);
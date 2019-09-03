% Two arbitrary partitions
clear all
close all
clc

p1 = [0 1 1 1 0 0];
p2 = [0 1 1 0 0 0];



% Compute the unadjusted rand index
ri = rand_index(p1, p2);
assert(ri - 0.466666 < 10^-5);

% Compute the adjusted rand index
ri = rand_index(p1, p2, 'adjusted');
assert(ri + 1/9 < 10^-5);
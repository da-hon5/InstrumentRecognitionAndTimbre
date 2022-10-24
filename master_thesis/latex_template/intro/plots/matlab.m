% init
clear; close all; clc;
addpath('~/spr/matlab'); globalinit('silent');

% example plots
%     example1
figure; plot(linspace(0,3*pi,100), cos(linspace(0,3*pi,100)));
grid on; xlim([0,3*pi]); ylim(xyzlimits([-1,1])); setlabels('', 't', 'cos(t)');
savefigure(gcf, 'example1', 'eps', struct('papersize',[14.5,7]));
%     example2
figure; plot(linspace(0,3*pi,100), -cos(linspace(0,3*pi,100)));
grid on; xlim([0,3*pi]); ylim(xyzlimits([-1,1])); setlabels('', 't', '-cos(t)');
savefigure(gcf, 'example2', 'eps', struct('papersize',[14.5,7]));

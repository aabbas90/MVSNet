%% Add path:
% addpath(genpath('../../npy-matlab-master/'));
% addpath(genpath('../../minboundbox/'));
%% Set-up:
clear; clc; close all; fclose('all');
D = dir('../TEST_DATA_FOLDER/depths_mvsnet/');
thresh = 0.65;
vertices = [];
%% See merged depth;
filter = ones(5, 5);
filter = filter ./ sum(filter(:));
for i = 1:length(D)
    if (isempty(strfind(D(i).name, '.pfm')) || ...
      ~isempty(strfind(D(i).name, 'init')) || ...
      ~isempty(strfind(D(i).name, 'prob')))
        continue;
    end
    [R,t,K,dStart,dInt] = load_cam([D(i).folder, '\', D(i).name(1:end-4), '.txt']);
    [depthMap, scaleFactor] = parsePfm([D(i).folder, '\', D(i).name]);
    [probMap, scaleFactor] = parsePfm([D(i).folder, '\', D(i).name(1:end-4), '_prob.pfm']);
    probMap = conv2(probMap, filter, 'same');
    probMap(1:5,:) = 0;
    probMap(end-5:end,:) = 0;
    probMap(:, 1:5) = 0;
    probMap(:, end-5:end) = 0;
    probMap(depthMap < dStart + 120) = 0;
    indices = find(probMap > thresh);
    [u, w] = ind2sub(size(probMap), indices);
    v = depthMap(indices);
    [w1, w2, w3] = pixelCoordToWorldCoord(K, R, t, w(:), u(:), v, [], 0);          
    vertices = [vertices; w1 w2 w3];
end
% scatter3(vertices(:,1),vertices(:,2),vertices(:,3))
% axis equal
dlmwrite('pcMore.txt', vertices);
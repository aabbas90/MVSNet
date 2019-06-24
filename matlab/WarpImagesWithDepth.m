%% Set-up:
clear; clc; close all; fclose('all');
D = dir('../TEST_DATA_FOLDER/depths_mvsnet/');
refCam = 194;
refI = im2double(imread([D(refCam).folder, '\', D(refCam).name(1:end-11), '.jpg']));
depth = parsePfm([D(refCam).folder, '\', D(refCam).name(1:end-11), '.pfm']);
[R,t,K,dStart,dInt] = load_cam([D(refCam).folder, '\', D(refCam).name(1:end-11), '.txt']);
[N, P, c] = size(refI);
[u, w] = meshgrid(1:1:P, 1:1:N);
v = depth';
[w1, w2, w3] = pixelCoordToWorldCoord(K, R, t, w(:), u(:), v, [], 0);
for i = 202
    I = im2double(imread([D(i).folder, '\', D(i).name(1:end-11), '.jpg']));
    
    [R,t,K,dStart,dInt] = load_cam([D(i).folder, '\', D(i).name(1:end-11), '.txt']);
    [wp, up, Zp] = WorldCoordTopixelCoord(K, R, t, w1, w2, w3);
    warped = cat(3, interp2(I(:,:,1), up, wp, 'cubic', 0), ...
                    interp2(I(:,:,2), up, wp, 'cubic', 0),...
                    interp2(I(:,:,3), up, wp, 'cubic', 0));
    warpedReshape = reshape(warped, size(I));
%    depth = parsePfm([D(i).folder, '\', D(i).name(1:end-11), '.pfm']);
end
figure(1),imagesc(refI)
figure(2),imagesc(warpedReshape)


%% Add path:
addpath(genpath('../../npy-matlab-master/'));
addpath(genpath('../../minboundbox/'));
%% Set-up:
clear; clc; close all; fclose('all');
D = dir('../TEST_DATA_FOLDER/depths_mvsnet/');
thresh = 0.6;
%% Interp3 Check:
volTest = readNPY([D(26).folder, '\', D(26).name]);
[M, N, P] = size(volTest);
[uR, vR, wR] = meshgrid(1:1:N, 1:1:M, 1:1:P);
interpD = reshape(interp3(volTest, uR, vR, wR, 'linear'), size(volTest)); % 192, 128, 256
[p, d] = max(interpD, [], 1);
figure,imagesc(squeeze(d));

%% See merged depth;
referenceCam = 26;
[R,t,K,dStart,dInt] = load_cam([D(referenceCam).folder, '\', D(referenceCam).name(1:end-11), '.txt']);
mergedP = readNPY([D(referenceCam).folder, '\', D(referenceCam).name]);
[M, N, P] = size(mergedP);
dEnd = squeeze(dInt * (M - 1) + dStart);
[u, v, w] = meshgrid(1:1:N, dStart:dInt:dEnd, 1:1:P);
[w1, w2, w3] = pixelCoordToWorldCoord(K, R, t, w(:), u(:), v, [], 0);

neighbours = [referenceCam-8,referenceCam, referenceCam+8];
for i = 1:length(D)
    if ~contains(D(i).name, '.npy')
        continue;
    end
    [R,t,K,dStart,dInt] = load_cam([D(i).folder, '\', D(i).name(1:end-11), '.txt']);
    vol = readNPY([D(i).folder, '\', D(i).name]);
    [wp, up, Zp] = WorldCoordTopixelCoord(K, R, t, w1, w2, w3);
    vp = 1 + (Zp - dStart) / dInt;
    interpD = reshape(interp3(vol, up, vp, wp, 'cubic'), size(mergedP));
    interpD(interpD < 0.7) = 0;
    mergedP = mergedP+interpD;
end
[p, d] = max(mergedP, [], 1);
figure,imagesc(squeeze(d));
save('mergedProbs', 'mergedP');


%% Merge in world coordinates:
figure(1), hold on;
colors = eye(3);
for i = 1:length(D)
    if ~contains(D(i).name, '.npy')
        continue;
    end
    I = imread([D(i).folder, '\', D(i).name(1:end-11), '.jpg']);
    [R,t,K,dStart,dInt] = load_cam([D(i).folder, '\', D(i).name(1:end-11), '.txt']);
    vol = readNPY([D(i).folder, '\', D(i).name]);
    vol(:,1:5,:) = 0;
    vol(1:5,:,:) = 0;
    vol(:,:,1:5) = 0;
    [p, d] = max(vol, [], 1);
    p = squeeze(p);
    d = squeeze(dInt * (d - 1) + dStart);
    [M, N] = size(d);
    [u, v] = meshgrid(0.5:1:N-0.5, 0.5:1:M-0.5);
    [X, Y, Z] = pixelCoordToWorldCoord(K, R, t, u(:), v(:), d, p, 0);
    Ir = reshape(I, [M*N,3]);
    Ir(p < thresh, :) = [];
    scatter3(X(:),Y(:),-Z(:),5,i * ones(size(X(:)))), hold on;
end
%% Find bounding box around the point cloud and intialize voxel grid:
boundingPoints = []; % zeros(length(D) * 4, 3);
index = 1; SI = 1;
for i = 1:length(D)
    if ~contains(D(i).name, '.npy')
        continue;
    end
    I = imread([D(i).folder, '\', D(i).name(1:end-11), '.jpg']);
    [R,t,K,dStart,dInt] = load_cam([D(i).folder, '\', D(i).name(1:end-11), '.txt']);
    vol = readNPY([D(i).folder, '\', D(i).name]);
%    [numD, M, N] = size(vol);
%    [u, v] = meshgrid([0.5, N-0.5], [0.5, M-0.5]);
    [P, M, N] = size(vol);
    dEnd = squeeze(dInt * (P - 1) + dStart);
    [u, v, w] = meshgrid([1:M], [1,N], [dStart,dEnd]);
    [w1, w2, w3] = pixelCoordToWorldCoord(K, R, t, u(:), v(:), w, [], 0);
     boundingPoints = [boundingPoints; w1(:), w2(:), w3(:)];
%     for dIndex = 1:numD
%         d = repmat(dStart + (dIndex - 1)* dInt, size(u));
%         [X, Y, Z] = pixelCoordToWorldCoord(K, R, t, u(:), v(:), d, [], -Inf);
%         boundingPoints(index:index+length(X) - 1, :) = [X(:), Y(:), Z(:)];
%         index = index+length(X);
%     end
end
[Rbbox, bbox] = minboundbox(boundingPoints(:,1), boundingPoints(:,2), boundingPoints(:,3), 'v', 1);
bboxAligned = Rbbox' * bbox';
[w1, w2, w3] = meshgrid(min(bboxAligned(1,:)):SI:max(bboxAligned(1,:)), ...
                     min(bboxAligned(2,:)):SI:max(bboxAligned(2,:)), ...
                     min(bboxAligned(3,:)):SI:max(bboxAligned(3,:)));
mergedP = zeros(size(w1), 'single');
w1 = single(w1(:)); w2 = single(w2(:)); w3 = single(w3(:));
W = [w1'; w2'; w3'];
w1 = single(Rbbox(1,:) * W); w2 = single(Rbbox(2,:) * W); w3 = single(Rbbox(3,:) * W);
clear W;
%% Process:
fclose('all');
mergedP = zeros(size(mergedP));
map = hsv(3);
index = 1;
for i = 1:length(D)
    if ~contains(D(i).name, '.npy')
        continue;
    end
    [R,t,K,dStart,dInt] = load_cam([D(i).folder, '\', D(i).name(1:end-11), '.txt']);
    vol = readNPY([D(i).folder, '\', D(i).name]);
    vol(:,1:5,:) = 0;
    vol(1:10,:,:) = 0;
    vol(:,:,1:10) = 0;
    vol(vol < 0.9) = 0;
    [Xp, Yp, Zp] = WorldCoordTopixelCoord(K, R, t, w1, w2, w3);
    Zp = 1 + (Zp - dStart) / dInt;
    mergedP = mergedP + reshape(interp3(vol, Xp, Yp, Zp, 'cubic', 0), size(mergedP));
    clear Xp;
    clear Yp;
    index = index + 1;
end
save('mergedProbs', 'mergedP');

%% Visualize:
close all
figure(3)
%p1d = find(mergedP > 2);
%[x, y, z] = ind2sub(size(mergedP), p1d);
% scatter3(x(:),y(:),-z(:), 5);
% 
p = patch(isosurface(mergedP,0.5), 'FaceColor', 'Red');
isonormals(mergedP,p)
p.EdgeColor = 'none';
daspect([1 1 1])
view(3); 
axis tight
camlight 
lighting gouraud
axis equal
% hiso = patch(isosurface(mergedP,0.5));
% isonormals(mergedP,hiso)
% set(hiso,'FaceColor','red','EdgeColor','none');
%  
% hcap = patch(isocaps(mergedP,0.5),...
%      'FaceColor','interp',...
%      'EdgeColor','none');
%  colormap hsv
% 
% patch(isosurface(mergedP, 0.5));
%lighting phong
%set(hcap,'AmbientStrength',1.5)


%% Debug:
%% Debugg
% clc;
% close all;
% 
% m_1 = -21.7837; m_2 = 35.4461; m_3 = 485.0688;
% for i = 362:8:370
%     if ~contains(D(i).name, '.npy')
%         continue;
%     end
%     [R,t,K,dStart,dInt] = load_cam([D(i).folder, '\', D(i).name(1:end-11), '.txt']);
%     Img = imread([D(i).folder, '\', D(i).name(1:end-11), '.jpg']);
%     [Xp, Yp, Zp] = WorldCoordTopixelCoord(K, R, t, m_1, m_2, m_3);
%     Zp = 1 + (Zp - dStart) / dInt
%     Xp
%     Yp
%     figure(i), imagesc(Img);
% end
% 
% 
% %% Debug interp3:
% vol3 = readNPY([D(referenceCam).folder, '\', D(referenceCam).name]);
% [P, M, N] = size(vol3);
% dEnd = squeeze(dInt * (P - 1) + dStart);
% [u, v, w] = meshgrid(1:1:M, 1:1:N, 1:P);
% interpD = reshape(interp3(vol3, u, v, w), size(vol3));
% [p, d1] = max(vol3, [], 1);
% [p, d2] = max(interpD, [], 1);
% figure,imagesc([squeeze(d1), squeeze(d2)]);

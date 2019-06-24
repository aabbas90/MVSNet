%% Set-up:
clear; clc; close all; fclose('all');
D = dir('../TEST_DATA_FOLDER/depths_mvsnet/');
refCam = 194;
refI = im2double(imread([D(refCam).folder, '\', D(refCam).name(1:end-11), '.jpg']));
depth = parsePfm([D(refCam).folder, '\', D(refCam).name(1:end-11), '.pfm']);
[Rref,tref,Kref,dStart,dInt] = load_cam([D(refCam).folder, '\', D(refCam).name(1:end-11), '.txt']);
[N, P, c] = size(refI);
[u, w] = meshgrid(1:1:P, 1:1:N);
numD = 128;
dEnd = dStart + (numD - 1)* dInt;
costV = zeros(N, P, numD);
bestImage = zeros(size(refI));
for i = [186]
    I = im2double(imread([D(i).folder, '\', D(i).name(1:end-11), '.jpg']));
    [Ri,ti,Ki,dStarti,dInti] = load_cam([D(i).folder, '\', D(i).name(1:end-11), '.txt']);
    for d = 1:numD
        v = dStart + (d - 1) * ones(size(depth')) * dInt;
        currentD = dStart + (d - 1) * dInt;
%         [w1, w2, w3] = pixelCoordToWorldCoord(Kref, Rref, tref, u(:), w(:), v, [], 0);
%         [up, wp, Zp] = WorldCoordTopixelCoord(Ki, Ri, ti, w1, w2, w3);
        H = GetHomography(Rref, tref, Kref, Ri, ti, Ki, currentD);
        newP = H * [w(:)'; u(:)'; ones(size(u(:)'))];
        wp = newP(1,:) ./ newP(3,:);
        up = newP(2,:) ./ newP(3,:);
        warped = cat(3, interp2(I(:,:,1), up, wp, 'cubic', 0), ...
                        interp2(I(:,:,2), up, wp, 'cubic', 0),...
                        interp2(I(:,:,3), up, wp, 'cubic', 0));
        warpedReshape = reshape(warped, size(I));
        [pi, pj] = find(abs(depth - currentD) < dInt);
        for ind = 1:length(pi)
            bestImage(pi(ind), pj(ind), :) = warpedReshape(pi(ind), pj(ind), :);
        end
        
%         figure(2),imagesc([I, warpedReshape]);
%         drawnow;
%         pause
        costV(:,:,d) = costV(:,:,d) + convn(convn(sum((I - warpedReshape).^2, 3), ones(5,1), 'same'), ones(1,5), 'same');
    end
end
figure(3), imagesc([bestImage]);
figure(4), imagesc([I]);
% [minC, minD] = min(costV, [], 3);
% figure(1),imagesc([mat2gray(minC), mat2gray(minD)]);

function [Xp, Yp, Zp] = WorldCoordTopixelCoord(K, R, t, Xw, Yw, Zw)
C = t + R * [Xw(:)'; Yw(:)'; Zw(:)'];
p = K * C;
Zp = single(C(3, :)); % 1 + ((C(3, :) - depthStart) / depthInterval);
clear C(:);
Xp = single(p(1, :) ./ p(3, :));
Yp = single(p(2, :) ./ p(3, :));
end
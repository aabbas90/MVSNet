function [X, Y, Z] = pixelCoordToWorldCoord(K, R, t, u, v, d, prob, thresh)
d1 = d(:);
% for u:
a = K(3,1) * u - K(1,1);
b = K(3,2) * u - K(1,2);
c = d1 .* (K(1,3) - K(3,3) * u);

% for v:
g = K(3,1) * v - K(1,2);
h = K(3,2) * v - K(2,2);
l = d1 .* (K(2,3) - K(3,3) * v);


y = (l - (g .* c) ./ a) ./ (h - (g .* b) ./ a);
x = (c - b.*y) ./ a;
z = d1; % * ones(size(x));
% t = t;
C = [x(:)'; y(:)'; z(:)'];
W = inv(R) * (C - t);
X = W(1,:);
Y = W(2,:);
Z = W(3,:);
X = reshape(X, size(d));
Y = reshape(Y, size(d));
Z = reshape(Z, size(d));
X(prob < thresh) = [];
Y(prob < thresh) = [];
Z(prob < thresh) = [];
end
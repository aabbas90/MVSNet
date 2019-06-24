function volS = SampleVolume(vol, x1, x2, x3)
[M, N, P] = size(vol);
x1 = round(x1);
x2 = round(x2);
x3 = round(x3);

outOfBound = x1 < 1 | x1 >= M | x2 < 1 | x2 >= N | x3 < 1 | x3 >= P;
x1(outOfBound) = [];
x2(outOfBound) = [];
x3(outOfBound) = [];
ind1 = sub2ind(size(vol), x1, x2, x3);
volS = zeros(size(vol));
volS = vol(ind1);
end
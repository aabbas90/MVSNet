function [H] = GetHomography(R1, t1, K1, R2, t2, K2, d)
% H = K2 * R2 * (eye(3) - ((t1 - t2)*n1')/d)*R1'*inv(K1);
fronto_direction = R1(end, :);
c_left = -R1'*t1;
c_right = -R2'*t2;
c_relative = c_right - c_left;
temp_vec = c_relative * fronto_direction; %Is it a vec or mat? should be mat

middlemat0 = eye(3) - temp_vec / d;
middlemat1 = R1' * inv(K1);
middlemat2 = middlemat0 * middlemat1;

H = K2 * R2 * middlemat2;
end
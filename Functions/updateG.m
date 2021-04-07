function [G, T] = updateG(Hp, R, rho)
%updating variable G;
num = size(Hp, 1);
num_view = size(Hp, 3);
T = zeros(num, num, num_view);
for p = 1 : num_view
    T( : , : , p) = Hp( : , : , p) * Hp( : , : , p)';
end
tempT = shiftdim(T + 1 / rho * R, 1);
[tempG, ~, ~] = prox_tnn(tempT, 1 / rho);
G = shiftdim(tempG, 2);
end
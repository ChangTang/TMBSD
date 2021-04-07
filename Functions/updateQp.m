function Qp = updateQp(Hp, F)
%updating variable Qp;
k = size(Hp, 2);
num_view = size(Hp, 3);
Qp = zeros(k, k, num_view);
for p = 1 : num_view
    Tp = Hp( : , : , p)' * F;
    [Up, ~, Vp] = svd(Tp, 'econ');
    Qp( : , : , p) = Up * Vp';
end
function beta = updateBeta(Hp, Qp, F, qnorm)
%updating variable beta;
num_view = size(Qp, 3);
FHpQp = zeros(num_view, 1);
for  p = 1 : num_view
    FHpQp(p) = trace(F' * (Hp( : , : , p) * Qp( : , : , p)));
end
% beta = FHpQp./norm(FHpQp);
beta = FHpQp .^ (1 / (qnorm - 1)) / sum(FHpQp .^ (qnorm / (qnorm - 1))) ^ (1 / qnorm);
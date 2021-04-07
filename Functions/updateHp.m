function [Hp, Pp] = updateHp(Ap, Qp, F, S, beta, B, Hp, Pp, rho, lambda1, lamdba2)
%updating variable Hp;
num = size(F, 1);
num_view = size(Qp, 3);
opts = [];
opts.inF_o = 0;
opts.Kptol = 1e-5;
for p = 1 : num_view
    mis_indx = S{p}.indx';
    obs_indx = setdiff(1 : num, mis_indx);
    %%%% update missinKp part
    Bp_uo = 0.5 * (B(mis_indx, obs_indx, p) + B(obs_indx, mis_indx, p)' );
    Bp_uu = 0.5 * (B(mis_indx, mis_indx, p) + B(mis_indx, mis_indx, p)' );
    Pp_u = Pp(mis_indx, mis_indx, p);
    Pp_o = Pp(obs_indx, obs_indx, p);
    Hp_o = Hp(obs_indx, : , p);
    hatHp_o = Pp_o * Hp_o;
    Hp_u = Hp(mis_indx, : , p);
    hatHp_u = Pp_u * Hp_u;
    F_u = F(mis_indx, : );
    Jp = - rho / 2 * (Pp_u * (Bp_uu + Bp_uu') * Pp_u + Pp_u * (hatHp_u * hatHp_u') * Pp_u );
    Kp = - rho * Pp_u * Bp_uo * hatHp_o - lamdba2 * beta(p) * F_u * Qp( : , : , p)';
    if sum(isnan(Jp( : ))) > 0
        Jp = eye(size(Jp, 1));
    end
    if sum(isnan(Kp( : ))) > 0
        Kp = ones(size(Kp));
    end
    tX = Hp(mis_indx, : , p);
    if size(tX, 1) < size(tX, 2)
        Vp = F(mis_indx, : ) * Qp( : , : , p)';
        [Up, ~, Vp] = svd(Vp, 'econ');
        Hp(mis_indx, : , p) = Up * Vp';
    else
        [Hp_u, ~] = FOForth(tX, Kp, @fun, opts, Jp, Kp);
        Hp(mis_indx, : , p) = Hp_u;
    end
    %%%% update observed part
    Bp_ou = 0.5 * (B(mis_indx, obs_indx, p)' + B(obs_indx, mis_indx, p));
    Bp_oo = 0.5 * (B(obs_indx, obs_indx, p) + B(obs_indx, obs_indx, p)' );
    Kp_oo = Ap(obs_indx, obs_indx, p);
    F_o = F(obs_indx, : );
    Cp = - lambda1 * ((Kp_oo + Kp_oo') / 2 + 1e-8 * eye(length(obs_indx))) ...
         - rho / 2 * (Pp_o * (Bp_oo + Bp_oo') * Pp_o + Pp_o * (hatHp_o * hatHp_o') * Pp_o );
    Dp = - lamdba2 * beta(p) * F_o * Qp( : , : , p)' - rho * Pp_o * Bp_ou * hatHp_u;
    [Hp_o, ~] = FOForth(Hp(obs_indx, : , p), Dp, @fun, opts, Cp, Dp);
    Hp(obs_indx, : , p) = Hp_o;
    Pp( : , : , p) = diag(1 ./ sqrt(diag(Hp( : , : , p) * Hp( : , : , p)' + eps)));
end
    function [funX, F] = fun(X, A, Kp)
        F = 2 * A * X + Kp;
        funX = sum(sum(X .* (A * X + Kp)));
    end
end
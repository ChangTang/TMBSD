function [F_normalized, Qp, Hp, beta, diffTG] = TMBSD(Ap, S, k, qnorm, lambda1, lamdba2)
% Tensor-based Multi-view Block-diagonal Structure Diffusion for Clustering
% Incomplete Multi-view Data
%       min     L(Hp, F, Qp, beta, G) = -\lambda_1 \sum_p=1^V trace(Hp_o'
%       A_oo Hp_oo ) -\lambda_2 trace(F' \sum_p=1^V HpQp) + ||G||_* + <R,
%       T-G> + rho/2||T-G||_F^2 
%       s. t.         Hp_o'Hp_o = I_k, Hp_u'Hp_u =
%       I_k, Qp'Qp = I_k, F'F = I-k, beta >0, sum_p^V beta = 1.
% ----------------------------------
%  Input:
%       Ap --- normalized Laplacian matrix                                 
%       S ---indicators                                                    
%       k --- num of clusters                                               
%       qnorm ---- parameter
%       lambda1 and lambda2 ---- two balance parameters
%  Output:
%       F_normalized --- consensus clustering representation                                 
%       Qp --- rotation matrix
%       Hp --- spectral embedding matrx
%       beta --- view weight
%       diffTG --- error: ||G - T||_inf
% --------------------------------------------------------------------
%  Reference:
%   Zhenglai Li, Chang Tang, Xinwang Liu, Xiao Zheng,Wei Zhang, En Zhu.
%   Tensor-based Multi-view Block-diagonal Structure Diffusion for
%   Clustering Incomplete Multi-view Data. ICME 2021
% ----------------------------------
%  Author: Zhenglai Li (yuezhenguan@cug.edu.cn)
% --------------------------------------------------------------------
rho = 1e-4;
flag = 1;
iter = 0;
num = size(Ap, 2); %the number of samples
num_view = size(Ap, 3); %the number of views
maxIter = 150; %the number of iterations
[Hp, Qp, Pp] = myInitializationHp(Ap, S, k);
beta = ones(num_view, 1) * (1 / num_view) ^ (1 / qnorm);
%%%%%%%%%%
R = zeros(num, num, num_view);
G = zeros(num, num, num_view);
bHpQp = zeros(num, k);
for p = 1 : num_view
    bHpQp = bHpQp + beta(p) * (Hp( : , : , p) * Qp( : , : , p));
end
while (flag && iter < maxIter)
    iter = iter + 1;
    %optimize F
    [Uf, ~, Vf] = svd(bHpQp, 'econ');
    F = Uf * Vf';
    %optimize Qp
    Qp = updateQp(Hp, F);
    %optimize Hp
    B = G - 1 / rho * R;
    [Hp, Pp] = updateHp(Ap, Qp, F, S, beta, B, Hp, Pp, rho, lambda1, lamdba2);
    %optimize G
    [G, T] = updateG(Hp, R, rho);
    %update R and rho
    R = R + rho * (T - G);
    rho = min(rho * 1.2, 1e6);
    %optimize beta
    beta = updateBeta(Hp, Qp, F, qnorm);
    %calculate Obj
    bHpQp = zeros(num, k);
    for p = 1 : num_view
        bHpQp = bHpQp + beta(p) * Hp( : , : , p) * Qp( : , : , p);
    end
    diffTG(iter) = max(max(max(abs(G - T ))));
    if iter > 2 && diffTG(iter) < 1e-6
        flag = 0;
    end
end
F_normalized = F ./ repmat(sqrt(sum(F .^ 2, 2)), 1, k);
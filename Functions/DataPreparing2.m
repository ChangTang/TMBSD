function [X1, O1, X2, O2] = DataPreparing2(data, index, I)
% X{k} = X1{k}*O1{k} + X2{k}*O2{k}
% [D, N] = size(X{k}), [D, N_1] = size(X1{k}), [D, N_i] = size(X2{k}),
% [N_1, N] = size(O1{k}), [N_i, N] = size(O2{k})

K = length(data); %numOfView
N = size(data{1}, 2); %numOfSample
X1 = cell(K,1); %the complete parts
X2 = cell(K,1); %the missing parts
O1 = cell(K, 1);
O2 = cell(K, 1);
Xc = cell(K,1);
for k = 1:K
    data{k} = data{k}(:, I);
    W1 = ones(N, 1);
    W1(index{k}, 1) = 0;
    
    W1 = W1(I);
    
    ind_1 = W1 == 1;
    W2 = eye(N);
    W2(ind_1, :) = [];
    O1{k} = W2;
    
    W3 = zeros(N, 1);
    W3(index{k}, 1) = 1;
    
    W3 = W3(I);
    
    ind_2 = W3 == 1;
    W4 = eye(N);
    W4(ind_2, :) = [];
    O2{k} = W4;
    
    data{k} = double(data{k});
    data{k}(isnan(data{k})) = 0;
    Xc{k} = data{k} * O1{k}';
    [X1{k}] = NormalizeData(Xc{k});
    %     [Xc{k}] = NormalizeData(Xc{k});
    %     S = constructW_PKN(Xc{k});
    %     DN = diag( 1./sqrt(sum(S)+eps) );
    %     L = DN * S * DN;
    %     opt.disp = 0;
    %     [F, ~] = eigs(L, cls_num, 'la', opt);
    %     NorF = F ./ sqrt(sum(F.^2, 2));
    %     X1{k} = NorF';
    fillV = repmat(mean(X1{k}, 2), 1, size(O2{k}, 1));
    [X2{k}] = NormalizeData(fillV);
end
end
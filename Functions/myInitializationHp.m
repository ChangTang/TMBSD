function [HP,WP,PP] = myInitializationHp(KH,S,k)

numker = size(KH,3);
num = size(KH,1);
% %--Initializing HP and WP
HP = zeros(num,k,numker);
PP = zeros(num,num,numker);
WP = zeros(k,k,numker);
opt.disp = 0;
for p =1:numker
    %% missing index: S{p}.indx
    obs_indx = setdiff(1:num,S{p}.indx');
    KAp = KH(obs_indx,obs_indx,p);
    KAp = (KAp+KAp')/2+1e-8*eye(length(obs_indx));
    [Hp, ~] = eigs(KAp, k, 'la', opt);
    Pp = diag(1 ./ sqrt(diag(Hp*Hp')));
    PP(obs_indx,obs_indx,p) = Pp;
    HP(obs_indx,:,p) = Hp;
    WP(:,:,p) = eye(k);
end
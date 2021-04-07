%Tensor-based Multi-view Block-diagonal Structure Diffusion for Clustering Incomplete Multi-view Data
clear;
clc;
warning off
addpath('ClusteringMeasure');
addpath('Functions');
resultdir1 = 'Results/';
addpath(genpath('Results/'));
resultdir2 = 'maxResults/';
addpath(genpath('maxResults/'));
datadir = './Data/';
dataname = {'ORL'};
numdata = length(dataname); % number of the test datasets
numname = {'_Per0.1',  '_Per0.3',  '_Per0.5',  '_Per0.7',  '_Per0.9'};
for idata = 1 : 1
    ResBest = zeros(5, 8);
    ResStd = zeros(5, 8);
    % result = [Fscore Precision Recall nmi AR Entropy ACC Purity];
    for dataIndex = 1 : 5
        datafile = [datadir, cell2mat(dataname(idata)), cell2mat(numname(dataIndex)), '.mat'];
        %data preparation...
        load(datafile);
        Y = truelabel{1};
        [~, I] = sort(Y);
        M = idx2pm(I);
        if size(Y, 1) == 1
            Y = Y';
        end
        Y = Y(I);
        tic;
        [Xc, ind, O1] = DataPreparing(data, index, I);
        V = length(Xc);
        N = size(O1{1}, 2);
        Ap = zeros(N, N, V);
        S = cell(V, 1);
        for v = 1 : V
            tempW = constructW_PKN(Xc{v}, 15);
            DN = diag(1 ./ sqrt(sum(tempW) + eps));
            tempK = DN * tempW * DN;
            Ap( : , : , v) = O1{v}' * tempK * O1{v};
            S{v}.indx = [];
            S{v}.indx = find(ind( : , v)' == 0);
        end
        qnorm = 2;
        numclass = length(unique(Y));
        time1 = toc;
        maxAcc = 0;
        TempLambda1 = - 3 : 1 : 3;
        TempLambda2 = - 3 : 1 : 3;
        %         TempLambda1 = 2;
        %         TempLambda2 = 1;
        ACC = zeros(length(TempLambda1), length(TempLambda2));
        NMI = zeros(length(TempLambda1), length(TempLambda2));
        Purity = zeros(length(TempLambda1), length(TempLambda2));
        idx = 1;
        for LambdaIndex1 = 1 : length(TempLambda1)
            lambda1 = TempLambda1(LambdaIndex1);
            for LambdaIndex2 = 1 : length(TempLambda2)
                lambda2 = TempLambda2(LambdaIndex2);
                disp([char(dataname(idata)), char(numname(dataIndex)), '-l1=', num2str(lambda1), '-l2=', num2str(lambda2)]);
                tic;
                [F, WP, HP, beta, obj] = TMBSD(Ap, S, numclass, qnorm, 10 ^ lambda1, 10 ^ lambda2);
                time2 = toc;
                stream = RandStream.getGlobalStream;
                reset(stream);
                MAXiter = 100; % Maximum number of iterations for KMeans
                REPlic = 20; % Number of replications for KMeans
                tic;
                res = zeros(20, 8);
                for rep = 1 : 20
                    pY = kmeans(real(F), numclass, 'maxiter', MAXiter, 'replicates', REPlic, 'emptyaction', 'singleton');
                    res(rep, : ) = Clustering8Measure(Y, pY);
                end
                time3 = toc;
                runtime(idx) = time1 + time2 + time3 / 20;
                disp(['runtime:', num2str(runtime(idx))])
                idx = idx + 1;
                tempResBest(dataIndex, : ) = mean(res);
                tempResStd(dataIndex, : ) = std(res);
                ACC(LambdaIndex1, LambdaIndex2) = tempResBest(dataIndex, 7);
                NMI(LambdaIndex1, LambdaIndex2) = tempResBest(dataIndex, 4);
                Purity(LambdaIndex1, LambdaIndex2) = tempResBest(dataIndex, 8);
                save([resultdir1, char(dataname(idata)), char(numname(dataIndex)), '-l1=', num2str(lambda1), '-l2=', num2str(lambda2), ...
                    '-acc=', num2str(tempResBest(dataIndex, 7)), '_result.mat'], 'tempResBest', 'tempResStd');
                for tempIndex = 1 : 8
                    if tempResBest(dataIndex, tempIndex) > ResBest(dataIndex, tempIndex)
                        if tempIndex == 7
                            newF = F;
                            newObj = obj;
                            newHP = HP;
                            newBeta = beta;
                        end
                        ResBest(dataIndex, tempIndex) = tempResBest(dataIndex, tempIndex);
                        ResStd(dataIndex, tempIndex) = tempResStd(dataIndex, tempIndex);
                    end
                end
            end
        end
        aRuntime = mean(runtime);
        PResBest = ResBest(dataIndex, : );
        PResStd = ResStd(dataIndex, : );
        save([resultdir2, char(dataname(idata)), char(numname(dataIndex)), 'ACC_', num2str(max(ACC( : ))), '_result.mat'], 'ACC', 'NMI', 'Purity', 'aRuntime', ...
            'newF', 'newObj', 'newHP', 'newBeta', 'PResBest', 'PResStd', 'gt');
    end
    save([resultdir2, char(dataname(idata)), '_result.mat'], 'ResBest', 'ResStd');
end
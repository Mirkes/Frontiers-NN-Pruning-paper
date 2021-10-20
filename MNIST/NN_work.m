%% Add my utils
addpath('C:\localdata\local\App\Matlab\utils');

%% Load training set
[xTrainImages,tTrain] = digitTrainCellArrayData;
%% Load the test images
[xTestImages,tTest] = digitTestCellArrayData;
%% Convert data to appropriate format
xDigitTrain = zeros(28*28, size(xTrainImages, 2));
for k = 1:size(xTrainImages, 2)
    tmp = xTrainImages{k};
    xDigitTrain(:, k) = tmp(:);
end
xDigitTest = zeros(28*28, size(xTestImages, 2));
for k = 1:size(xTestImages, 2)
    tmp = xTestImages{k};
    xDigitTest(:, k) = tmp(:);
end
%% Save converted data
save('DigitDataSet.mat', 'xDigitTrain', 'tTrain', 'xDigitTest', 'tTest');

%% Load previously saved data
load('DigitDataSet.mat');

%% Convert to one data matrix
xDigit = [xDigitTrain, xDigitTest];
t = [tTrain, tTest];

%% Split data into three sets: 70% training, 15% validation and 15% test
[~, labels] = max(t);
labels = labels(:);
[trainInd, valInd, testInd] = divideClassRand(labels, 0.7, 0.15);
%% Test of correctness
[items, counts] = countOfItems(labels);
[itemsL, countsL] = countOfItems(labels(trainInd));
[itemsV, countsV] = countOfItems(labels(valInd));
[itemsT, countsT] = countOfItems(labels(testInd));

%% save data and indices of sets
save('DigitDataSetSplit.mat', 'xDigit', 't', 'trainInd', 'valInd', 'testInd');

%% Load data for NN
load('DigitDataSetSplit.mat');

%% Check the data for constant rows
mi = min(xDigit, [], 2);
ma = max(xDigit, [], 2);
const = sum(mi == ma);

%% Net creation
% Choose a Training Function
trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.
% Create a Pattern Recognition Network
hiddenLayerSize = 10;
net = patternnet(hiddenLayerSize, trainFcn);

% Choose Input and Output Pre/Post-Processing Functions
% For a list of all processing functions type: help nnprocess
%net.input.processFcns = {'removeconstantrows','mapminmax'};
net.input.processFcns = {'mapminmax'};

% Setup Division of Data for Training, Validation, Testing as defined
% before
net.divideFcn = 'divideind';
net.divideParam.trainInd = trainInd;
net.divideParam.valInd = valInd;
net.divideParam.testInd = testInd;

% Choose a Performance Function
net.performFcn = 'crossentropy';  % Cross-Entropy

% Choose Plot Functions
% For a list of all plot functions type: help nnplot
% net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
%     'plotconfusion', 'plotroc'};
net.plotFcns = {'ploterrhist', 'plotconfusion'};

% Train the Network
[net, tr] = train(net, xDigit, t);

% test of NN result
[~, labels] = max(t);
labels = labels(:);
y = net(xDigit);
[~, labHat] = max(y);
labHat = labHat(:);

C = confusionmat(labels(testInd), labHat(testInd));
acc = sum(diag(C)) / sum(C, 'all');

save('net10_2.mat', 'net', 'acc');

%% Rpeated calculation of networks
hiddenLayerSize = 10;
nRep = 100;
for k = 2:nRep
    % Net creation
    % Choose a Training Function
    trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.
    % Create a Pattern Recognition Network
    net = patternnet(hiddenLayerSize, trainFcn);
    
    % Choose Input and Output Pre/Post-Processing Functions
    % For a list of all processing functions type: help nnprocess
    %net.input.processFcns = {'removeconstantrows','mapminmax'};
    net.input.processFcns = {'mapminmax'};
    
    % Setup Division of Data for Training, Validation, Testing as defined
    % before
    net.divideFcn = 'divideind';
    net.divideParam.trainInd = trainInd;
    net.divideParam.valInd = valInd;
    net.divideParam.testInd = testInd;
    
    % Choose a Performance Function
    net.performFcn = 'crossentropy';  % Cross-Entropy
    
    % Choose Plot Functions
    % For a list of all plot functions type: help nnplot
    % net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
    %     'plotconfusion', 'plotroc'};
    net.plotFcns = {'ploterrhist', 'plotconfusion'};
    
    % Train the Network
    [net, tr] = train(net, xDigit, t);
    
    % test of NN result
    [~, labels] = max(t);
    labels = labels(:);
    y = net(xDigit);
    [~, labHat] = max(y);
    labHat = labHat(:);
    
    C = confusionmat(labels(testInd), labHat(testInd));
    acc = sum(diag(C)) / sum(C, 'all');
    
    save(['net', mat2str(hiddenLayerSize), '_', mat2str(k), '.mat'], 'net', 'acc');
end

%% Load accuracies of NN
files = dir('net10_*.mat');
% file by file
n = size(files, 1);
accs = zeros(n, 1);
for k = 1:n
    load(files(k).name, 'acc');
    accs(k) = acc;
end

%% Select the best network and load it
[acc, ind] = max(accs);
load(files(ind).name);

%% Create NN function
genFunction(net,'myNN','MatrixOnly','yes');

%% Test of generated function
y = net(xDigit);
yy = myNN(xDigit);
%%
dif = abs(y - yy);
diff = sum(dif(:) > 1.e-14);

%% Pruning
% To save weight decomment line 24 in myNN.m
% It is not good idea to save this line uncommented after saving weights
% matrix because it take time without any gain.
load('weights.mat');
%% original statistics
mi = min(IW1_1);
ma = max(IW1_1);
emptyInputs = find((mi == 0) & (ma == 0));

%% Pruning with 0 order approach.
% Tolerance in count
tol = 0;
% Load data for NN
load('DigitDataSetSplit.mat');
% load weights
load('weights.mat');
% Create copy
W = IW1_1;
% estimate accuracy of current net for validation set
[~, labels] = max(t);
y = myNN0(xDigit, W);
[~, lab] = max(y);
accInit = sum(labels(valInd) == lab(valInd)) - tol;
% Define order of weights
% Constant for zero substitution
zer = 2 * max(abs(W(:)));
W1 = abs(W(:));
W1(W1 == 0) = zer;
[~, ord] = sort(W1);
% Loop of pruning
while true
    % Copy current net
    oldW = W;
    % Prune weight
    W(ord(1)) = 0;
    fprintf("%d removing\n", sum(W == 0, 'all') - 550);
    % Evaluate performance
    y = myNN0(xDigit, W);
    [~, lab] = max(y);
    acc = sum(labels(valInd) == lab(valInd));
    if acc < accInit
        W = oldW;
        break;
    end
    ord(1) = [];
end
% Save results
save("weightsZero.mat", 'W');

%% Pruning with 0 order until the first input removing.
% Load data for NN
load('DigitDataSetSplit.mat');
% load weights
load('weights.mat');
% Create copy
W = IW1_1;
% estimate number of unused inputs
mi = min(IW1_1);
ma = max(IW1_1);
emptyNeurInit = sum((mi == 0) & (ma == 0));
% Define order of weights
% Constant for zero substitution
zer = 2 * max(abs(W(:)));
W1 = abs(W(:));
W1(W1 == 0) = zer;
[~, ord] = sort(W1);
% Loop of pruning
while true
    % Copy current net
    oldW = W;
    % Prune weight
    W(ord(1)) = 0;
    fprintf("%d removing\n", sum(W == 0, 'all') - 550);
    % Evaluate performance
    mi = min(W);
    ma = max(W);
    emptyNeur = sum((mi == 0) & (ma == 0));
    if emptyNeurInit < emptyNeur
        break;
    end
    ord(1) = [];
end
% Save results
save("weightsZeroNeur.mat", 'W');

%% Prepare graph of weights removing from the first layer
% Load data for NN
load('DigitDataSetSplit.mat');
% load weights
load('weights.mat');
% Create copy
W = IW1_1;
% Define order of weights
% Constant for zero substitution
zer = 2 * max(abs(W(:)));
W1 = abs(W(:));
tmp = sum(W1 == 0);
W1(W1 == 0) = zer;
[~, ord] = sort(W1);
% Remove indices of zero weights
ord(end - tmp + 1:end) = [];
% Create location for result
K = length(ord);
res = zeros(K + 1, 5);
% Prepare true labels
[~, labels] = max(t);
% Test original nerwork
y = myNN0(xDigit, IW1_1);
% Estimate number of unused inputs
mi = min(IW1_1);
ma = max(IW1_1);
res(1, 1) = sum((mi == 0) & (ma == 0));
res(1, 2:5) = assess(labels, y, testInd, valInd, trainInd);
% Loop of pruning
for k = 1:K
    % Prune weight
    W(ord(k)) = 0;
    fprintf("%d removing\n", k);
    % Evaluate performance
    mi = min(W);
    ma = max(W);
    res(k + 1, 1) = sum((mi == 0) & (ma == 0));
    y = myNN0(xDigit, W);
    res(k + 1, 2:5) = assess(labels, y, testInd, valInd, trainInd);
end
% Normalise data
res = res ./ [size(W, 2), length(trainInd), length(valInd), length(testInd), length(t)];
% Save results
save("pruningZero.mat", 'res');
% figure
figure;
plot(res);
legend('Removed inputs', 'Training', 'Validation', 'Test', 'Total', 'Location', 'west');
saveFigures('figures/zeroPrunungInpMean.png');

%% Zero order mean input signal remover
% Load data for NN
load('DigitDataSetSplit.mat');
% load weights
load('weights.mat');
% Create copy
W = IW1_1;
% Define order of weights
% Constant for zero substitution
zer = 2 * max(abs(W(:)));
W1 = abs(W);
W1 = sum(W1);
zer = 2 * max(W1);
tmp = sum(W1 == 0);
W1(W1 == 0) = zer;
[~, ord] = sort(W1);
% Remove indices of zero weights
ord(end - tmp + 1:end) = [];
% Create location for result
K = length(ord);
res = zeros(K + 1, 6);
% Prepare true labels
[~, labels] = max(t);
% Test original nerwork
y = myNN0(xDigit, IW1_1);
% Estimate number of unused inputs
mi = min(IW1_1);
ma = max(IW1_1);
res(1, 1) = sum((mi == 0) & (ma == 0));
res(1, 2:5) = assess(labels, y, testInd, valInd, trainInd);
% Loop of pruning
for k = 1:K
    % Prune weight
    W(:, ord(k)) = 0;
    res(k, 6) = ord(k);
    fprintf("%d removing\n", k);
    % Evaluate performance
    mi = min(W);
    ma = max(W);
    res(k + 1, 1) = sum((mi == 0) & (ma == 0));
    y = myNN0(xDigit, W);
    res(k + 1, 2:5) = assess(labels, y, testInd, valInd, trainInd);
end
% Normalise data
res = res ./ [size(W, 2), length(trainInd), length(valInd), length(testInd), length(t), 1];
% Save results
save("pruningZeroIntMean.mat", 'res');
%% Figure
figure;
plot(res(:, 1: 5));
legend('Removed inputs', 'Training', 'Validation', 'Test', 'Total', 'Location', 'east');
xlabel('Number of removed inputs');
ylabel('Accuracy/fraction of removed inputs');
title('Zero order based on mean absolute value');
saveFigures('figures/zeroPrunungInpMean.png');

%% Zero order max input signal remover
% Load data for NN
load('DigitDataSetSplit.mat');
% load weights
load('weights.mat');
% Create copy
W = IW1_1;
% Define order of weights
% Constant for zero substitution
zer = 2 * max(abs(W(:)));
W1 = abs(W);
W1 = max(W1);
zer = 2 * max(W1);
tmp = sum(W1 == 0);
W1(W1 == 0) = zer;
[~, ord] = sort(W1);
% Remove indices of zero weights
ord(end - tmp + 1:end) = [];
% Create location for result
K = length(ord);
res = zeros(K + 1, 6);
% Prepare true labels
[~, labels] = max(t);
% Test original nerwork
y = myNN0(xDigit, IW1_1);
% Estimate number of unused inputs
mi = min(IW1_1);
ma = max(IW1_1);
res(1, 1) = sum((mi == 0) & (ma == 0));
res(1, 2:5) = assess(labels, y, testInd, valInd, trainInd);
% Loop of pruning
for k = 1:K
    % Prune weight
    W(:, ord(k)) = 0;
    fprintf("%d removing\n", k);
    % Evaluate performance
    mi = min(W);
    ma = max(W);
    res(k, 6) = ord(k);
    res(k + 1, 1) = sum((mi == 0) & (ma == 0));
    y = myNN0(xDigit, W);
    res(k + 1, 2:5) = assess(labels, y, testInd, valInd, trainInd);
end
% Normalise data
res = res ./ [size(W, 2), length(trainInd), length(valInd), length(testInd), length(t), 1];
% Save results
save("pruningZeroIntMax.mat", 'res');
%% Figure
figure;
plot(res(:, 1:5));
legend('Removed inputs', 'Training', 'Validation', 'Test', 'Total', 'Location', 'east');
xlabel('Number of removed inputs');
ylabel('Accuracy/fraction of removed inputs');
title('Zero order based on maximal absolute value');
saveFigures('figures/zeroPrunungInpMax.png');

%% Recheck validation set and check training set
y = myNN0(xDigit, W);
[~, lab] = max(y);
val = sum(labels(valInd) == lab(valInd)) / length(valInd);
test  = sum(labels(testInd) == lab(testInd)) / length(testInd);
y = myNN0(xDigit, IW1_1);
[~, lab] = max(y);
testInit  = sum(labels(testInd) == lab(testInd)) / length(testInd);
%% Calcuate nuber of removed neurons
mi = min(IW1_1);
ma = max(IW1_1);
emptyNeurInit = sum((mi == 0) & (ma == 0));
mi = min(W);
ma = max(W);
emptyNeur = sum((mi == 0) & (ma == 0));

%First order fragment
%% neuron removing
% Load data for NN
load('DigitDataSetSplit.mat');
% load weights
load('weights.mat');
% Create copy
W = IW1_1;
% Estimate number of unused inputs
mi = min(IW1_1);
ma = max(IW1_1);
zeroInp = sum((mi == 0) & (ma == 0));
% estimate number of inputs to remove
K = size(W, 2) - zeroInp;
% Create array for results
res = zeros(K + 1, 6);
res(1, 1) = zeroInp;
% Prepare true labels
[~, labels] = max(t);
for k = 1:K
    fprintf("%d removing\n", k);
    % Calculate sensitivity indicators
    [~, sensIndic] = myNNDif(xDigit(:, trainInd), t(:, trainInd), W);
    % Calculate accuracies
    y = myNN0(xDigit, W);
    res(k, 2:5) = assess(labels, y, testInd, valInd, trainInd);
    res(k + 1, 1) = res(k, 1) + 1;
    % Search the best indicator
    ma = 2 * max(sensIndic);
    sensIndic(sensIndic == 0) = ma;
    [mi, ind] = min(sensIndic);
    res(k, 6) = ind;
    W(:, ind) = 0;
end
[y, sensIndic] = myNNDif(xDigit, t, W);
res(k + 1, 2:5) = assess(labels, y, testInd, valInd, trainInd);
res = res ./ [size(W, 2), length(trainInd), length(valInd), length(testInd), length(t), 1];
save("pruningFirstInt.mat", 'res');
%% Figure
figure;
plot(res(:, 1:5));
legend('Removed inputs', 'Training', 'Validation', 'Test', 'Total', 'Location', 'east');
xlabel('Number of removed inputs');
ylabel('Accuracy/fraction of removed inputs');
title('First order');
saveFigures('figures/firstPrunungInp.png');

%% Final figure
figure;
load('pruningFIrstInt.mat');
plot(res(:, 4));
hold on;
load('pruningZeroIntMean.mat');
plot(res(:, 4));
load('pruningZero.mat');
ind = res(1:end - 1, 1) ~= res(2: end, 1);
tmp = [res(1, 4); res(ind, 4)];
plot(tmp);
xlabel('Number of pruned inputs');
ylabel('Test set accuracy');
legend('First order impute pruning', 'Zero order mean pruning', 'Zero order weights pruning', 'Location', 'north');
saveFigures('figures/DigitsPruning.png');
saveFigures('figures/DigitsPruning.eps');
saveFigures('figures/DigitsPruning.pdf');

%% functions
function accs = assess(labels, y, testInd, valInd, trainInd)
    [~, lab] = max(y);
    accs = zeros(1, 4);
    accs(1) = sum(labels(trainInd) == lab(trainInd));
    accs(2) = sum(labels(valInd) == lab(valInd));
    accs(3) = sum(labels(testInd) == lab(testInd));
    accs(4) = sum(labels == lab);
end




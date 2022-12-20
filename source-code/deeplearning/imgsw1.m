%% Load and Explore Image Data
digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
    'nndatasets','DigitDataset');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

figure;
perm = randperm(10000,20);
for i = 1:20
    subplot(4,5,i);
    imshow(imds.Files{perm(i)});
end

labelCount = countEachLabel(imds)

img = readimage(imds,1);
size(img)
%% Specify Training and Validation Sets
numTrainFiles = 750;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');
%% Define Network Architecture
layers = [
    imageInputLayer([28 28 1])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];
%% Specify Training Options
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',12, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');
%% Train Network Using Training Data
net = trainNetwork(imdsTrain,layers,options);
%% Classify Validation Images and Compute Accuracy
YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)
%% Classify Images Using Trained Convolutional Neural Network
digitDatasetPath = fullfile(toolboxdir("nnet"),"nndemos","nndatasets","DigitDataset");
imdsTest = imageDatastore(digitDatasetPath,IncludeSubfolders=true);

YTest = classify(net,imdsTest);

numImages = 9;
idx = randperm(numel(imdsTest.Files),numImages);

figure
tiledlayout("flow")
for i = 1:numImages
    nexttile
    imshow(imdsTest.Files{idx(i)});
    title("Predicted Label: " + string(YTest(idx(i))))
end
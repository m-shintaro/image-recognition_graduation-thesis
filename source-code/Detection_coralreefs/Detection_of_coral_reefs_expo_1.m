
% pool = parpool("LocalProfile1") 
%% step1:グラウンド トゥルース データの読み込み
data = load('expo_5.mat'); %グラウンド トゥルース データの読み込み
gTruth = data.gTruth.DataSource.Source; %画像が入っているパス
%% step2:学習用のlayerGraphオブジェクトを含む検出器の読み込み
%vehicleDetector = load('yolov2VehicleDetector.mat'); %YOLOv2オブジェクト検出ネットワークをロード
%lgraph = vehicleDetector.lgraph %layerGraphオブジェクトの読み込み
%% グラウンド トゥルース オブジェクトを使用し、イメージ データストアとボックス ラベル データストアを作成
[imds,bxds] = objectDetectorTrainingData(data.gTruth);
%% データストアを統合
%cds = combine(imds,bxds);



%% 学習済みモデル
network=resnet50();
%% 特徴抽出として使うレイヤーを指定
featureLayer = 'avtivatuin_40_relu';
%% 入力画像サイズ
imageSize = network.Layers(1).InputSize;
%% クラス数
numClasses= width(trainingDataset)-1;
%% YOLO v2物体検出ネットワークを定義
lgraph = yolov2Layers(imageSize,numClasses,round(anchorBoxes),network,fratureLayer);
%%　物体検出ネットワークの可視化
analyzeNetwork(lgraph);
%% 学習オプションの設定
options = trainingOptions('sgdm', ...
       'InitialLearnRate', 1e-3, ...
       'Verbose',true, ...
       'MiniBatchSize',16, ...
       'MaxEpochs',250, ...
       'ValidationPatience',3, ...
       'Shuffle','every-epoch', ...
           'ValidationFrequency',30, ...
               'Plots','training-progress',...
       'VerboseFrequency',10); 
%% 検出器の学習
[detector,info] = trainYOLOv2ObjectDetector(trainNetwork,lgraph,options);
%% イメージの読み込み

I = imread('reef11.png');
P = imread('reef104.png');
%% 検出器を実行
[bboxes,scores] = detect(detector,I);
%% 結果を表示
if(~isempty(bboxes))
  I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
end
figure
imshow(I)

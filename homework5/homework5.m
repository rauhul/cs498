%% homework 5
%% 1
clear; close all; clc
data = csvread('default_plus_chromatic_features_1059_tracks.txt');

% a
% set up data
Y_1 = data(:,117);
Y_2 = data(:,118);
X = horzcat(ones(size(data(:, 1))), data(:, 1:116));

clear data;

% regress 1
[b,~,~,~,~] = mvregress(X, Y_1);

pred = (b' * X')';
clear b;
figure;
scatter(Y_1 - pred, pred);
r2_1 = sum((pred - mean(Y_1)).^2) / sum((Y_1 - mean(Y_1)).^2);

% regress 2
[b,~,~,~,~] = mvregress(X, Y_2);

pred = (b' * X')';
clear b;

figure;
scatter(Y_2 - pred, pred);
r2_2 = sum((pred - mean(Y_2)).^2) / sum((Y_2 - mean(Y_2)).^2);

% b
% set up data box
Y_1_box = boxcox(Y_1 + 360);
Y_2_box = boxcox(Y_2 + 360);

% regress 3
[b,~,~,~,~] = mvregress(X, Y_1_box);

pred = (b' * X')';
clear b;
figure;
scatter(Y_1_box - pred, pred);
r2_3 = sum((pred - mean(Y_1_box)).^2) / sum((Y_1_box - mean(Y_1_box)).^2);

% regress 4
[b,~,~,~,~] = mvregress(X, Y_2_box);

pred = (b' * X')';
clear b;
figure;
scatter(Y_2_box - pred, pred);
r2_4 = sum((pred - mean(Y_2_box)).^2) / sum((Y_2_box - mean(Y_2_box)).^2);

%% c
options = glmnetSet();
options.alpha = 1;
fit = glmnet(X, Y_1, [], options);
fit.dim
% figure;
% glmnetPlot(fit);
pred = glmnetPredict(fit, X, 0);
r2_5 = sum((pred - mean(Y_1)).^2) / sum((Y_1 - mean(Y_1)).^2);

options = glmnetSet();
options.alpha = 1;
fit = glmnet(X, Y_2, [], options);
fit.dim
% figure;
% glmnetPlot(fit);
pred = glmnetPredict(fit, X, 0);
r2_6 = sum((pred - mean(Y_2)).^2) / sum((Y_2 - mean(Y_2)).^2);

options = glmnetSet();
options.alpha = 0;
fit = glmnet(X, Y_1, [], options);
fit.dim
% figure;
% glmnetPlot(fit);
pred = glmnetPredict(fit, X);
r2_7 = sum((pred - mean(Y_1)).^2) / sum((Y_1 - mean(Y_1)).^2);

options = glmnetSet();
options.alpha = 0;
fit = glmnet(X, Y_2, [], options);
fit.dim
% figure;
% glmnetPlot(fit);
pred = glmnetPredict(fit, X, 0);
r2_8 = sum((pred - mean(Y_2)).^2) / sum((Y_2 - mean(Y_2)).^2);

clear fit options pred;

%% 2
clear; close all; clc
data = csvread('default_of_credit_card_clients.txt');

X = data(:, 1:23);
Y = data(:, 24);
clear data;

[B,FitInfo] = lassoglm(X,Y,'binomial',...
    'NumLambda',25,'CV',10,'Alpha',1);

indx = FitInfo.Index1SE;
B0 = B(:,indx);
nonzeros = sum(B0 ~= 0); 

B1 = [FitInfo.Intercept(indx); B0];
preds = glmval(B1,X,'logit');

Y_pred = preds > 0.5;
Y_correct = Y_pred == Y;
percent_correct = sum(Y_correct)/length(Y_correct);

[B,FitInfo] = lassoglm(X,Y,'binomial',...
    'NumLambda',25,'CV',10,'Alpha',0.5);

indx = FitInfo.Index1SE;
B0 = B(:,indx);
nonzeros = sum(B0 ~= 0); 

B1 = [FitInfo.Intercept(indx); B0];
preds = glmval(B1,X,'logit');

Y_pred = preds > 0.5;
Y_correct = Y_pred == Y;
percent_correct = sum(Y_correct)/length(Y_correct);

%% 3
clear; close all; clc
X = csvread('I2000.txt')';
Y = csvread('I2000_labels.txt') > 0;

[B,FitInfo] = lassoglm(X,Y,'binomial',...
    'NumLambda',25,'CV',10,'Alpha',1);

indx = FitInfo.Index1SE;
B0 = B(:,indx);
nonzeros = sum(B0 ~= 0); 

B1 = [FitInfo.Intercept(indx); B0];
preds = glmval(B1,X,'logit');

Y_pred = preds > 0.5;
Y_correct = Y_pred == Y;
percent_correct = sum(Y_correct)/length(Y_correct);

fit = cvglmnet(X, Y, [], [], 'auc');



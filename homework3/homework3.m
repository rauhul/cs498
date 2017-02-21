%% homework 3 rev 2
% load data
close all; clear; clc;
load('/Users/rauhul/Developer/cs498/homework3/data/data_batch_1.mat');
data_1 = data;
labels_1 = labels;

load('/Users/rauhul/Developer/cs498/homework3/data/data_batch_2.mat');
data_2 = data;
labels_2 = labels;

load('/Users/rauhul/Developer/cs498/homework3/data/data_batch_3.mat');
data_3 = data;
labels_3 = labels;

load('/Users/rauhul/Developer/cs498/homework3/data/data_batch_4.mat');
data_4 = data;
labels_4 = labels;

load('/Users/rauhul/Developer/cs498/homework3/data/data_batch_5.mat');
data_5 = data;
labels_5 = labels;

clear data labels batch_label;

data   = vertcat(data_1,   data_2,   data_3,   data_4,   data_5);
labels = vertcat(labels_1, labels_2, labels_3, labels_4, labels_5);

data = double(data);

clear data_1   data_2   data_3   data_4   data_5;
clear labels_1 labels_2 labels_3 labels_4 labels_5;

data_10 = zeros(10, 5000, 3072);
counts = zeros(10,  1);

for idx = 1:50000 
    label = labels(idx) + 1;
    count = counts(label);
    data_10(label, count + 1, :) = data(idx, :);
    counts(label) = counts(label) + 1;
end

clear data labels counts count label idx;

%% a) PCA
errs_a = zeros(10, 1);
mus    = zeros(10, 3072);
coeffs = zeros(10, 3072, 3072);

for label = 1:10
    data = squeeze(data_10(label, :, :));
    [coeff,~,~,~,explained,mu] = pca(data, 'NumComponents', 20);
    mus(label, :) = squeeze(mu);
    errs_a(label) = 100 - sum(explained(1:20));
    coeffs(label, :, 1:20) = coeff;
end

figure;
bar(errs_a, 0.4, 'r');
labels = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'};
set(gca,'xticklabel', labels);

for label = 1:10
    figure; 
    imshow(reshape(uint8(mus(label, :)), [32 32 3]));
end

clear data explained mu label coeff labels;

%% b) PCoA
dissimilarities = pdist(mus);

[transformed,~,~] = mdscale(dissimilarities,2);

figure;
scatter(transformed(:, 1), transformed(:, 2))
labels = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'};
text(transformed(:,1), transformed(:,2), labels, 'horizontal','left', 'vertical','bottom')

clear dissimilarities transformed labels;

%% c) PCoA 2
errs_c = zeros(10, 10);
count  = 5000;

for a = 1:10
    mu_a = mus(a, :)';
    for b = 1:10
        if a ~= b
            coeff_b = squeeze(coeffs(b, :, :));

            for idx = 1:count
                x_a  = squeeze(data_10(a, idx, :));
                x_ab = pca_ipca(x_a, coeff_b, mu_a);
                d = x_a - x_ab;
                error = d' * d;
                errs_c(a, b) = errs_c(a, b) + error/count;
            end 
        end
    end
end

clear count mu_a coeff_b idx x_a x_ab d error;

for a = 1:10
    for b = a+1:10
        error = (1/2) * (errs_c(a, b) + errs_c(b, a));
        errs_c(a, b) = error;
        errs_c(b, a) = 0;
    end
end

clear error a b;

dissimilarities = pdist(errs_c);

[transformed,~,~] = mdscale(dissimilarities,2);

figure;
scatter(transformed(:, 1), transformed(:, 2))
labels = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'};
text(transformed(:,1), transformed(:,2), labels, 'horizontal','left', 'vertical','bottom')

clear dissimilarities transformed labels;

%% end
clear coeffs data_10 mus
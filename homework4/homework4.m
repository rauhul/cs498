%% homework 4
%% 7.9
clear; close all; clc

load('/Users/rauhul/Developer/cs498/homework4/brunhild.txt');

figure;

% a)
subplot(2,1,1);
scatter(log(brunhild(:, 1)), log(brunhild(:, 2)));
title('Log-Log Plot of Brunhilda Blood Sulfate Levels, with linear regression');
ylabel('Log( Sulfate )');
xlabel('Log( Hours )');
lsline;

% b)
subplot(2,1,2);
scatter(brunhild(:, 1), brunhild(:, 2));
title('Plot of Brunhilda Blood Sulfate Levels, with linear regression');
ylabel('Sulfate');
xlabel('Hours');
lsline;

% c)
log_coeff = polyfit(log(brunhild(:, 1)), log(brunhild(:, 2)), 1);
log_resid = log(brunhild(:, 2)) - ((log(brunhild(:, 1)) .* log_coeff(1)) + log_coeff(2));
coeff = polyfit(brunhild(:, 1), brunhild(:, 2), 1);
resid = brunhild(:, 2) - ((brunhild(:, 1) .* coeff(1)) + coeff(2));

figure;
subplot(2,1,1);
scatter(log(brunhild(:, 1)), log_resid);
title('Log-Log Residuals of Brunhilda Blood Sulfate Levels');
ylabel('Log( Sulfate )');
xlabel('Log( Hours )');
lsline;

subplot(2,1,2);
scatter(brunhild(:, 1), resid);
title('Residuals of Brunhilda Blood Sulfate Levels');
ylabel('Sulfate');
xlabel('Hours');
lsline;

%% 7.10


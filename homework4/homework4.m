%% homework 4
%% 7.9
clear; close all; clc

tdfread('brunhild.txt','\t');

figure;

% a)
subplot(2,1,1);
scatter(log(Hours), log(Sulfate));
title('Log-Log Plot of Brunhilda Blood Sulfate Levels, with linear regression');
ylabel('Log( Sulfate )');
xlabel('Log( Hours )');
lsline;

% b)
subplot(2,1,2);
scatter(Hours, Sulfate);
title('Plot of Brunhilda Blood Sulfate Levels, with linear regression');
ylabel('Sulfate');
xlabel('Hours');
lsline;

% c)
log_coeff = polyfit(log(Hours), log(Sulfate), 1);
log_resid = log(Sulfate) - ((log(Hours) .* log_coeff(1)) + log_coeff(2));
coeff = polyfit(Hours, Sulfate, 1);
resid = Sulfate - ((Hours .* coeff(1)) + coeff(2));

figure;
subplot(2,1,1);
scatter(log(Hours), log_resid);
title('Log-Log Residuals of Brunhilda Blood Sulfate Levels');
ylabel('Log( Sulfate )');
xlabel('Log( Hours )');
lsline;

subplot(2,1,2);
scatter(Hours, resid);
title('Residuals of Brunhilda Blood Sulfate Levels');
ylabel('Sulfate');
xlabel('Hours');
lsline;

%% 7.10
clear; close all; clc

tdfread('physical.txt','\t');


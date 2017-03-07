%% homework 4
%% 7.9
clear; close all; clc
tdfread('brunhild.txt','\t');

b         = polyfit(log(Hours), log(Sulfate), 1);
resid     = Sulfate - exp(((log(Hours) .* b(1)) + b(2)));
log_resid = log(Sulfate) - ((log(Hours) .* b(1)) + b(2));

figure;
% a)
subplot(2,1,1);
hold;
scatter(log(Hours), log(Sulfate));              % data points
plot(log(Hours), (log(Hours) .* b(1)) + b(2));  % regression line
hold;
title('Log-Log Plot of Brunhilda Blood Sulfate Levels, with regression line');
ylabel('Log( Sulfate )');
xlabel('Log( Hours )');

% b)
subplot(2,1,2);
hold;
scatter(Hours, Sulfate);                        % data points
plot(Hours, exp((log(Hours) .* b(1)) + b(2)));  % regression line
hold;
title('Plot of Brunhilda Blood Sulfate Levels, with regression curve');
ylabel('Sulfate');
xlabel('Hours');

% c)
figure;
subplot(2,1,1);
scatter((log(Hours) .* b(1)) + b(2), log_resid);
title('Log-Log Residuals of Brunhilda Blood Sulfate Levels Linear Regression vs Fitted Values');
ylabel('Residual');
xlabel('Log( Y est )');

subplot(2,1,2);
scatter(exp((log(Hours) .* b(1)) + b(2)), resid);
title('Residuals of Brunhilda Blood Sulfate Levels Linear Regression vs Fitted Values');
ylabel('Residual');
xlabel('Y est');

%d) See Report

%% 7.10
clear; close all; clc
tdfread('physical.txt','\t');

Y =  Mass;
X = [ones(size(Y)), Bicep, Calf, Chest, Fore, Head, Height, Neck, Shoulder, Thigh, Waist];
clear Bicep Calf Chest Fore Head Height Neck Shoulder Thigh Waist Mass;

% a)
[b,~,~,~,~] = mvregress(X, Y);

figure;
scatter(X * b, Y - (X * b))
title('Residuals of Mass & Physical Measurements Linear Regression vs Fitted Values');
ylabel('Residual');
xlabel('Y est');

% b)
croot_Y     = nthroot(Y, 3);
[b,~,~,~,~] = mvregress(X, croot_Y);
croot_resid = croot_Y - (X * b);
resid       = croot_resid .^ 3;

figure;
subplot(2,1,1);
scatter(X * b, croot_resid)
title('Cube Root Residuals of Mass & Physical Measurements Cube Root Linear Regression vs Fitted Values');
ylabel('Residual');
xlabel('croot( Y est )');

subplot(2,1,2);
scatter(X * b, resid)
title('Residuals of Mass & Physical Measurements Cube Root Linear Regression vs Fitted Values');
ylabel('Residual');
xlabel('Y est');

%c) See Report

%% 7.11
clear; close all; clc
tdfread('abalone.txt',',');

L = length(Diameter);
Sex_num = zeros(size(Sex));
for index = 1:L
    switch Sex(index)
        case 'M'
            Sex_num(index) =  1;
        case 'F'
            Sex_num(index) = -1;
        case 'I'
            Sex_num(index) =  0;
        otherwise
            error('bad data');
    end
end

clear Sex L index;

%a)
X = [ones(size(Diameter)), Diameter, Height, Length, Shell_weight, Shucked_weight, Viscera_weight, Whole_weight];
Y = Rings + 1.5;
cvfit_a = cvglmnet(X, Y);

[b,~,~,~,~] = mvregress(X, Y);

figure;
scatter(X * b, Y - (X * b))
title('Residuals of Abalone Age vs Non-sex Attributes Linear Regression vs Fitted Values');
ylabel('Residual');
xlabel('Y est');

clear X Y b;

% b)
X = [ones(size(Diameter)), Diameter, Height, Length, Sex_num, Shell_weight, Shucked_weight, Viscera_weight, Whole_weight];
Y = Rings + 1.5;
cvfit_b = cvglmnet(X, Y);

[b,~,~,~,~] = mvregress(X, Y);

figure;
scatter(X * b, Y - (X * b))
title('Residuals of Abalone Age vs Other Attributes Linear Regression vs Fitted Values');
ylabel('Residual');
xlabel('Y est');

clear X Y b;

% c)
X = [ones(size(Diameter)), Diameter, Height, Length, Shell_weight, Shucked_weight, Viscera_weight, Whole_weight];
Y = log(Rings + 1.5);
cvfit_c = cvglmnet(X, Y);

[b,~,~,~,~] = mvregress(X, Y);

figure;
scatter(X * b, Y - (X * b))
title('Residuals of Abalone Log Age vs Non-sex Attributes Linear Regression vs Fitted Values');
ylabel('Residual');
xlabel('Y est');

clear X Y b;

% d)
X = [ones(size(Diameter)), Diameter, Height, Length, Sex_num, Shell_weight, Shucked_weight, Viscera_weight, Whole_weight];
Y = Rings + 1.5;
cvfit_d = cvglmnet(X, Y);

[b,~,~,~,~] = mvregress(X, Y);

figure;
scatter(X * b, Y - (X * b))
title('Residuals of Abalone Log Age vs Other Attributes Linear Regression vs Fitted Values');
ylabel('Residual');
xlabel('Y est');

clear X Y b;

clear Diameter Height Length Rings Sex_num Shell_weight Shucked_weight Viscera_weight Whole_weight;

% e) See Report

% f)
cvglmnetPlot(cvfit_a);
cvglmnetPlot(cvfit_b);
cvglmnetPlot(cvfit_c);
cvglmnetPlot(cvfit_d);

clear cvfit_a cvfit_b cvfit_c cvfit_d;
function machine = train_rbm(X, h, type, eta, max_iter)
%TRAIN_RBM Trains a Restricted Boltzmann Machine using contrastive divergence
%
%   machine = train_rbm(X, h, type, eta, max_iter)
%
% Trains a first-order Restricted Boltzmann Machine on dataset X. The RBM
% has h hidden nodes (default = 20). The training is performed by means of
% the contrastive divergence algorithm. The activation function that
% is applied in the hidden layer is specified by type. Possible values are
% 'linear' and 'sigmoid' (default = 'sigmoid'). In the training of the RBM,
% the learning rate is determined by eta (default = 0.25). The maximum 
% number of iterations can be specified through max_iter (default = 50). 
% The trained RBM is returned in the machine struct.
%
% A Boltzmann Machine is a graphical model which in which every node is
% connected to all other nodes, except to itself. The nodes are binary,
% i.e., they have either value -1 or 1. The model is similar to Hopfield
% networks, except for that its nodes are stochastic, using a logistic
% distribution. It can be shown that the Boltzmann Machine can be trained by
% means of an extremely simple update rule. However, training is in
% practice not feasible.
%
% In a Restricted Boltzmann Machine, the nodes are separated into visible
% and hidden nodes. The visible nodes are not connected to each other, and
% neither are the hidden nodes. When training an RBM, the same update rule
% can be used, however, the data is now clamped onto the visible nodes.
% This training procedure is called contrastive divergence. Alternatively, 
% the visible nodes may be Gaussians instead of binary logistic nodes.
%

% This file is part of the Matlab Toolbox for Dimensionality Reduction v0.4b.
% The toolbox can be obtained from http://www.cs.unimaas.nl/l.vandermaaten
% You are free to use, change, or redistribute this code in any way you
% want for non-commercial purposes. However, it is appreciated if you 
% maintain the name of the original author.
%
% (C) Laurens van der Maaten
% Maastricht University, 2007


    % Process inputs
    if ~exist('h', 'var') || isempty(h)
        h = 20;
    end
    if ~exist('type', 'var') || isempty(type)
        type = 'sigmoid';
    end
    if ~exist('eta', 'var') || isempty(eta)
        eta = 0.1;
    end
    if ~exist('max_iter', 'var') || isempty(max_iter)
        max_iter = 100;
    end
    
    % Important parameters
    initial_momentum = 0.5;     % momentum for first five iterations
    final_momentum = 0.9;       % momentum for remaining iterations
    weight_cost = 0.0002;       % costs of weight update
    if strcmp(type, 'sigmoid')
        sigmoid = true;
    else
        sigmoid = false;
    end
    
    % Initialize some variables
    [n, v] = size(X);
    batch_size = 1 + round(n / 20);
    W = randn(v, h) * 0.1;
    bias_upW = zeros(1, h);
    bias_downW = zeros(1, v);
    deltaW = zeros(v, h);
    deltaBias_upW = zeros(1, h);
    deltaBias_downW = zeros(1, v);
    
    % Main loop
    for iter=1:max_iter
                
        % Print progress
        if rem(iter, 10) == 0
            disp(['Iteration ' num2str(iter) '...']);
        end
        
        % Set momentum
        if iter <= 5
            momentum = initial_momentum;
        else
            momentum = final_momentum;
        end
        
        % Run for all mini-batches (= Gibbs sampling step 0)
        ind = randperm(n);
        for batch=1:batch_size:n
            
            if batch + batch_size <= n
            
                % Set values of visible nodes (= Gibbs sampling step 0)
                vis1 = double(X(ind(batch:min([batch + batch_size - 1 n])),:));

                % Compute probabilities for hidden nodes (= Gibbs sampling step 0)
                if sigmoid
                    hid1 = 1 ./ (1 + exp(-(vis1 * W + repmat(bias_upW, [batch_size 1]))));
                else
                    hid1 = vis1 * W + repmat(bias_upW, [batch_size 1]);
                end

                % Compute probabilities for visible nodes (= Gibbs sampling step 1)
                vis2 = 1 ./ (1 + exp(-(hid1 * W' + repmat(bias_downW, [batch_size 1]))));

                % Compute probabilities for hidden nodes (= Gibbs sampling step 1)
                if sigmoid
                    hid2 = 1 ./ (1 + exp(-(vis2 * W + repmat(bias_upW, [batch_size 1]))));
                else
                    hid2 = vis2 * W + repmat(bias_upW, [batch_size 1]);
                end

                % Now compute the weights update (= contrastive divergence)
                posprods = hid1' * vis1;
                negprods = hid2' * vis2;
                deltaW = momentum * deltaW + (eta / batch_size) * ((posprods - negprods)' - (weight_cost * W));
                deltaBias_upW   = momentum * deltaBias_upW   + (eta / batch_size) * (sum(hid1, 1) - sum(hid2, 1));
                deltaBias_downW = momentum * deltaBias_downW + (eta / batch_size) * (sum(vis1, 1) - sum(vis2, 1));
                
                % Divide by number of elements for linear activations
                if ~sigmoid
                    deltaW = deltaW                   ./ numel(deltaW);
                    deltaBias_upW = deltaBias_upW     ./ numel(deltaBias_upW);
                    deltaBias_downW = deltaBias_downW ./ numel(deltaBias_downW);
                end
                
                % Update the network weights
                W           = W          + deltaW;
                bias_upW    = bias_upW   + deltaBias_upW;
                bias_downW  = bias_downW + deltaBias_downW;
            end
        end
    end
    
    % Return RBM
    machine.W = W;
    machine.bias_upW = bias_upW;
    machine.bias_downW = bias_downW;
    machine.type = type;

    machine.tied = 'yes';
    disp(' ');
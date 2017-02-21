function [ y ] = pca_ipca( x, coeff, mu )
%PCA_IPCA Summary of this function goes here
%   Detailed explanation goes here
    y = coeff * ( coeff' * (x - mu) ) + mu;
end


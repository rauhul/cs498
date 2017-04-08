% 1) EM Topic models The UCI Machine Learning dataset repository hosts 
%   several datasets recording word counts for documents here. You will use
%   the NIPS dataset. You will find (a) a table of word counts per document
%   and (b) a vocabulary list for this dataset at the link. You must
%   implement the multinomial mixture of topics model, lectured in class.
%   For this problem, you should write the clustering code yourself (i.e. 
%   not use a package for clustering).
clc; clear; close all;
data = load('docword.nips.txt');
num_docs  = data(1,1);
num_words = data(1,2);

docs_2d = zeros(num_docs, num_words);
for idx = 2:length(data)
    datum = data(idx,:);
    docs_2d(datum(1), datum(2)) = datum(3);
end
sparse_docs = sparse(docs_2d);

num_topics = 30;
topics = rand(num_topics, num_words);
magnitude = sum(topics, 2);
topics = topics ./ magnitude;

pis = rand(num_topics, 1);
magnitude = sum(pis, 1);
pis = pis ./ magnitude;

clear data datum documents_2d idx magnitude;

% a) Cluster this to 30 topics, using a simple mixture of multinomial topic
%   model, as lectured in class.

num_steps = 3;
w_per_doc = zeros(num_topics, num_docs);

w_per_doc_m_doc = zeros(num_topics, num_docs, num_words);

topics_start = topics; 
pis_start    = pis; 

for step = 1:num_steps 
    parfor idx = 1:num_docs
        doc = sparse_docs(idx, :);
        w_per_doc(:, idx) = w_from(doc, topics, pis);
        w_per_doc_m_doc(:, idx, :) = w_per_doc(:, idx) * doc;
    end
    
    pis = sum(w_per_doc, 2) / num_docs;
    topics = squeeze(sum(w_per_doc_m_doc, 2));
    magnitude = sum(topics, 2);
    topics = topics ./ magnitude;
end

topics_end = topics; 
pis_end    = pis; 

% b) Produce a graph showing, for each topic, the probability with which
%   the topic is selected.


% c) Produce a table showing, for each topic, the 10 words with the highest
%   probability for that topic.

function w = w_from(doc, topics, pis)
    w = pis;
    parfor idx = 1:length(pis)
        w(idx) = w(idx) * logsumexp(topics(idx, :) .^ doc, 2);        
    end
    w = w / sum(w);
end


function s = logsumexp(a, dim)
    % Returns log(sum(exp(a),dim)) while avoiding numerical underflow.
    % Default is dim = 1 (columns).
    % logsumexp(a, 2) will sum across rows instead of columns.
    % Unlike matlab's "sum", it will not switch the summing direction
    % if you provide a row vector.

    % Written by Tom Minka
    % (c) Microsoft Corporation. All rights reserved.

    if nargin < 2
      dim = 1;
    end

    % subtract the largest in each column
    [y, ~] = max(a,[],dim);
    dims = ones(1,ndims(a));
    dims(dim) = size(a,dim);
    a = a - repmat(y, dims);
    s = y + log(sum(exp(a),dim));
    i = find(~isfinite(y));
    if ~isempty(i)
      s(i) = y(i);
    end
end

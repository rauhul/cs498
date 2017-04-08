function w = w_from(doc, topics, pis)
    w = pis;
    for idx = 1:length(pis)
        w(idx) = w(idx) * logsumexp(topics(idx, :) .^ doc, 2);        
    end
    w = w / sum(w);
end
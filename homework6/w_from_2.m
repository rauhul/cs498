function w = w_from_2( pix, segments, pis )
    w = pis;
    for idx = 1:length(pis)
        temp = pix - segments(idx);
        w(idx) = w(idx) * exp( (-0.5) * (temp * temp') );
    end
    w = w / sum(w);
end


//
//  naive_bayes.hpp
//  homework2
//
//  Created by Rauhul Varma on 2/5/17.
//  Copyright © 2017 cs498. All rights reserved.
//

#ifndef naive_bayes_hpp
#define naive_bayes_hpp

#include <stdio.h>
#include <vector>

using std::vector;

class NaiveBayes {


    vector<long double> label_0_means;
    vector<long double> label_0_sds;

    vector<long double> label_1_means;
    vector<long double> label_1_sds;

    long double training_log_prob_0;
    long double training_log_prob_1;

public:
    NaiveBayes();
    ~NaiveBayes();

    void trainModel(const vector<vector<long double>>& features,
                    const vector<int>& labels, const vector<size_t>& idxs);

    void testModel(const vector<vector<long double>>& features,
                   const vector<int>& labels, const vector<size_t>& idxs,
                   long double& accuracy) const;

private:
    void calculateMeans(const vector<vector<long double>>& features,
                        const vector<int>& labels, const vector<size_t>& idxs);

    void calculateSDs(const vector<vector<long double>>& features,
                      const vector<int>& labels, const vector<size_t>& idxs);

    void reset();

};



#endif /* naive_bayes_hpp */

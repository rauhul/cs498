//
//  support_vector_machine.hpp
//  homework2
//
//  Created by Rauhul Varma on 2/6/17.
//  Copyright © 2017 cs498. All rights reserved.
//

#ifndef support_vector_machine_hpp
#define support_vector_machine_hpp

#include <stdio.h>
#include <vector>

using std::vector;

class SVM {
    
    vector<long double> aleph;
    long double b = 0.0;
    long double lambda;

public:
    SVM();
    ~SVM();
    
    void trainModel(const vector<vector<long double>>& features,
                    const vector<int>& labels,
                    const vector<size_t>& idxs);
    
    long double testModel(const vector<vector<long double>>& features,
                          const vector<int>& labels,
                          const vector<size_t>& idxs) const;

private:
    void reset();

    long double dotProduct(const vector<long double>& a,
                           const vector<long double>& b) const;
    
    void gradientDescent(vector<long double>& aleph,
                         long double& b,
                         const vector<long double>& feature,
                         long double eta,
                         int label,
                         long double lambda) const;
    
    long double cost(const vector<long double>& aleph,
                     const long double b,
                     const vector<vector<long double>>& features,
                     const vector<int>& labels,
                     const vector<size_t> idxs,
                     const long double lambda) const;
};

#endif /* support_vector_machine_hpp */

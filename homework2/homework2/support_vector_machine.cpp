//
//  support_vector_machine.cpp
//  homework2
//
//  Created by Rauhul Varma on 2/6/17.
//  Copyright © 2017 cs498. All rights reserved.
//

#include <iostream>
#include <vector>
#include <assert.h>

#include "data_helper.hpp"
#include "support_vector_machine.hpp"

using std::vector;

SVM::SVM() { }

SVM::~SVM() {
    reset();
}

// Training
void SVM::reset() {
    aleph.clear();
    b = 0;
    lambda = 0;
}

void SVM::trainModel(const vector<vector<long double>>& features,
                     const vector<int>& labels,
                     const vector<size_t>& idxs) {
    reset();
    assert(features.size() == labels.size());
    assert(features.size() > 0 && idxs.size() > 0);
    
    srand((uint)time(NULL));
    int NUM_SEASONS = 100;
    int NUM_STEPS = 420;
    
    //Min is used to keep track across all lambdas what the lowest cost across a U vector is
    long double min_cost = 10000000000; //Should never be this high
    
    //NEED TO A RANGE OF LAMBDAS FROM 0 TO 0.5; ANYTHING HIGHER IS NOT PRACTICAL FOR A LEARNING RATE
    //PICKING A LAMBDA FROM A SET DESCRIBED IN PROCEDURE 3.1 ON PG 29
    for(long double test_lambda = 0.05; test_lambda < 0.5; test_lambda += 0.1){
        
        vector<size_t> lambda_training_idxs;
        vector<size_t> lambda_validation_idxs;
        DataHelper::randSegmentIndices(idxs, lambda_training_idxs, lambda_validation_idxs, 0.15);
        assert(lambda_training_idxs.size() + lambda_validation_idxs.size() == idxs.size());
        
        vector<long double> test_aleph(features[0].size());                     //The A in the U vector used in SGD
        long double test_b = 20 * ((double) rand() / RAND_MAX) - 10;            //The b in the U vector used in SGD
        
        for(uint32_t i = 0; i < aleph.size(); ++i){
            test_aleph[i] = 20*((double) rand() / RAND_MAX) - 10; //Range from -10 to 10, as initial values for the a vector in the U vector
        }
        
        //Run through the seasons. After each season use best possible candidate from the season on test data
        //Pg 29 in Procedure
        for(uint32_t season = 0; season < NUM_SEASONS; ++season){
            //Numbers right now borrowed from Forsyth's book
            long double eta = (long double)(1.0/(50 + 0.01*season));
            
            //Step through the season, modifying aleph and b to find the best local minima for U vector to test
            for(uint16_t step = 0; step < NUM_STEPS; ++step) {
                size_t rand_lambda_training_idx = rand() % lambda_training_idxs.size();
                size_t idx = lambda_training_idxs[rand_lambda_training_idx];
                
                SVM::gradientDescent(test_aleph, test_b, features[idx], eta, labels[idx], test_lambda);
            }
            
        }
        
        long double cost = SVM::cost(test_aleph, test_b, features, labels, lambda_validation_idxs, test_lambda);
        
        if (cost < min_cost) {
            min_cost = cost;
            aleph   = test_aleph;
            b       = test_b;
            lambda  = test_lambda;
        }
    }
}


long double SVM::dotProduct(const vector<long double>& a,
                            const vector<long double>& b) const {
    assert(a.size() == b.size());
    long double dot_product = 0;
    for(size_t i = 0; i < a.size(); i++){
        dot_product += a[i] * b[i];
    }
    return dot_product;
}

//MATHEMATICS DESCRIBED IN PG 30 IN THE BOOK
void SVM::gradientDescent(vector<long double>& aleph,
                          long double& b,
                          const vector<long double>& feature,
                          long double eta,
                          int label,
                          long double lambda) const {
    
    bool cost = label * (dotProduct(aleph, feature) + b) >= 1;
    for(size_t i = 0; i < aleph.size(); i++) {
        if (cost) {
            aleph[i] -= eta * aleph[i];
        } else {
            aleph[i] -= lambda * aleph[i] - label * feature[i];
        }
    }
    b -= eta * (cost ? 0 : -1 * label);
}


//COST FUNCTION DESCRIBED IN PG 26 OF BOOK
long double SVM::cost(const vector<long double>& aleph,
                      const long double b,
                      const vector<vector<long double>>& features,
                      const vector<int>& labels,
                      const vector<size_t> idxs,
                      const long double lambda) const {
    
    long double aleph_magnitude = dotProduct(aleph, aleph);
    
    long double cost = 0;
    
    for(size_t idx: idxs) {
        int label = labels[idx];
        vector<long double> feature = features[idx];
    
        long double temp = 1 - label * SVM::dotProduct(aleph, feature);
        cost += (temp > 0) ? temp : 0;
    }
    
    return (cost/idxs.size()) + (lambda * aleph_magnitude / 2);
}


// Testing
long double SVM::testModel(const vector<vector<long double>>& features,
                           const vector<int>& labels,
                           const vector<size_t>& idxs) const {
    
    assert(features.size() == labels.size());
    assert(features.size() > 0 && idxs.size() > 0);
    
    long double accuracy = 0;
    for (size_t idx: idxs) {
        int label = labels[idx];
        vector<long double> feature = features[idx];
        
        long double temp = dotProduct(aleph, feature) + b;
        int guess = temp > 0 ? 1 : -1;
        
        if (guess == label) {
            accuracy++;
        }
    }
    
    return accuracy * 100 / idxs.size();
}


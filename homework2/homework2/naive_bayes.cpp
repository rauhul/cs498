//
//  naive_bayes.cpp
//  homework2
//
//  Created by Rauhul Varma on 2/5/17.
//  Copyright © 2017 cs498. All rights reserved.
//

#include "naive_bayes.hpp"
#include <iostream>
#include <assert.h>
#include <math.h>
#include <limits>

NaiveBayes::NaiveBayes() { }

NaiveBayes::~NaiveBayes() {
    reset();
}

// Training
void NaiveBayes::reset() {
    label_0_means.clear();
    label_0_sds.clear();

    label_1_means.clear();
    label_1_sds.clear();

    training_log_prob_0 = 0;
    training_log_prob_1 = 0;
}

void NaiveBayes::trainModel(const vector<vector<long double>>& features,
                            const vector<int>& labels,
                            const vector<size_t>& idxs) {
    reset();
    assert(features.size() == labels.size());
    assert(features.size() > 0 && idxs.size() > 0);

    size_t num_dimensions = features[0].size();
    
    for (int i = 0; i < num_dimensions; i++) {
        label_0_means.push_back(0);
        label_0_sds.push_back(0);
        label_1_means.push_back(0);
        label_1_sds.push_back(0);
    }

    NaiveBayes::calculateMeans(features, labels, idxs);
    NaiveBayes::calculateSDs(features, labels, idxs);
}


void NaiveBayes::calculateMeans(const vector<vector<long double>>& features,
                                const vector<int>& labels,
                                const vector<size_t>& idxs) {
    size_t num_dimensions = features[0].size();

    long double num_0_examples = 0;
    long double num_1_examples = 0;

    // sum each dimension per label
    for (size_t idx: idxs) {
        for (size_t dim = 0; dim < num_dimensions; dim++) {
            if (labels[idx] == 0) {
                label_0_means[dim] += features[idx][dim];
            } else if (labels[idx] == 1) {
                label_1_means[dim] += features[idx][dim];
            } else {
                assert(false);
            }
        }
        if (labels[idx] == 0) {
            num_0_examples++;
        } else if (labels[idx] == 1) {
            num_1_examples++;
        } else {
            assert(false);
        }
    }

    // check non-zero number of examples for each label
    assert(num_0_examples != 0);
    assert(num_1_examples != 0);
    assert(num_0_examples + num_1_examples == idxs.size());

    // divide by number of examples
    for (size_t dim = 0; dim < num_dimensions; dim++) {
        label_0_means[dim] /= num_0_examples;
        label_1_means[dim] /= num_1_examples;
    }

    long double sum = num_0_examples + num_1_examples;
    training_log_prob_0 = logbl(num_0_examples / sum);
    training_log_prob_1 = logbl(num_1_examples / sum);

}

void NaiveBayes::calculateSDs(const vector<vector<long double>>& features,
                              const vector<int>& labels,
                              const vector<size_t>& idxs) {
    size_t num_dimensions = features[0].size();

    long double num_0_examples = 0;
    long double num_1_examples = 0;

    // sum square deviation from the mean of each dimension per label
    for (size_t idx: idxs) {
        for (size_t dim = 0; dim < num_dimensions; dim++) {
            if (labels[idx] == 0) {
                long double deviation = features[idx][dim] - label_0_means[dim];
                long double deviation_2 = deviation*deviation;
                label_0_sds[dim] += deviation_2;
            } else if (labels[idx] == 1) {
                long double deviation = features[idx][dim] - label_1_means[dim];
                long double deviation_2 = deviation*deviation;
                label_1_sds[dim] += deviation_2;
            } else {
                assert(false);
            }
        }
        if (labels[idx] == 0) {
            num_0_examples++;
        } else if (labels[idx] == 1) {
            num_1_examples++;
        } else {
            assert(false);
        }
    }

    // check non-zero number of examples for each label
    assert(num_0_examples != 0);
    assert(num_1_examples != 0);
    assert(num_0_examples + num_1_examples == idxs.size());

    // sqrt & divide by number of examples
    for (size_t dim = 0; dim < num_dimensions; dim++) {
        label_0_sds[dim] = sqrt(label_0_sds[dim] / num_0_examples);
        label_1_sds[dim] = sqrt(label_1_sds[dim] / num_1_examples);
    }
}

// Testing

void NaiveBayes::testModel(const vector<vector<long double>>& features,
                           const vector<int>& labels,
                           const vector<size_t>& idxs,
                           long double& accuracy) const {

    assert(features.size() == labels.size());
    assert(features.size() > 0 && idxs.size() > 0);

    size_t num_dimensions = features[0].size();
    assert(num_dimensions == label_0_means.size());

    long double num_correct = 0;
    long double num_incorrect = 0;

    for (size_t idx: idxs) {
        long double log_prob_0 = training_log_prob_0;
        long double log_prob_1 = training_log_prob_1;

        for (size_t dim = 0; dim < num_dimensions; dim++) {
            long double deviation_0 = features[idx][dim] - label_0_means[dim];
            long double deviation_0_2 = deviation_0*deviation_0;
            long double sd_0 = label_0_sds[dim];
            long double sd_0_2 = sd_0*sd_0;

            long double deviation_1 = features[idx][dim] - label_1_means[dim];
            long double deviation_1_2 = deviation_1*deviation_1;
            long double sd_1 = label_1_sds[dim];
            long double sd_1_2 = sd_1*sd_1;

            if (sd_0 == 0 || sd_1 == 0) {
                continue;
            }
            
            log_prob_0 -= deviation_0_2/(2*sd_0_2);
            log_prob_0 -= logbl(sd_0);

            log_prob_1 -= deviation_1_2/(2*sd_1_2);
            log_prob_1 -= logbl(sd_1);
        }
        
        int guess = log_prob_1 > log_prob_0 ? 1 : 0;

        if (labels[idx] == guess) {
            num_correct++;
        } else {
            num_incorrect++;
        }

    }

    assert(num_correct + num_incorrect == idxs.size());
    accuracy = num_correct * 100 / (num_correct + num_incorrect);
}

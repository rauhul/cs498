//
//  main.cpp
//  homework2
//
//  Created by Rauhul Varma on 2/4/17.
//  Copyright © 2017 cs498. All rights reserved.
//

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <stdlib.h>
#include <assert.h>

#include "data_helper.hpp"
#include "naive_bayes.hpp"
#include "support_vector_machine.hpp"

using namespace std;

void testNaiveBayes(const vector<vector<long double>>& features,
                    const vector<int>& labels,
                    const vector<size_t> training_idxs,
                    const vector<size_t> testing_idxs) {
    
    NaiveBayes naive_bayes_model;
    naive_bayes_model.trainModel(features, labels, training_idxs);
    
    long double naive_bayes_training_accuracy = naive_bayes_model.testModel(features, labels, training_idxs);
    long double naive_bayes_testing_accuracy  = naive_bayes_model.testModel(features, labels, testing_idxs);
    
    cout << "Naive-Bayes training accuracy: " << naive_bayes_training_accuracy << endl;
    cout << "Naive-Bayes testing accuracy: "  << naive_bayes_testing_accuracy  << endl;
}


void testSVM(const vector<vector<long double>>& features,
             const vector<int>& labels,
             const vector<size_t> training_idxs,
             const vector<size_t> testing_idxs) {

    SVM svm_model;
    svm_model.trainModel(features, labels, training_idxs);
    
    long double svm_training_accuracy = svm_model.testModel(features, labels, training_idxs);
    long double svm_testing_accuracy  = svm_model.testModel(features, labels, testing_idxs);
    
    cout << "SVM training accuracy: " << svm_training_accuracy << endl;
    cout << "SVM testing accuracy: "  << svm_testing_accuracy  << endl;
}

int main(int argc, const char * argv[]) {
    // MARK: load file
    ifstream data_file;
    data_file.open("/Users/rauhul/Developer/cs498/homework2/homework2/K9.data", ifstream::in);
    assert(data_file.is_open());
    
    // MARK: parse file
    vector<vector<long double>> features;
    vector<int> labels;
    DataHelper::parseCSV(data_file, features, labels, 31420, 5409);
    data_file.close();
    assert(labels.size() == features.size());
    
    // MARK: split testing/training
    vector<size_t> idxs;
    for (size_t i = 0; i < labels.size(); i++) {
        idxs.push_back(i);
    }
    
    vector<size_t> training_idxs;
    vector<size_t> testing_idxs;
    DataHelper::randSegmentIndices(idxs, training_idxs, testing_idxs, 0.1);
    assert(training_idxs.size() + testing_idxs.size() == labels.size());

    // MARK: test with both classifiers
    testNaiveBayes(features, labels, training_idxs, testing_idxs);
    testSVM(features, labels, training_idxs, testing_idxs);
    
    return 0;
}

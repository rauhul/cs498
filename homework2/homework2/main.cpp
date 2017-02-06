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

using namespace std;

int main(int argc, const char * argv[]) {

    // MARK: load file
    ifstream data_file;
    data_file.open("/Users/rauhul/Developer/cs498/homework2/homework2/K9.data", ifstream::in);
    assert(data_file.is_open());

    // MARK: parse file
    vector<int> labels;
    vector<vector<long double>> features;
    DataHelper::parseCSV(data_file, labels, features, 31420, 5409);
    data_file.close();
    size_t count = labels.size();
    assert(count == features.size());

    // MARK: split testing/training
    vector<size_t> idxs;
    for (size_t i = 0; i < count; i++) {
        idxs.push_back(i);
    }

    vector<size_t> training_idxs;
    vector<size_t> testing_idxs;
    DataHelper::randSegmentIndices(idxs, training_idxs, testing_idxs, 0.1);
    assert(training_idxs.size() + testing_idxs.size() == count);

    // MARK: naive-bayes
    NaiveBayes naive_bayes_model;
    naive_bayes_model.trainModel(features, labels, training_idxs);

    long double naive_bayes_training_accuracy;
    long double naive_bayes_testing_accuracy;
    naive_bayes_model.testModel(features, labels, training_idxs, naive_bayes_training_accuracy);
    naive_bayes_model.testModel(features, labels, testing_idxs,  naive_bayes_testing_accuracy);

    cout << "Naive-Bayes training accuracy: " << naive_bayes_training_accuracy << endl;
    cout << "Naive-Bayes testing accuracy: "  << naive_bayes_testing_accuracy  << endl;

    return 0;
}



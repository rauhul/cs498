//
//  data_helper.cpp
//  homework2
//
//  Created by Rauhul Varma on 2/5/17.
//  Copyright © 2017 cs498. All rights reserved.
//

#include "data_helper.hpp"

#include <iostream>
#include <fstream>
#include <string>
#include <assert.h>

using std::string;
using std::cout;
using std::endl;

void DataHelper::randSegmentIndices(vector<size_t> idxs, vector<size_t> &a_idxs, vector<size_t> &b_idxs, long double partition) {
    assert(partition > 0 && partition < 1);
    srand((uint)time(NULL));

    size_t spilt_size = idxs.size() * partition;

    size_t indices_remaining;
    while ((indices_remaining = idxs.size()) && indices_remaining > spilt_size) {
        size_t idx = rand() % indices_remaining;
        size_t value = idxs[idx];

        idxs[idx] = idxs.back();
        idxs.pop_back();

        a_idxs.push_back(value);
    }
    
    b_idxs = idxs;
}

void DataHelper::parseCSV(ifstream &data_file, vector<vector<long double>> &features, vector<int> &labels, int rows, int traits) {
    string token;
    char *token_end;
    for (int row = 0; row < rows; row++) {
        vector<long double> line;
        bool include = true;
        for (int feature = 0; feature < traits; feature++) {
            assert(getline(data_file, token, ','));

            if (feature != traits - 1) {
                long double value = std::strtold(token.c_str(), &token_end);
                if (token.c_str() != token_end && *token_end == '\0') {
                    line.push_back(value);
                } else {
                    include = false;
                }
            } else {
                if (token == "inactive") {
                    if (include) {
                        labels.push_back(-1);
                    }
                } else if (token == "active") {
                    if (include) {
                        labels.push_back(1);
                    }
                } else {
                    include = false;
                }

                if (include) {
                    features.push_back(line);
                }
            }
        }

        if (row%1000 == 0) {
            int progress = row * 100/rows;
            cout << "progress: " << progress << "%" << endl;
        }
    }
    cout << "progress: " << 100 << "%" << endl;
}

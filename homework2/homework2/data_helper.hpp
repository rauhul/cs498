//
//  data_helper.hpp
//  homework2
//
//  Created by Rauhul Varma on 2/5/17.
//  Copyright © 2017 cs498. All rights reserved.
//

#ifndef data_helper_hpp
#define data_helper_hpp

#include <stdio.h>
#include <vector>
#include <fstream>

using std::vector;
using std::ifstream;

class DataHelper {

public:
    static void randSegmentIndices(vector<size_t> idxs, vector<size_t> &a_idxs, vector<size_t> &b_idxs, long double partition);
    static void randSegmentIndices(vector<size_t>& a_idxs, vector<size_t>& b_idxs, long double partition);

    static void parseCSV(ifstream &data_file, vector<vector<long double>> &features, vector<int> &labels, int rows, int traits);

};


#endif /* data_helper_hpp */

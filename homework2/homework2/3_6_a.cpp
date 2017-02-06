#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <random>
#include <algorithm>
#include <iterator>
#include <time.h>
#include <vector>
#include <stdint.h>

using namespace std;

void parseCSV(	ifstream &data_file,
			vector<int> &labels,
			vector<vector<double>> &features,
			int rows,
			int traits) {

	string token;
  char *token_end;
  for (int row = 0; row < rows; row++) {
      vector<double> line;
      bool include = true;
      for (int feature = 0; feature < traits; feature++) {
          if (feature != traits - 1) {
              double value = strtold(token.c_str(), &token_end);
              if (token.c_str() != token_end && *token_end == '\0') {
                  line.push_back(value);
              } else {
                  include = false;
              }
          } else {
              if (token == "inactive") {
                  if (include) {
                      labels.push_back(0);
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
//COPIES FROM A SOURCE FEATURES MATRIX AND LABELS INTO SOURCES, PRESERVING THE SOURCE
void sourceRandTwoDest(vector<vector<double>>& source,
				vector<int>& labels,
				vector<vector<double>>& train,
				vector<vector<double>>& test,
				vector<int>& train_y,
				vector<int>& test_y,
				long double partition) {
		for(size_t i = 0; i < source.size(); ++i){
				double te_or_tr = (double) ((rand() % 10000) / 10000); //Simple way to get decimal val
				if(te_or_tr <= partition){
					train_y.push_back(labels[i]);
					 train[i] = vector<double>();
					 for(uint32_t j = 0; j < source[i].size(); ++j){
						 train[i].push_back(source[i][j]);
					 }
				}
				else{
					test[i] = vector<double>();
					test_y.push_back(labels[i]);
					for(uint32_t j = 0; j < source[i].size(); ++j){
						train[i].push_back(source[i][j]);
					}
				}
    }
}

	//COPIES OVER VECTORS AND LABELS FROM ONE DATASET TO ANOTHER
	//MODIFIES BOTH SOURCE AND DEST VECTORS
void splitRandOneDest(vector<vector<double>>& original,
			vector<int>& original_y,
			vector<vector<double>>& dest,
			vector<int>& dest_y,
			long double partition) {
    size_t dest_size = original.size() * partition;
    while (dest.size() < dest_size) {
        size_t idx = rand() % original.size();
        vector<double>value(original[idx]);
        original[idx] = original.back();
        original.pop_back();
        dest.push_back(value);
				dest_y.push_back(original_y[idx]);
				original_y[idx] = original_y.back();
				original_y.pop_back();
		}
}
//NOTE: NO GUARANTEE THAT BOTH VECTORS HAVE SAME LENGTH
double dotProduct(vector<double>& a, vector<double>& x){
	double total = 0.0;
	for(uint32_t i = 0; i < a.size(); ++i){
		total += a[i] * x[i];
	}
	return total;
}

//NOTE: NO GUARANTEE THAT BOTH VECTORS HAVE SAME LENGTH
//MATHEMATICS DESCRIBED IN PG 30 IN THE BOOK
void gradientDescent(vector<double>& aleph,
			double &b,
			vector<double>& x,
			double& eta,
			vector<int>& label,
			double lambda){

	for(uint32_t i = 0; i < aleph.size(); ++i){
		int8_t sign = (label[i] == 0) ? -1 : label[i];
		bool cost = (sign * (dotProduct(aleph, x) + b)) >= 1;
		aleph[i] = aleph[i] - eta * (cost ? lambda * aleph[i] : lambda * aleph[i] - sign * x[i]);	//This is all SGD is with linear vectors
		b = b - eta * (cost ? 0 : -1 * sign);
	}

}

//NOTE: NO GUARANTEE THAT BOTH VECTORS HAVE SAME LENGTH
//COST FUNCTION DESCRIBED IN PG 26 OF BOOK
void cost(vector<double>& aleph,
						vector<vector<double>>& x,
						vector<int>& y,
						vector<double>& best_model_candidate,
						double b,
						double& best_b_candidate,
						double lambda,
						uint32_t v,
						double& min_cost){
	double aleph_magnitude = 0.0;

	double sum = 0.0;

	for(uint32_t v = 0; v < x.size(); ++v){	//Traverse through the size of the validation set
		aleph_magnitude += aleph[v]; //Catch all the terms in aleph and square at the end for magnitude

		for(uint32_t j = 0; j < x[v].size(); ++j){
			double temp = 1 - y[j] * dotProduct(aleph, x[j]);
			sum += (temp > 0) ? temp : 0;
		}

	}

	sum = (double)(1.0 / x.size()) * sum + (double)(lambda / 2.0 * aleph_magnitude);
	if(sum < min_cost){
		min_cost = sum;
		best_b_candidate = b;

		for(uint32_t i = 0; i < best_model_candidate.size(); ++i){
			best_model_candidate[i] = aleph[i];
		}

	}
}

int main(int argc, const char* argv[]){
	// MARK: load file
	ifstream data_file;
	data_file.open("/Users/rauhul/Developer/cs498/homework2/homework2/K9.data", ifstream::in);

	// MARK: parse file
	vector<int> labels;
	vector<vector<double>> features;
	parseCSV(data_file, labels, features, 31420, 5409);
	data_file.close();
	size_t label_count = labels.size();
	vector<double> best_model(features[0].size());	//Best model for classification
	vector<double> possible_best_model(features[0].size()); //Use new possible model on test data; if better, update best model
	vector<double> aleph(features[0].size()); //The A in the U vector used in SGD
	double b; //The b in the U vector used in SGD
	double best_b;
	double best_b_candidate;
	srand(time(NULL));
	uint16_t num_seasons = 100;
	uint16_t num_steps = 420;
	//Min is used to keep track across all lambdas what the lowest cost across a U vector is
	double min = 100000; //Should never be this high

	//NEED TO A RANGE OF LAMBDAS FROM 0 TO 0.5; ANYTHING HIGHER IS NOT PRACTICAL FOR A LEARNING RATE
	//PICKING A LAMBDA FROM A SET DESCRIBED IN PROCEDURE 3.1 ON PG 29
	for(double lambda = 0.5; lambda < 0.5; lambda += 0.05){
		vector<vector<double>> train;
		vector<int> train_y;
		vector<vector<double>> test;
		vector<int> test_y;
		//Need a seasonal min to figure out out of the seasons which one has the lowest cost
		double seasonal_min = 100000;
		sourceRandTwoDest(features, labels, train, test, train_y, test_y, 0.15);

		for(uint32_t i = 0; i < aleph.size(); ++i){
			aleph[i] = -10 + rand() % 20; //Range from -10 to 10, as initial values for the a vector in the U vector
		}

		b = -10 + rand() % 20;	//Set a random initial value for b

		//Run through the seasons. After each season use best possible candidate from the season on test data
		//Pg 29 in Procedure
		for(uint32_t season = 0; season < num_seasons; ++season){
			vector<vector<double>> validate;	//Create a validation set for our Aleph and b
			vector<int> validate_y;
			vector<vector<double>> seasonal_train(train); //We don't want to edit train itself because we'll be reusing it over the seasons
			vector<int> seasonal_train_y(train_y);				//Create a set of labels pertinent to the training data at hand
			splitRandOneDest(seasonal_train, seasonal_train_y, validate, validate_y, 0.15);

			//Step through the season, modifying aleph and b to find the best local minima for U vector to test
			for(uint16_t step = 0; step < num_steps; ++step){
				double eta = (double)(1.0/(50 + 0.01*step)); //Numbers right now borrowed from Forsyth's book
				gradientDescent(aleph, b, seasonal_train[step], eta, seasonal_train_y, lambda);
			}

			//Calculate the total precision of the classifier made by the season
			cost(aleph, validate, validate_y, possible_best_model, b, best_b_candidate, lambda, validate.size(), seasonal_min);

		}

		//Test on data using best possible model from the seasons given a certain lambda
		cost(possible_best_model, test, test_y, best_model, best_b_candidate, best_b, lambda, test.size(), min);

	}
}

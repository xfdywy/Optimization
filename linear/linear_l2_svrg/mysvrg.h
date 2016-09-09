
#include <iostream>
#include <fstream>
#include <iomanip>
#include <random>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <map>
#include <math.h>
#include "Eigen/Sparse"
#include<iomanip>



using namespace std;
using namespace Eigen;


vector<string> split(const string &s, char delim, vector<string> &elems) {
	stringstream ss(s);
	string item;
	while (getline(ss, item, delim)) {
		elems.push_back(item);
	}
	return elems;
}

vector<string> split(const string &s, char delim) {
	vector<string> elems;
	split(s, delim, elems);
	return elems;
}

void usage(const char* prog){

	cout << "Read training data then classify test data using logistic regression:\nUsage:\n" << prog << " [options] [training_data]" << endl << endl;
	cout << "Options:" << endl;
	cout << "-s <int>   Shuffle dataset after each iteration. default 1" << endl;
	cout << "-i <int>   Maximum iterations. default 50000" << endl;
	cout << "-e <double> Convergence rate. default 0.005" << endl;
	cout << "-a <double> Learning rate. default 0.001" << endl;
	cout << "-l <double> L2 regularization weight. default 0.0001" << endl;
	cout << "-m <file>  Read weights from file" << endl;
	cout << "-o <file>  Write weights to file" << endl;
	cout << "-t <file>  Test file to classify " << endl;
	cout << "-p <file>  Write predictions to file" << endl;
	cout << "-r         Randomise weights between -1 and 1, otherwise 0" << endl;
	cout << "-v         Verbose." << endl << endl;
	cout << "-innerm        inner iterate number" << endl << endl;
	cout << "-losstype         " << endl << endl;
	cout << "-rc  regression or classify,default is 2" << endl;
}

double vecnorm(map<int, double>& w1, map<int, double>& w2){

	double sum = 0.0;
	for (auto it = w1.begin(); it != w1.end(); it++){
		double minus = w1[it->first] - w2[it->first];
		double r = minus * minus;
		sum += r;
	}
	return sqrt(sum);
}






SparseMatrix<double> myreadfile(string &filename, int &maxfeature, int rc)

{
	int numsample = 0;

	ifstream fin;
	string line;

	fin.open(filename.c_str());

	vector<Triplet<double> > triple;
	int n = 0;
	while (getline(fin, line)){
		if (n % 500 == 0)
		{
			cout << "have read " << n << "data" << endl;
		}
		if (line[0] != '#' && line[0] != ' '){

			vector<string> tokens = split(line, ' ');
			if (rc == 2){
				//cout << "error" << endl;
				if (atoi(tokens[0].c_str()) == 1){
					triple.push_back(Triplet<double>(n, 0, 1));
				}
				else{
					triple.push_back(Triplet<double>(n, 0, 0));
				}

				for (unsigned int i = 1; i < tokens.size(); i++){
					vector<string> feat_val = split(tokens[i], ':');

					if (atoi(feat_val[0].c_str()) > maxfeature){
						maxfeature = atoi(feat_val[0].c_str());
					}

					triple.push_back(Triplet<double>(n, atoi(feat_val[0].c_str()), atof(feat_val[1].c_str())));

				}


			}
			else
			{

				triple.push_back(Triplet<double>(n, 0, (atof(tokens[0].c_str()))));
				//cout << atof(tokens[0].c_str()) << endl;
				for (unsigned int i = 1; i < tokens.size(); i++){
					vector<string> feat_val = split(tokens[i], ':');

					if (atoi(feat_val[0].c_str()) > maxfeature){
						maxfeature = atoi(feat_val[0].c_str());
					}

					triple.push_back(Triplet<double>(n, atoi(feat_val[0].c_str()), atof(feat_val[1].c_str())));

				}
			}


			n = n + 1;
		}
	}

	fin.close();

	SparseMatrix<double> m(n, maxfeature + 1);
	m.setFromTriplets(triple.begin(), triple.end());
	cout << n << "#####" << n << "#####" << maxfeature << "#####" << m.cols() << endl;
	//cout << m.col(0);
	return(m);
}


MatrixXd classify(SparseMatrix<double>& features, MatrixXd& weights){

	MatrixXd logit;

	logit = (features * weights);

	return (logit);
}





double classifyloss(SparseMatrix<double>& m, MatrixXd& weights, int losstype, double l2)
{

	double result = 0;
	//cout << weights.rows() << endl;
	SparseMatrix<double> example = m.rightCols(weights.rows());

	MatrixXd label = m.col(0);

	MatrixXd predicted = classify(example, weights);


	int ndata = m.rows();

	if (losstype == 1){

		result += (predicted - label).array().square().sum();
		result = result / double(m.rows()) + l2*(weights.array().square().sum());
		/*for (int ii = 0; ii < ndata; ii++){
		if ((label(ii, 0) == 1 && predicted(ii, 0)>0.5) || (label(ii, 0) == 0 && predicted(ii, 0) < 0.5))
		{
		result++;
		}
		}*/
	}
	if (losstype == 2){
		result += (predicted - label).array().square().sum();
		result = result / double(m.rows());
		//cout << "############################" <<label << endl;
	}




	return (result);

}


MatrixXd computegradient(MatrixXd weights, SparseMatrix<double> data, MatrixXd label, double l2){

	MatrixXd predicted = classify(data, weights);
	MatrixXd gradient = ((((predicted - label).transpose()) * data).transpose()) / double(data.rows()) + l2*weights;
	return(gradient);
}
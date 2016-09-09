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
using namespace	Eigen;




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
	cout << "-minibatchsize        minibatchsize" << endl << endl;
	cout << "-losstype         " << endl << endl;
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







SparseMatrix<double> myreadfile(string &filename, int &maxfeature)

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
			n = n + 1;
		}
	}

	fin.close();

	SparseMatrix<double> m(n, maxfeature + 1);
	m.setFromTriplets(triple.begin(), triple.end());
	cout << n << "#####" << n << "#####" << maxfeature << "#####" << m.cols() << endl;
	return(m);
}


MatrixXd sigmoid(MatrixXd x){



	auto y = x.array().exp();

	return 1.0 / (1.0 + y);
}



MatrixXd classify(SparseMatrix<double>& features, MatrixXd& weights){

	MatrixXd logit;

	logit = -1 * (features * weights);

	return sigmoid(logit);
}



double classifyloss(SparseMatrix<double>& m, MatrixXd& weights, int losstype, double l2)
{

	double result = 0;
	//cout << weights.rows() << endl;
	SparseMatrix<double> example = m.rightCols(weights.rows());

	MatrixXd label = m.col(0);

	MatrixXd predicted = classify(example, weights);



	if (losstype == 1){

		result = 0 - (label.array()*(predicted.array().log()) + (1 - label.array())*((1 - predicted.array()).array().log())).sum();
		//cout << result  <<"************" << double(m.rows()) << endl;

		result = result / double(m.rows()) + l2*(weights.array().square().sum()) / 2.0;
		//cout << "yes1" << weights.array().square().sum() << endl;

		/*for (int ii = 0; ii < ndata; ii++){
		if ((label(ii, 0) == 1 && predicted(ii, 0)>0.5) || (label(ii, 0) == 0 && predicted(ii, 0) < 0.5))
		{
		result++;
		}
		}*/
	}
	if (losstype == 2){
		result -= (label.array()*(predicted.array().log()) + (1 - label.array())*((1 - predicted.array()).array().log())).sum();

		result = result / double(m.rows());
		cout << "yes2" << result << endl;
	}


	return (result);

}


MatrixXd computegradient(MatrixXd weights, SparseMatrix<double> data, MatrixXd label, double l2){

	MatrixXd predicted = classify(data, weights);
	MatrixXd gradient = (((((predicted - label).transpose()) * data).transpose()) / double(data.rows())) + l2*weights;
	return(gradient);
}

int main(int argc, const char* argv[]){

	// Learning rate
	double alpha = 0.001;
	// L2 penalty weight
	double l2 = 0.0001;
	// Max iterations
	unsigned int maxit = 500;
	// Shuffle data set
	int shuf = 1;
	// Verbose
	int verbose = 0;
	// Randomise weights
	int randw = 0;
	// Minibatch size
	int minibatchsize = -1;
	// Read model file
	string model_in = "";
	// Write model file
	string model_out = "";
	// Test file
	string test_file = "";
	//Loss type
	int losstype;

	// Predictions file
	string predict_file = "";

	if (argc < 2){
		usage(argv[0]);
		return(1);
	}
	else{
		cout << "# called with:       ";
		for (int i = 0; i < argc; i++){
			cout << argv[i] << " ";
			if (string(argv[i]) == "-a" && i < argc - 1){
				alpha = atof(argv[i + 1]);
			}
			if (string(argv[i]) == "-m" && i < argc - 1){
				model_in = string(argv[i + 1]);
			}
			if (string(argv[i]) == "-o" && i < argc - 1){
				model_out = string(argv[i + 1]);
			}
			if (string(argv[i]) == "-t" && i < argc - 1){
				test_file = string(argv[i + 1]);
			}

			if (string(argv[i]) == "-p" && i < argc - 1){
				predict_file = string(argv[i + 1]);
			}
			if (string(argv[i]) == "-s" && i < argc - 1){
				shuf = atoi(argv[i + 1]);
			}
			if (string(argv[i]) == "-i" && i < argc - 1){
				maxit = atoi(argv[i + 1]);
			}
			if (string(argv[i]) == "-l" && i < argc - 1){
				l2 = atof(argv[i + 1]);
			}
			if (string(argv[i]) == "-v"){
				verbose = 1;
			}
			if (string(argv[i]) == "-r"){
				randw = 1;
			}
			if (string(argv[i]) == "-h"){
				usage(argv[0]);
				return(1);
			}
			if (string(argv[i]) == "-losstype" && i < argc - 1){
				losstype = atof(argv[i + 1]);
			}
			if (string(argv[i]) == "-minibatchsize" && i < argc - 1){
				minibatchsize = atof(argv[i + 1]);
			}
		}
		cout << endl;
	}

	if (!model_in.length()){
		cout << "# learning rate:     " << alpha << endl;
		cout << "# l2 penalty weight: " << l2 << endl;
		cout << "# max. iterations:   " << maxit << endl;
		cout << "# training data:     " << argv[argc - 1] << endl;
		if (model_out.length()) cout << "# model output:      " << model_out << endl;
	}
	if (model_in.length()) cout << "# model input:       " << model_in << endl;
	if (test_file.length()) cout << "# test data:         " << test_file << endl;

	if (predict_file.length()) cout << "# predictions:       " << predict_file << endl;




	random_device rd;
	mt19937 g(rd());
	ifstream fin;
	string line;







	string train_file = argv[argc - 1];
	int trainmaxfeature = 0;
	int testmaxfeature = 0;

	SparseMatrix<double>  traindataraw;
	SparseMatrix<double>  testdataraw;

	SparseMatrix<double>  traindata;
	SparseMatrix<double>  testdata;

	cout << "start read traindata" << endl;
	traindataraw = myreadfile(train_file, trainmaxfeature);
	traindata = traindataraw.rightCols(trainmaxfeature);
	MatrixXd	trainlabel = traindataraw.col(0);

	cout << "start read test data" << endl;
	testdataraw = myreadfile(test_file, trainmaxfeature);
	testdata = testdataraw.rightCols(trainmaxfeature);
	MatrixXd	testlabel = testdataraw.col(0);

	cout << "read data done" << endl;




	MatrixXd  weights(trainmaxfeature, 1);
	cout << "ini weights" << endl;
	for (int ii = 0; ii < trainmaxfeature; ii++){
		weights(ii, 0) = 0;
	}

	cout << "# training examples: " << traindata.rows() << endl;
	cout << "# test examples: " << testdata.rows() << endl;
	cout << "# features:          " << traindata.cols() << endl;
	if (minibatchsize < 1)
	{
		minibatchsize = traindata.rows();
	}
	cout << "# minibatchsize :     " << minibatchsize << endl;
	double norm = 1.0;
	unsigned int n = 0;
	vector<int> index(traindata.rows());
	//iota(index.begin(),index.end(),0);//���ɴ�0��ʼ�ĵ�������


	cout << "# stochastic gradient descent" << endl;

	if (model_out.length()){
		ofstream outfile;
		outfile.open(model_out.c_str(), ofstream::ate);
		outfile << "num_iter" << "," << "testloss" << "," << "trainloss" << ","<< "passeddata" << endl;

		double alphao = alpha;
		for (n = 0; n < maxit; n++)
		{	
			alpha = alphao;

			if (n){
				

				auto grad = computegradient(weights, traindata, trainlabel, l2);
				cout << "\n" << "&&&&&&&&&" << grad.maxCoeff() << "&&&&&&&" << endl << endl;
				weights = weights - alpha*grad;

				//cout << alpha << "@@@@" << weights.maxCoeff() << "@@@@@@" << grad.maxCoeff() << endl;
			}
			if (n>=0){

				

				cout << "\titerations:\t" << n << endl;

				double testresult1;
				double testresult2;

				testresult1 = classifyloss(testdataraw, weights,2,l2);



				testresult2 = classifyloss(traindataraw, weights,1,l2);

				cout << setiosflags(ios::fixed) << setprecision(13);
				outfile << setiosflags(ios::fixed) << setprecision(13);
				outfile << n << "," << testresult1 << "," << testresult2 << "," << n*traindata.rows() << endl;
				cout << n << "," << testresult1 << "," << testresult2 << "," << n*traindata.rows() << endl;
				//cout << label.topRows(10) << "\t" << label.bottomRows(10) << endl;
				//cout << predicted.topRows(10) << "\t" << predicted.bottomRows(10) << endl;

			}

		}

	}


	return(0);

}

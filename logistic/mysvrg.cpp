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
#include "mysvrg.h"
#include<iomanip>



using namespace std;
using namespace	Eigen;







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
	// inner iteration number
	int innerm = 1;
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
			if (string(argv[i]) == "-innerm" && i < argc - 1){
				innerm = atof(argv[i + 1]);
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
	MatrixXd testlabel = testdataraw.col(0);


	cout << "read data done" << endl;




	MatrixXd  weights(trainmaxfeature, 1);
	cout << "ini weights" << endl;
	for (int ii = 0; ii < trainmaxfeature; ii++){
		weights(ii, 0) = 0;
	}
	MatrixXd outweights = weights;
	MatrixXd inweights = weights;


	cout << "# training examples: " << traindata.rows() << endl;
	cout << "# test examples: " << testdata.rows() << endl;
	cout << "# features:          " << traindata.cols() << endl;
	if (innerm < 1)
	{
		innerm = 1;
	}
	cout << "# inner iteration number :     " << innerm << endl;
	double norm = 1.0;
	unsigned int n = 0;
	vector<int> index(traindata.rows());
	
	cout << "# stochastic gradient descent" << endl;

	if (model_out.length()){
		ofstream outfile;
		outfile.open(model_out.c_str(), ofstream::ate);
		outfile << "num_iter" << "," << "testloss" << "," << "trainloss" << ","<<"passeddata" << endl;

		double alphao = alpha;
		for (n = 0; n < maxit; n++)
		{
			alpha = alphao;
			cout << n << endl;
			if (n == 0){
				cout << "\titerations:\t" << n << endl;

				double testresult1;
				double testresult2;

				testresult1 = classifyloss(testdataraw, inweights, 2, l2);



				testresult2 = classifyloss(traindataraw, inweights, 1, l2);
				cout << setiosflags(ios::fixed) << setprecision(13);
				outfile << setiosflags(ios::fixed) << setprecision(13);


				outfile << n << "," << testresult1 << "," << testresult2 << "," << 0 << endl;
				cout << n << "," << testresult1 << "," << testresult2 << "," << 0 << endl;
			}
			if (n>0){
				MatrixXd outgradient = computegradient(outweights, traindata, trainlabel, l2);

				inweights = outweights;

				for (int tt = 0; tt < innerm; tt++){
					//cout << tt << endl;
					int thisdata = g() % (traindata.rows());
					SparseMatrix<double> thisdata_val = traindata.row(thisdata);
					MatrixXd thisdata_label = trainlabel.row(thisdata);


					MatrixXd ingradientnew = computegradient(inweights, thisdata_val, thisdata_label, l2);
					//cout << ingradientnew.maxCoeff() << endl;

					MatrixXd ingradientold = computegradient(outweights, thisdata_val, thisdata_label, l2);
					//cout << ingradientold.maxCoeff() << endl;
					//cout << alpha << endl;
					inweights = inweights - alpha*(ingradientnew - ingradientold + outgradient);
					//cout << "@@@@@@@" << ingradientnew.maxCoeff() << endl;




					if (tt%1000==0){

						//alpha = sqrt(alpha);

						cout << "\titerations:\t" << n << endl;

						double testresult1;
						double testresult2;

						testresult1 = classifyloss(testdataraw, inweights, 2, l2);



						testresult2 = classifyloss(traindataraw, inweights, 1, l2);
						cout << setiosflags(ios::fixed) << setprecision(13);
						outfile << setiosflags(ios::fixed) << setprecision(13);
						outfile << n << "," << testresult1 << "," << testresult2 << "," << 20242 + (n - 1)*(innerm + 20242) + tt << endl;
						cout << n << "," << testresult1 << "," << testresult2 << "," << 20242 + (n - 1)*(innerm + 20242) + tt << endl;
						//cout << label.topRows(10) << "\t" << label.bottomRows(10) << endl;
						//cout << predicted.topRows(10) << "\t" << predicted.bottomRows(10) << endl;

					}







				}

				outweights = inweights;

			}







		}

	}


	return(0);

}

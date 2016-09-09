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

	cout << "Read training data then classify test data using linear regression:\nUsage:\n" << prog << " [options] [training_data]" << endl << endl;
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

		result += (predicted - label).array().square().sum() ;
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

	//MatrixXd gradient = ((((predicted - label).transpose()) * data).transpose() + l2*weights) / double(data.rows());
	return(gradient);
}

void upadateweights(MatrixXd label, MatrixXd predicted, double alpha, MatrixXd& data, MatrixXd& oldweights, double l2){

	//cout << oldweights.maxCoeff() << endl;

	oldweights -= alpha *((((predicted - label).transpose()) * data).transpose() + l2*oldweights)/double(data.rows());



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
	//regression or classification
	int rc = 2;

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
			if (string(argv[i]) == "-minibatchsize" && i < argc - 1){
				minibatchsize = atof(argv[i + 1]);
			}

			if (string(argv[i]) == "-losstype" && i < argc - 1){
				losstype = atof(argv[i + 1]);
			}
			if (string(argv[i]) == "-rc" && i < argc - 1){
				rc = atof(argv[i + 1]);
			}
		}
		cout << endl;
	}

	if (!model_in.length()){
		cout << "# learning rate:     " << alpha << endl;
		cout << "# l2 penalty weight: " << l2 << endl;
		cout << "# max. iterations:   " << maxit << endl;
		cout << "# training data:     " << argv[argc - 1] << endl;
		cout << "loss type" << losstype << endl;
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
	traindataraw = myreadfile(train_file, trainmaxfeature,rc);
	traindata = traindataraw.rightCols(trainmaxfeature);
	MatrixXd	trainlabel = traindataraw.col(0);

	cout << "start read test data" << endl;
	testdataraw = myreadfile(test_file, trainmaxfeature,rc);
	testdata = testdataraw.rightCols(trainmaxfeature);
	MatrixXd	testlabel = traindataraw.col(0);

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
	double alphao = alpha;
	if (model_out.length()){
		ofstream outfile;
		outfile.open(model_out.c_str(), ofstream::ate);
		outfile << "num_iter" << "," << "testloss" << "," << "trainloss" << "," << "passeddata" << endl;

		
		for (n = 0; n < maxit; n++)
		{	
			

			if (n>0){
				//if (n % (20) == 0 && alpha >0.05 )	{
				//	alpha = 80 / ((n / double(20)) + 1);

				//}
				//cout << alpha << endl;*/
				//cout << alpha << "$$$$" << n << endl;

				//if(shuf) shuffle(index.begin(),index.end(),g);//g ������������

				//alpha =alphao / (1+n/15.0);

				alpha = 1.0 / (n + 20);

				MatrixXd  thisdata_val(minibatchsize, trainmaxfeature);
				SparseMatrix<double> sp_thisdata_val(minibatchsize, trainmaxfeature);
				MatrixXd thisdata_label(minibatchsize, 1);

				MatrixXd grad;
				grad = MatrixXd::Zero(trainmaxfeature, 1);

				for (int ii = 0; ii < minibatchsize; ii++){
					int thisdata = g() % (traindata.rows());

					grad  = grad + computegradient(weights, traindata.row(thisdata), trainlabel.row(thisdata), l2);


				}

				grad = grad / double(minibatchsize);
				//if (minibatchsize == 1)
				//{
				//	int thisdata = g() % (traindata.rows());
				//	//cout << thisdata << endl;
				//	sp_thisdata_val = traindata.row(thisdata);
				//	thisdata_label = trainlabel.row(thisdata);
				//	//cout << 1111 << endl;
				//}
				//else{
				//	for (unsigned int i = 0; i < minibatchsize; i++)
				//	{


				//		int thisdata = g() % (traindata.rows());
				//		//cout << 1111 << endl;
				//		thisdata_val.row(i) = MatrixXd(traindata.row(thisdata));
				//		//cout << 22222 << endl;
				//		thisdata_label.row(i) = trainlabel.row(thisdata);
				//		//cout << 3333 << endl;
				//		sp_thisdata_val = thisdata_val.sparseView();


				//	}

				//}


				//cout << sp_thisdata_val.sum() << endl;
				//cout << "##########" << weights.maxCoeff() << endl;
				//MatrixXd predicted = classify(sp_thisdata_val, weights);//sigmoid
				//cout << alpha << endl;
				//cout << predicted.maxCoeff()<<endl;
				
				weights = weights - alpha*(grad);
				//cout << weights.maxCoeff() << "%%%%%%%%%%" << endl;
				//padateweights(thisdata_label, predicted, alpha, thisdata_val, weights, l2);

			}



			if (n>=0 && n % 1000 == 0){

				//alpha = sqrt(alpha);

				cout << "\titerations:\t" << n << endl;

				double testresult1;
				double testresult2;
				//cout << "@@@@@@@" << grad.maxCoeff() << endl;
				testresult1 = classifyloss(testdataraw, weights,2,l2);

				
				testresult2 = classifyloss(traindataraw, weights,1,l2);

				cout << setiosflags(ios::fixed) << setprecision(13);
				outfile << setiosflags(ios::fixed) << setprecision(13);
				outfile << n << "," << testresult1 << "," << testresult2 << "," << n*minibatchsize <<endl;
				cout << n << "," << testresult1 << "," << testresult2 << "," << n*minibatchsize  << endl;
				//cout << label.topRows(10) << "\t" << label.bottomRows(10) << endl;
				//cout << predicted.topRows(10) << "\t" << predicted.bottomRows(10) << endl;

			}

		}

	}


	return(0);

}

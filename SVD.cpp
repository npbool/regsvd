//
// Created by Ningping Wang on 15/8/19.
//

#include "SVD.h"
#include "Dataset.h"
#include <iostream>
#include <cmath>
using namespace Eigen;
using namespace std;
inline float logistic(float x){
    return log(1+exp(-x));
}
inline float derivative(float x,int y){
    float t = 1/(1+exp(-x*y));
    return -y * t *(1-t);
}
void SVD::train(std::vector<Sample> train_data, int num_user_fea, int num_item_fea, int dim){
    this->num_item_fea = num_item_fea;
    this->num_user_fea = num_user_fea;
    this->dim = dim;
    this->n_train = train_data.size();
    init();

    int minibatch=1000;

    for(int n_round=0;n_round<20000;++n_round) {
        MatrixXf user_fea_vec_grad = MatrixXf::Zero(num_user_fea, dim);
        MatrixXf item_fea_vec_grad = MatrixXf::Zero(num_item_fea, dim);
        ArrayXf user_fea_bias_grad = ArrayXf::Zero(num_user_fea);
        ArrayXf item_fea_bias_grad = ArrayXf::Zero(num_item_fea);
        float bias_grad = 0;

        float loss = 0;
        for (int i = 0; i < minibatch; ++i) {
            Sample &sample = train_data[random() % n_train];

            float sum = bias;
            VectorXf user_fea_vec = VectorXf::Zero(dim);
            for (float user_fea : sample.user_features) {
                sum += user_feature_bias(user_fea);
                user_fea_vec += user_feature_mat.row(user_fea);
            }

            VectorXf item_fea_vec = VectorXf::Zero(dim);
            for (float item_fea : sample.item_features) {
                sum += item_feature_bias(item_fea);
                item_fea_vec += item_feature_mat.row(item_fea);
            }

            sum += item_fea_vec.dot(user_fea_vec);

            loss += log(1+exp(-sample.label*sum));
            float grad = derivative(sum,sample.label);
            //cout<<sum<<" "<<grad<<endl;
            //getchar();

            for(float user_fea : sample.user_features){
                user_fea_vec_grad.row(user_fea) += grad*item_fea_vec;
                user_fea_bias_grad(user_fea) += grad;
            }
            for(float item_fea : sample.item_features){
                item_fea_vec_grad.row(item_fea) += grad*user_fea_vec;
                item_fea_bias_grad(item_fea) += grad;
            }

            bias_grad += grad;
        }

        float lr = 0.1 * 500/ (500+ n_round);
        float reg = 0.00001;
        user_feature_bias = user_feature_bias * (1-reg) - user_fea_bias_grad / minibatch * lr;
        item_feature_bias = item_feature_bias * (1-reg) - item_fea_bias_grad / minibatch * lr;

        user_feature_mat = user_feature_mat * (1-reg) - user_fea_vec_grad /minibatch * lr;
        item_feature_mat = item_feature_mat*(1-reg) - item_fea_vec_grad /minibatch*lr;

        bias -= bias_grad/minibatch*lr;

        //cout<<"Loss:"<<loss<<endl;
    }
}

void SVD::evaluate(std::vector<Sample> test_data) {
    int corr = 0;
    for(int i=0;i<test_data.size();++i){
        Sample &sample = test_data[i];

        float sum = bias;
        VectorXf user_fea_vec = VectorXf::Zero(dim);
        for (float user_fea : sample.user_features) {
            sum += user_feature_bias(user_fea);
            user_fea_vec += user_feature_mat.row(user_fea);
        }

        VectorXf item_fea_vec = VectorXf::Zero(dim);
        for (float item_fea : sample.item_features) {
            sum += item_feature_bias(item_fea);
            item_fea_vec += item_feature_mat.row(item_fea);
        }

        sum += item_fea_vec.dot(user_fea_vec);
        int pred;
        if(sum>=0) pred=1;
        else pred = -1;
        if(pred==sample.label){
            corr+=1;
        }
    }
    cout<<"Accuracy: "<<corr*1.0/test_data.size()<<endl;
}



void SVD::init(){
    user_feature_mat = (MatrixXf::Random(num_user_fea, dim).array()*0.1).matrix();
    item_feature_mat = (MatrixXf::Random(num_item_fea, dim).array()*0.1).matrix();

    user_feature_bias = ArrayXf::Zero(num_user_fea);
    item_feature_bias = ArrayXf::Zero(num_item_fea);

    bias = 0;
}

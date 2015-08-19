//
// Created by Ningping Wang on 15/8/19.
//

#ifndef REGSVD_SVD_H
#define REGSVD_SVD_H

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <unordered_map>
#include <vector>
#include "Dataset.h"


class SVD {

public:
    void train(std::vector<Sample> train_data, int num_user_fea, int num_item_fea, int dim);
    void evaluate(std::vector<Sample> test_data);

private:
    void init();


    int n_train;
    int dim;
    int num_user_fea;
    int num_item_fea;
    Eigen::MatrixXf user_feature_mat;
    Eigen::MatrixXf item_feature_mat;

    Eigen::ArrayXf user_feature_bias;
    Eigen::ArrayXf item_feature_bias;
    float bias;

    //std::unordered_map<long, int> item_feature_map;
    //std::unordered_map<long, int> user_feature_map;
};


#endif //REGSVD_SVD_H

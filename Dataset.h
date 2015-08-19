//
// Created by Ningping Wang on 15/8/19.
//

#ifndef REGSVD_SAMPLE_H
#define REGSVD_SAMPLE_H

#include <vector>

struct Sample {
    int label;
    std::vector<int> user_features;
    std::vector<int> item_features;
};
struct Dataset{
    std::vector<Sample> data;
    int n_sample;
    int n_user_fea;
    int n_item_fea;
};



#endif //REGSVD_SAMPLE_H

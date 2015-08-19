#include <iostream>
#include <algorithm>
#include "SVD.h"
#include "Dataset.h"
#include "IO.h"
using namespace std;

int main() {
    SVD svd;

    Dataset ds = IO::readDataset("data");
    cout<<ds.data[0].label<<endl;
    cout<<ds.data[2].label<<endl;
    cout<<ds.data[3].label<<endl;
    cout<<ds.data[5].label<<endl;
    cout<<ds.n_sample<<endl;
    cout<<endl;

    vector<Sample> train;
    vector<Sample> test;
    for(int i=0;i<4000;++i){
        train.push_back(ds.data[i]);
    }

    for(int i=4000;i<ds.data.size();++i){
        test.push_back(ds.data[i]);
    }
    svd.train(train, ds.n_user_fea, ds.n_item_fea, 10);
    svd.evaluate(test);
    return 0;
}
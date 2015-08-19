//
// Created by Ningping Wang on 15/8/19.
//

#include "IO.h"

#include <fstream>
#include <iostream>
using namespace std;
Dataset IO::readDataset(const char *filename) {
    FILE* file = fopen(filename, "r");
    Dataset ds;
    fscanf(file, "%d%d%d", &ds.n_user_fea, &ds.n_item_fea, &ds.n_sample );
    cout<<ds.n_user_fea<<" "<<ds.n_item_fea<<" "<<ds.n_sample<<endl;
    cout<<"Reading"<<endl;

    for(int i=0;i<ds.n_sample;++i){
        Sample sample;
        int nu,ni;
        fscanf(file, "%d%d%d", &sample.label, &nu, &ni);

        while(nu>0){
            int userf;
            fscanf(file, "%d", &userf);
            sample.user_features.push_back(userf);
            --nu;
        }
        while(ni>0){
            int itemf;
            fscanf(file, "%d", &itemf);
            sample.item_features.push_back(itemf);
            --ni;
        }

        ds.data.push_back(sample);
    }
    fclose(file);
    return ds;
}

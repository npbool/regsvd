//
// Created by Ningping Wang on 15/8/19.
//

#ifndef REGSVD_IO_H
#define REGSVD_IO_H

#include <vector>
#include "Dataset.h"

class IO {
public:
    static Dataset readDataset(const char* filename);
};


#endif //REGSVD_IO_H

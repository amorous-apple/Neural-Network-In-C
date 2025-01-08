#ifndef NEURAL_H
#define NEURAL_H

#include "utils_files.h"
#include "utils_mat.h"
#include "init.h"
#include "activationFunctions.h"

typedef struct {
    Mat **hiddenLayers;
    Mat **weights;
    Mat **biases;
} Network;

Network *net_init(Mat **weights, Mat **biases);
void net_free(Network *net);
Mat **init_h_layers();
Mat **init_weights();
Mat **init_biases();
Mat *propagate(double (*actFnct)(double), Mat *input, Network *net);
void test_weights(Mat **weights, Mat **biases);

#endif

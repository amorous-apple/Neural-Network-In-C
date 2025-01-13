#ifndef NEURAL_H
#define NEURAL_H

#include "utils_files.h"
#include "utils_mat.h"
#include "init.h"
#include "activationFunctions.h"

typedef struct {
    Mat **preLayers;
    Mat **layers;
    Mat **weights;
    Mat **biases;
} Network;

Network *net_init(Mat **weights, Mat **biases);
void net_free(Network *net);
Mat **init_layers();
Mat **init_weights();
Mat **init_weightsZ();
Mat **init_biases();
Mat *propagate(double (*actFnct)(double), Mat *input, Network *net);
// Mat *prePropagate(double (*actFnct)(double), Mat *input, Network *net);
int *init_labels(int dataSize);
void test_weights(Mat **weights, Mat **biases);

#endif

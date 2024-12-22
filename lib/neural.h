#ifndef NEURAL_H
#define NEURAL_H

#include "utils_mat.h"
#include "init.h"
#include "activationFunctions.h"

Mat **init_h_layers();
Mat **init_weights();
Mat **init_biases();
Mat *propagate(Mat *input, Mat **hidden_layers, Mat **weights, Mat **biases);

#endif

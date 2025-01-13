#ifndef BACKPROP_H
#define BACKPROP_H

#include "utils_mat.h"
#include "init.h"
#include "neural.h"

Mat *error_output(double (*actFnct)(double), Mat *input, Network *net, int label);
Mat **calc_errors(double (*actFnct)(double), Mat *input, Network *net, int label);
void update_weights(double (*actFnct)(double), Mat **inputs, Mat **weights,
                    Mat **biases, int *labels, int startingIndex);
void traininator(double (*actFnct)(double), Mat **inputs, Mat **weights,
                 Mat **biases, int *labels);

#endif

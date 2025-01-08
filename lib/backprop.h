#ifndef BACKPROP_H
#define BACKPROP_H

#include "utils_mat.h"
#include "init.h"
#include "neural.h"

Mat *error_output(double (*actFnct)(double), Mat *input, Mat **weights,
                  Mat **biases, int label);

#endif

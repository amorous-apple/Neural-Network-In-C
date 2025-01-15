#ifndef BACKPROP_H
#define BACKPROP_H

#include "utils_mat.h"
#include "init.h"
#include "neural.h"

Mat *error_output(double (*actFnct)(double), Mat *input, Network *net, int label, Mat *errorOutMat, Mat *derivOutput, Mat *expectedVal);
Mat **calc_errors(double (*actFnct)(double), Mat *input, Network *net, int label, Mat **errors, Mat **derivOutputs, Mat *expectedVal, Mat **weightsTranspose);
void update_weights(double (*actFnct)(double), Mat **inputs, Mat **weights,
                    Mat **biases, int *labels, int startingIndex, Network **nets, Mat ***batchErrors, Mat **derivOutputs, Mat *expectedVal, Mat **weightsTranspose, Mat **weightsShift, Mat **inputsTranspose, Mat ***layersTranspose, Mat ***productsTemp);
void traininator(double (*actFnct)(double), Mat **inputs, Mat **weights,
                 Mat **biases, int *labels);

#endif

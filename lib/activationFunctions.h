#ifndef ACTIVATIONFUNCTIONS_H
#define ACTIVATIONFUNCTIONS_H

#include <math.h>
#include "utils_mat.h"

double relu(double input);
double drelu(double input);
double sigmoid(double input);
double dsigmoid(double input);
Mat *apply1(double (*ptr)(double), Mat *m);
Mat *apply2(double (*ptr)(double), Mat *m);

#endif

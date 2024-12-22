#ifndef ACTIVATIONFUNCTIONS_H
#define ACTIVATIONFUNCTIONS_H

#include <math.h>
#include "utils_mat.h"

double relu(double input);
double sigmoid(double input);
Mat *apply(double (*ptr)(double), Mat *m);

#endif

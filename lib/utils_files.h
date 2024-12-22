#ifndef UTILS_FILES_H
#define UTILS_FILES_H

#include <stdlib.h>
#include <stdio.h>

#include "init.h"
#include "utils_mat.h"

FILE *openDataFile(char *fileName);
Mat *fread_mat(char *filename);
Mat *dataToMat(FILE *pfile, int *label);

#endif

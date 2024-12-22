#include "utils_files.h"

FILE *openDataFile(char *fileName) {
    FILE *pfile = fopen(fileName, "r");
    if (pfile == NULL) {
        printf("Error opening dataFile %s\n", fileName);
        exit(EXIT_FAILURE);
    }
    return pfile;
}

Mat *fread_mat(char *filename) {
    FILE *pfile = openDataFile(filename);

    char line[MAX_LINE_LEN];
    fgets(line, MAX_LINE_LEN, pfile);
    int numRows = atoi(line);
    fgets(line, MAX_LINE_LEN, pfile);
    int numCols = atoi(line);

    Mat *matrix = mat_init(numRows, numCols);
    double entry;
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            fgets(line, MAX_LINE_LEN, pfile);
            entry = atof(line);
            matrix->values[i][j] = entry;
        }
    }
    return matrix;
}

Mat *dataToMat(FILE *pfile, int *label) {
    char tmpStr[MAX_LINE_LEN];
    fgets(tmpStr, MAX_LINE_LEN, pfile);

    label[0] = atoi(strtok(tmpStr, ","));

    Mat *fMat = mat_init(MAT_SIZE, MAT_SIZE);
    for (int i = 0; i < MAT_SIZE; i++) {
        for (int j = 0; j < MAT_SIZE; j++) {
            fMat->values[i][j] = atoi(strtok(NULL, ","));
        }
    }
    return fMat;
}

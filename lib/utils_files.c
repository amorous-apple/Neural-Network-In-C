#include "utils_files.h"

FILE *openDataFile(char *fileName) {
    FILE *pfile = fopen(fileName, "r");
    if (pfile == NULL) {
        printf("Error opening dataFile %s\n", fileName);
        exit(EXIT_FAILURE);
    }
    return pfile;
}

// Opening the input file and skipping over the first line (labeling)
FILE *openInputFile(char *fileName) {
    FILE *inputData = fopen(fileName, "r");
    if (inputData == NULL) {
        printf("Error opening dataFile %s\n", fileName);
        exit(EXIT_FAILURE);
    }
    char tmpStr[MAX_LINE_LEN];
    fgets(tmpStr, MAX_LINE_LEN, inputData);

    return inputData;
}

// Reading a stored matrix from a file
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

// Reading the input data into a column vector and placing the label into *label
Mat *dataToMat(FILE *pfile, int *label) {
    char tmpStr[MAX_LINE_LEN];
    fgets(tmpStr, MAX_LINE_LEN, pfile);

    label[0] = atoi(strtok(tmpStr, ","));

    Mat *fMat = mat_init(MAT_SIZE * MAT_SIZE, 1);
    for (int i = 0; i < MAT_SIZE * MAT_SIZE; i++) {
        fMat->values[i][0] = atoi(strtok(NULL, ","));
    }

    return fMat;
}

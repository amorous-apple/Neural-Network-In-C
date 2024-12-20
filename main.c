#include <string.h>

#include "lib/utils_files.h"
#include "lib/utils_mat.h"

const int MAT_SIZE = 28;
const int MAX_LINE_LEN = 20000;

int main() {
    // Mat *sample = mat_init(3, 4);
    // mat_populate(sample, 5);
    // mat_print(sample);

    FILE *inputData = openDataFile("./data/mnist_test.csv");

    char tmpStr[MAX_LINE_LEN];
    fgets(tmpStr, MAX_LINE_LEN, inputData);
    fgets(tmpStr, MAX_LINE_LEN, inputData);
    // printf("Line: %s \n", tmpStr);

    int label = atoi(strtok(tmpStr, ","));
    printf("\n\nLabel: %d\n", label);

    Mat *sample2 = mat_init(MAT_SIZE, MAT_SIZE);
    for (int i = 0; i < MAT_SIZE; i++) {
        for (int j = 0; j < MAT_SIZE; j++) {
            sample2->values[i][j] = atoi(strtok(NULL, ","));
        }
    }

    mat_printI(sample2);

    fclose(inputData);
}

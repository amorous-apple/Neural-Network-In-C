#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "lib/activationFunctions.h"
#include "lib/init.h"
#include "lib/neural.h"
#include "lib/utils_files.h"
#include "lib/utils_mat.h"

int main(int argc, char **argv) {
    // Setting the seed for rand()
    // srand(time(NULL));

    init(argc, argv);

    Mat **hiddenLayers = init_h_layers();
    Mat **weights = init_weights();
    weights[0] = fread_mat("./weights/weights1.txt");
    weights[1] = fread_mat("./weights/weights2.txt");
    Mat **biases = init_biases();

    FILE *inputData = openDataFile("./data/mnist_test.csv");

    char tmpStr[MAX_LINE_LEN];
    fgets(tmpStr, MAX_LINE_LEN, inputData);

    int dataSize = 1000;
    int numWrong = 0;
    int *label = malloc(sizeof(int));
    for (int i = 0; i < dataSize; i++) {
        Mat *input = dataToMat(inputData, label);
        Mat *inputCol = mat_flatten(input);

        Mat *output =
            propagate(sigmoid, inputCol, hiddenLayers, weights, biases);
        int guess = maxIndex(output);

        if (label[0] != guess) {
            printf("Error at line %d\n", i + 2);
            printf("Label: %d\n", label[0]);
            printf("Guess: %d\n", guess);
            mat_printI(input);
            numWrong++;
        }
    }
    double percentWrong = ((double)numWrong / dataSize) * 100;
    printf("Percent wrong: %lf %%", percentWrong);
}

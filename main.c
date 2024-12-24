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

    Network *net = net_init();

    FILE *inputData = openInputFile("./data/mnist_test.csv");

    int dataSize = 1000;
    int numWrong = 0;
    for (int i = 0; i < dataSize; i++) {
        int *label = malloc(sizeof(int));

        Mat *input = dataToMat(inputData, label);

        Mat *output = propagate(sigmoid, input, net);
        int guess = maxIndex(output);

        if (*label != guess) {
            // printf("Error at line %d\n", i + 2);
            // printf("Label: %d\n", label[0]);
            // printf("Guess: %d\n", guess);
            // mat_unflatten(&input, MAT_SIZE);
            // mat_printI(input);
            numWrong++;
        }
        free(label);
    }
    double percentWrong = ((double)numWrong / dataSize) * 100;
    printf("Percent wrong: %lf %%", percentWrong);
}

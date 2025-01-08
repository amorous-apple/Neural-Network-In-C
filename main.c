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

    FILE *inputData = openInputFile("./data/mnist_test.csv");
    int dataSize = 10000;

    // Loading all of the data into an array of matrices and labels
    int *labels = malloc(dataSize * sizeof(int));
    Mat **inputs = malloc(dataSize * sizeof(Mat *));
    for (int i = 0; i < dataSize; i++) {
        int *label = malloc(sizeof(int));
        inputs[i] = dataToMat(inputData, label);
        labels[i] = label[0];
        free(label);
    }

    printf("Input read from file.\n");

    Mat **weights = init_weights();
    weights[0] = fread_mat("./weights/weights1.txt");
    weights[1] = fread_mat("./weights/weights2.txt");
    Mat **biases = init_biases();

    int *guesses = malloc(dataSize * sizeof(int));
#pragma omp parallel for
    for (int i = 0; i < dataSize; i++) {
        Network *net = net_init(weights, biases);
        Mat *output = propagate(sigmoid, inputs[i], net);
        guesses[i] = maxIndex(output);

        free(output);
        net_free(net);
    }
    printf("Input propagated through network.\n");

    // Checking guesses
    int numWrong = 0;
    for (int i = 0; i < dataSize; i++) {
        if (labels[i] != guesses[i]) {
            printf("Error at line %d\n", i + 2);
            printf("Label: %d\n", labels[i]);
            printf("Guess: %d\n", guesses[i]);
            mat_unflatten(&inputs[i], MAT_SIZE);
            mat_printI(inputs[i]);
            numWrong++;
        }
    }

    double percentWrong = ((double)numWrong / dataSize) * 100;
    printf("Percent wrong: %lf %%", percentWrong);
}

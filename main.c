#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "lib/backprop.h"
#include "lib/init.h"
#include "lib/neural.h"
#include "lib/utils_files.h"
#include "lib/utils_mat.h"

int main(int argc, char **argv) {
    // Setting the seed for rand()
    // srand(time(NULL));

    init(argc, argv);

    Mat **weights = init_weights();
    weights[0] = fread_mat("./weights/weights1.txt");
    weights[1] = fread_mat("./weights/weights2.txt");
    Mat **biases = init_biases();

    int *labels = init_labels(TRAINING_DATA_SIZE);
    Mat **trainingData = init_trainingData(labels);
    // int sampleID = 2;
    //
    // Mat **errors = calc_errors(sigmoid, trainingData[sampleID], weights,
    // biases,
    //                            labels[sampleID]);
    //
    // printf("Label: %d\n", labels[sampleID]);
    // mat_unflatten(&trainingData[sampleID], 28);
    // mat_printI(trainingData[sampleID]);
    //
    // printf("errorOutput:\n");
    // mat_print(errors[1]);

    test_weights(weights, biases);

    printf("Starting training\n");
    traininator(sigmoid, trainingData, weights, biases, labels);
    // test_weights(weights, biases);
}

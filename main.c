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
    srand(time(NULL));

    init(argc, argv);

    Mat **weights = init_weights();
    // weights[0] = fread_mat("./weights/weights1.txt");
    // weights[1] = fread_mat("./weights/weights2.txt");
    Mat **biases = init_biases();

    int *labels = init_labels(TRAINING_DATA_SIZE);
    Mat **trainingData = init_trainingData(labels);

    test_weightsClosed(weights, biases);

    printf("Starting training\n");
    traininator(sigmoid, trainingData, weights, biases, labels);

    for (int i = 0; i < NUM_H_LAYERS + 1; i++) {
        mat_free(weights[i]);
        mat_free(biases[i]);
    }
    free(weights);
    free(biases);
    free(labels);
    for (int i = 0; i < TRAINING_DATA_SIZE; i++) {
        mat_free(trainingData[i]);
    }
    free(trainingData);
    free(NUM_LAYER_NODES);
}

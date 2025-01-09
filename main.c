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

    int *labels = init_labels(TRAINING_DATA_SIZE);
    Mat **trainingData = init_trainingData(labels);

    Mat **weights = init_weights();
    weights[0] = fread_mat("./weights/weights1.txt");
    weights[1] = fread_mat("./weights/weights2.txt");
    Mat **biases = init_biases();

    int sampleID = 2;

    Mat **errors = calc_errors(sigmoid, trainingData[sampleID], weights, biases,
                               labels[sampleID]);

    mat_unflatten(&trainingData[sampleID], 28);
    mat_printI(trainingData[sampleID]);

    // printf("errorOutput:\n");
    // mat_print(errors[0]);

    test_weights(weights, biases);
}

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
    srand(time(NULL));

    // init(argc, argv);
    //
    // Mat **weights = init_weights();
    // // weights[0] = fread_mat("./weights/weights1.txt");
    // // weights[1] = fread_mat("./weights/weights2.txt");
    // Mat **biases = init_biases();
    //
    // test_weights(weights, biases);

    Mat *A = mat_init(3, 4);
    Mat *B = mat_init(3, 4);
    mat_populate_rand(A);
    mat_populate_rand(B);

    mat_print(A);
    putchar('\n');
    mat_print(B);
    putchar('\n');

    Mat *C = schur_product(A, B);
    mat_print(C);
}

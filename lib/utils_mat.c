#include "utils_mat.h"

// Initializing a matrix struct to store the size of the matrix and its values
Mat *mat_init(int rows, int cols) {
    Mat *matrix = malloc(sizeof(Mat));
    if (matrix == NULL) {
        printf("Error allocating memory for matrix of size %d x %d\n", rows,
               cols);
        exit(EXIT_FAILURE);
    }
    matrix->rows = rows;
    matrix->cols = cols;

    double *valArr = malloc(rows * cols * sizeof(double));
    if (valArr == NULL) {
        perror("Error allocating memory for valArr\n");
        exit(EXIT_FAILURE);
    }

    matrix->values = malloc(rows * sizeof(double *));
    if (matrix->values == NULL) {
        perror("Error allocating memory for matrix->values\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < rows; i++) {
        matrix->values[i] = valArr + i * cols;
    }

    return matrix;
}

// Freeing a matrix struct and its components
void mat_free(Mat *m) {
    free(m->values[0]);
    free(m->values);
    free(m);
}

// Printing a matrix
void mat_print(Mat *m) {
    for (int i = 0; i < m->rows; i++) {
        // printf("r_%d: ", i);
        for (int j = 0; j < m->cols; j++) {
            printf("%.4lf", m->values[i][j]);
            if (j < m->cols - 1) {
                printf(",");
            }
        }
        printf("\n");
    }
}

// Printing a matrix as an ASCII image
void mat_printI(Mat *m) {
    for (int i = 0; i < m->rows; i++) {
        // printf("r_%d: ", i);
        for (int j = 0; j < m->cols; j++) {
            printf("%c", intToASCII(m->values[i][j]));
        }
        printf("\n");
    }
}

char intToASCII(int val) {
    char *gradient = " .,-~:;=!*#$@";
    int gradLeng = strlen(gradient);

    int valPerGrad = 255 / (gradLeng - 1);

    int index = val / valPerGrad;
    char ascii = gradient[index];
    return ascii;
}

// Populating every matrix value with a 'val' of choice
void mat_populate(Mat *m, double val) {
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            m->values[i][j] = val;
        }
    }
}

// Filling a matrix with the values m+n at [m][n] for tests
void mat_populate2(Mat *m) {
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            m->values[i][j] = i + j;
        }
    }
}

// Filling a matrix with random doubles from -1 to 1
void mat_populate_rand(Mat *m) {
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            m->values[i][j] = calc_random(-1, 1);
        }
    }
}

// Calculating a random double between lowerBound and upperBound
double calc_random(double lowerBound, double upperBound) {
    double difference = upperBound - lowerBound;

    // Generating a random double between 0 and 1
    double rando = (double)rand() / RAND_MAX;
    double adjusted = (difference * rando) + lowerBound;

    return adjusted;
}

// Flattening a matrix pointer m
void mat_flatten(Mat **m) {
    Mat *matrix = mat_init((*m)->rows * (*m)->cols, 1);

    for (int i = 0; i < (*m)->rows; i++) {
        for (int j = 0; j < (*m)->cols; j++) {
            matrix->values[i * (*m)->cols + j][0] = (*m)->values[i][j];
        }
    }

    mat_free((*m));
    *m = matrix;
}

// Unnflattening a matrix to one with cols columns
void mat_unflatten(Mat **m, int cols) {
    if ((*m)->rows % cols != 0) {
        printf(
            "A matrix with %d rows cannot be split into one with colums of "
            "length %d!\n",
            (*m)->rows, cols);
        exit(EXIT_FAILURE);
    }
    Mat *matrix = mat_init((*m)->rows / cols, cols);

    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            matrix->values[i][j] = (*m)->values[i * cols + j][0];
        }
    }

    mat_free((*m));
    *m = matrix;
}

// Performing and returning the matrix multiplication of B times A
Mat *mat_multiply(Mat *B, Mat *A) {
    if (B->cols != A->rows) {
        printf("Invalid matrix multiplication of %dx%d with %dx%d!\n", B->rows,
               B->cols, A->rows, A->cols);
        exit(EXIT_FAILURE);
    }
    Mat *matrix = mat_init(B->rows, A->cols);
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            double subTotal = 0;
            for (int k = 0; k < A->rows; k++) {
                subTotal += A->values[k][j] * B->values[i][k];
            }

            matrix->values[i][j] = subTotal;
        }
    }
    return matrix;
}

// Adding matrix src to matrix dest (changing matrix dest)
Mat *mat_add(Mat *dest, Mat *src) {
    if (dest->rows != src->rows || dest->cols != src->cols) {
        printf("Invalid addition of matrices with sizes %dx%d with %dx%d\n",
               dest->rows, dest->cols, src->rows, src->cols);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < dest->rows; i++) {
        for (int j = 0; j < dest->cols; j++) {
            dest->values[i][j] += src->values[i][j];
        }
    }

    return dest;
}

// Subtracting matrix src from matrix dest (dest = dest - src)(changing matrix
// dest)
Mat *mat_sub(Mat *dest, Mat *src) {
    if (dest->rows != src->rows || dest->cols != src->cols) {
        printf("Invalid subtraction of matrices with sizes %dx%d with %dx%d\n",
               dest->rows, dest->cols, src->rows, src->cols);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < dest->rows; i++) {
        for (int j = 0; j < dest->cols; j++) {
            dest->values[i][j] -= src->values[i][j];
        }
    }

    return dest;
}

// Returning the index of the greatest output values (the NN's guess)
int maxIndex(Mat *output) {
    if (output->cols != 1) {
        perror("Invalid number of columns for an output!\n");
        exit(EXIT_FAILURE);
    }
    double tmpMax = 0;
    int tmpMaxIndex = 0;
    for (int i = 0; i < output->rows; i++) {
        if (output->values[i][0] > tmpMax) {
            tmpMax = output->values[i][0];
            tmpMaxIndex = i;
        }
    }
    return tmpMaxIndex;
}

// Calculating and returning the Schur Product (AKA Hadamard Product) of two
// matrices A and B
Mat *schur_product(Mat *A, Mat *B) {
    if (A->rows != B->rows || A->cols != B->cols) {
        printf("Invalid Schur Product of %dx%d with %dx%d!\n", A->rows, A->cols,
               B->rows, B->cols);
        exit(EXIT_FAILURE);
    }

    Mat *results = mat_init(A->rows, A->cols);
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
            results->values[i][j] = A->values[i][j] * B->values[i][j];
        }
    }

    return results;
}

// Returning a matrix that is the transpose of M
Mat *mat_transpose(Mat *M) {
    Mat *mT = mat_init(M->cols, M->rows);

    for (int i = 0; i < mT->rows; i++) {
        for (int j = 0; j < mT->cols; j++) {
            mT->values[i][j] = M->values[j][i];
        }
    }

    return mT;
}

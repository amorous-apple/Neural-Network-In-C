#include "utils_mat.h"

// Initializing a matrix struct to store the size of the matrix an its values
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
        matrix->values[i] = valArr + i * rows;
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
            printf("%.1lf", m->values[i][j]);
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
    char *gradient = " ,-~:;=!*#$@";
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

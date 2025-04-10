#pragma once

#include <cstdio>
#include <cstdlib>

namespace cifar_sparse {

/* ========= 1D NDArray ========= */
typedef struct {
  int size;     // Length of the 1D array
  float *data;  // Pointer to contiguous block of floats
} Ndarray1D;

/* Create a new 1D NDArray */
Ndarray1D *create_ndarray1D(int size) {
  Ndarray1D *arr = (Ndarray1D *)malloc(sizeof(Ndarray1D));
  if (!arr) {
    fprintf(stderr, "Memory allocation failed for Ndarray1D\n");
    exit(EXIT_FAILURE);
  }
  arr->size = size;
  arr->data = (float *)malloc(size * sizeof(float));
  if (!arr->data) {
    fprintf(stderr, "Memory allocation failed for Ndarray1D data\n");
    free(arr);
    exit(EXIT_FAILURE);
  }
  // Initialize data to zero
  for (int i = 0; i < size; i++) {
    arr->data[i] = 0.0f;
  }
  return arr;
}

/* Get an element from the 1D NDArray */
float get_ndarray1D(Ndarray1D *arr, int i) {
  return arr->data[i];  // Boundary checking can be added if desired
}

/* Set an element in the 1D NDArray */
void set_ndarray1D(Ndarray1D *arr, int i, float value) { arr->data[i] = value; }

/* Free the 1D NDArray */
void free_ndarray1D(Ndarray1D *arr) {
  if (arr) {
    free(arr->data);
    free(arr);
  }
}

/* ========= 2D NDArray ========= */
typedef struct {
  int rows;
  int cols;
  float *data;  // Pointer to rows*cols floats stored in row-major order
} Ndarray2D;

/* Create a new 2D NDArray */
Ndarray2D *create_ndarray2D(int rows, int cols) {
  Ndarray2D *arr = (Ndarray2D *)malloc(sizeof(Ndarray2D));
  if (!arr) {
    fprintf(stderr, "Memory allocation failed for Ndarray2D\n");
    exit(EXIT_FAILURE);
  }
  arr->rows = rows;
  arr->cols = cols;
  arr->data = (float *)malloc(rows * cols * sizeof(float));
  if (!arr->data) {
    fprintf(stderr, "Memory allocation failed for Ndarray2D data\n");
    free(arr);
    exit(EXIT_FAILURE);
  }
  // Initialize data to zero
  for (int i = 0; i < rows * cols; i++) {
    arr->data[i] = 0.0f;
  }
  return arr;
}

/* Get an element from the 2D NDArray at (i, j) */
float get_ndarray2D(Ndarray2D *arr, int i, int j) { return arr->data[i * arr->cols + j]; }

/* Set an element in the 2D NDArray at (i, j) */
void set_ndarray2D(Ndarray2D *arr, int i, int j, float value) {
  arr->data[i * arr->cols + j] = value;
}

/* Free the 2D NDArray */
void free_ndarray2D(Ndarray2D *arr) {
  if (arr) {
    free(arr->data);
    free(arr);
  }
}

/* ========= 4D NDArray ========= */
typedef struct {
  int d0, d1, d2, d3;
  float *data;  // Pointer to d0*d1*d2*d3 floats stored in row-major order
} Ndarray4D;

/* Create a new 4D NDArray */
Ndarray4D *create_ndarray4D(int d0, int d1, int d2, int d3) {
  Ndarray4D *arr = (Ndarray4D *)malloc(sizeof(Ndarray4D));
  if (!arr) {
    fprintf(stderr, "Memory allocation failed for Ndarray4D\n");
    exit(EXIT_FAILURE);
  }
  arr->d0 = d0;
  arr->d1 = d1;
  arr->d2 = d2;
  arr->d3 = d3;

  int total_size = d0 * d1 * d2 * d3;
  arr->data = (float *)malloc(total_size * sizeof(float));
  if (!arr->data) {
    fprintf(stderr, "Memory allocation failed for Ndarray4D data\n");
    free(arr);
    exit(EXIT_FAILURE);
  }
  // Initialize data to zero
  for (int i = 0; i < total_size; i++) {
    arr->data[i] = 0.0f;
  }
  return arr;
}

/* Get an element from the 4D NDArray at (i, j, k, l) */
/* Offset calculation: offset = i*(d1*d2*d3) + j*(d2*d3) + k*d3 + l */
float get_ndarray4D(Ndarray4D *arr, int i, int j, int k, int l) {
  int offset = i * (arr->d1 * arr->d2 * arr->d3) + j * (arr->d2 * arr->d3) + k * (arr->d3) + l;
  return arr->data[offset];
}

/* Set an element in the 4D NDArray at (i, j, k, l) */
void set_ndarray4D(Ndarray4D *arr, int i, int j, int k, int l, float value) {
  int offset = i * (arr->d1 * arr->d2 * arr->d3) + j * (arr->d2 * arr->d3) + k * (arr->d3) + l;
  arr->data[offset] = value;
}

/* Free the 4D NDArray */
void free_ndarray4D(Ndarray4D *arr) {
  if (arr) {
    free(arr->data);
    free(arr);
  }
}

}  // namespace cifar_sparse

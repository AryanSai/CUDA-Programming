#include <stdio.h>
#include <cuda_runtime.h>

#define N 10000

// function to measure execution time
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void init_matrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

void matAdd_cpu(float *a, float *b, float *c, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            c[i * cols + j] = a[i * cols + j] + b[i * cols + j];
        }
    }
}

__global__ void matAdd_kernel(float *a, float *b, float *c, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        c[row * cols + col] = a[row * cols + col] + b[row * cols + col];
    }
}

float matAdd_gpu(float *h_a, float *h_b, float *h_c, int rows, int cols) {
    float *d_a, *d_b, *d_c;
    size_t size = rows * cols * sizeof(float);

    // allocate device memory
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // copy data to GPU
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // set up execution configuration
    const dim3 blockSize(16, 16);
    const dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

    double start_time = get_time();
    matAdd_kernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, rows, cols);
    cudaDeviceSynchronize();  // wait for GPU to finish
    double end_time = get_time();

    // copy from GPU to CPU
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return end_time - start_time;
}

int main() {
    printf("Matrix addition\n");
    int rows = N, cols = N;  
    size_t size = rows * cols * sizeof(float);
    srand(time(NULL));

    // allocate host memory
    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c = (float *)malloc(size);

    // initialize the matrices with random values
    init_matrix(h_a, rows, cols);
    init_matrix(h_b, rows, cols);

    double start_time = get_time();
    matAdd_cpu(h_a, h_b, h_c, rows, cols);
    double end_time = get_time();
    double cpu_time = end_time - start_time;
    printf("CPU Time: %f\n", cpu_time);

    double gpu_time = matAdd_gpu(h_a, h_b, h_c, rows, cols);
    printf("GPU Time: %f\n", gpu_time);

    printf("Speedup: %fx\n", cpu_time / gpu_time);
    
    free(h_a);
    free(h_b);
    free(h_c);
    return 0;
}

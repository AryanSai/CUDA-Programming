#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define N 10000000

// function to measure execution time
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void init_vector(float *vec) {
    for (unsigned int i = 0; i < N; i++) 
        vec[i] = (float) rand() / RAND_MAX;
}

void vecAdd_cpu(float *a, float *b, float *c) {
    for (unsigned int i = 0; i < N; i++) 
        c[i] = a[i] + b[i];
}

__global__ void vecAdd_kernel(float *a, float *b, float *c) {
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) 
        c[i] = a[i] + b[i];
}

float vecAdd_gpu(float *h_a, float *h_b, float *h_c, size_t size) {
    float *d_a, *d_b, *d_c;

    // allocate device memory
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);


    // copy data to GPU
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // perform addition
    const unsigned int numThreadsPerBlock = 256;
    const unsigned int numBlocks = (N + numThreadsPerBlock - 1) / numThreadsPerBlock;

    double start_time1 = get_time();
    vecAdd_kernel<<<numBlocks, numThreadsPerBlock>>>(d_a, d_b, d_c);
    double end_time1 = get_time();

    // copy from GPU to CPU
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return end_time1 - start_time1;
}

int main() {
    printf("Vector addition\n");
    size_t size = N * sizeof(float);
    srand(time(NULL));

    // Allocate host memory
    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c = (float *)malloc(size);

    // Initialize the vectors with random values
    init_vector(h_a);
    init_vector(h_b);

    double start_time = get_time();
    vecAdd_cpu(h_a, h_b, h_c);
    double end_time = get_time();
    printf("CPU Time: %f\n", end_time - start_time);

    double total_time = vecAdd_gpu(h_a, h_b, h_c, size);
    printf("GPU Time: %f\n", total_time);

    free(h_a);
    free(h_b);
    free(h_c);
    return 0;
}
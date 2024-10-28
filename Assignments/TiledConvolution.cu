#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>

#define IMAGE_WIDTH 1024
#define IMAGE_HEIGHT 1024
#define KERNEL_SIZE 3
#define TILE_SIZE 32

// function to measure execution time
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// initialize an image with random values
void init_image(float *img, int width, int height) {
    for (int i = 0; i < width * height; i++) {
        img[i] = (float)rand() / RAND_MAX;
    }
}

// CPU convolution function
void conv_cpu(float *img, float *kernel, float *output, int width, int height) {
    int half_kernel = KERNEL_SIZE / 2;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float sum = 0.0f;

            for (int ki = -half_kernel; ki <= half_kernel; ki++) {
                for (int kj = -half_kernel; kj <= half_kernel; kj++) {
                    int x = j + kj;
                    int y = i + ki;

                    if (x >= 0 && x < width && y >= 0 && y < height) {
                        sum += img[y * width + x] * kernel[(ki + half_kernel) * KERNEL_SIZE + (kj + half_kernel)];
                    }
                }
            }
            output[i * width + j] = sum;
        }
    }
}

// CUDA kernel for tiled convolution
__global__ void conv_kernel(float *img, float *kernel, float *output, int width, int height) {
    __shared__ float tile[TILE_SIZE + KERNEL_SIZE - 1][TILE_SIZE + KERNEL_SIZE - 1];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // lolad image tiles into shared memory
    for (int ky = 0; ky < KERNEL_SIZE; ky++) {
        for (int kx = 0; kx < KERNEL_SIZE; kx++) {
            int tile_x = threadIdx.x + kx;
            int tile_y = threadIdx.y + ky;

            int img_x = x + kx - KERNEL_SIZE / 2;
            int img_y = y + ky - KERNEL_SIZE / 2;

            if (tile_x < TILE_SIZE + KERNEL_SIZE - 1 && tile_y < TILE_SIZE + KERNEL_SIZE - 1) {
                if (img_x >= 0 && img_x < width && img_y >= 0 && img_y < height) {
                    tile[tile_y][tile_x] = img[img_y * width + img_x];
                } else {
                    tile[tile_y][tile_x] = 0.0f; //boundary condition
                }
            }
        }
    }
    __syncthreads();

    // perform convolution
    if (x < width && y < height) {
        float sum = 0.0f;
        for (int ky = 0; ky < KERNEL_SIZE; ky++) {
            for (int kx = 0; kx < KERNEL_SIZE; kx++) {
                sum += tile[threadIdx.y + ky][threadIdx.x + kx] *
                       kernel[ky * KERNEL_SIZE + kx];
            }
        }
        output[y * width + x] = sum;
    }
}

// GPU convolution function
double conv_gpu(float *h_img, float *h_kernel, float *h_output, int width, int height) {
    float *d_img, *d_kernel, *d_output;
    size_t img_size = width * height * sizeof(float);
    size_t kernel_size = KERNEL_SIZE * KERNEL_SIZE * sizeof(float);
    size_t output_size = img_size;

    // allocate device memory
    cudaMalloc((void **)&d_img, img_size);
    cudaMalloc((void **)&d_kernel, kernel_size);
    cudaMalloc((void **)&d_output, output_size);

    // copy data to GPU
    cudaMemcpy(d_img, h_img, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernel_size, cudaMemcpyHostToDevice);

    // set up execution configuration
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    double start_time = get_time();
    conv_kernel<<<gridSize, blockSize>>>(d_img, d_kernel, d_output, width, height);
    cudaDeviceSynchronize();
    double end_time = get_time();

    // copy result from GPU to CPU
    cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost);

    // free memory
    cudaFree(d_img);
    cudaFree(d_kernel);
    cudaFree(d_output);

    return end_time - start_time;
}

int main() {
    printf("Tiled 2D Convolution\n");
    int width = IMAGE_WIDTH, height = IMAGE_HEIGHT;
    size_t img_size = width * height * sizeof(float);
    size_t kernel_size = KERNEL_SIZE * KERNEL_SIZE * sizeof(float);
    
    // allocate host memory
    float *h_img = (float *)malloc(img_size);
    float *h_kernel = (float *)malloc(kernel_size);
    float *h_output_cpu = (float *)malloc(img_size);
    float *h_output_gpu = (float *)malloc(img_size);

    // initialize the image and kernel
    srand(time(NULL));
    init_image(h_img, width, height);
    for (int i = 0; i < KERNEL_SIZE * KERNEL_SIZE; i++) {
        h_kernel[i] = (float)rand() / RAND_MAX;
    }

    // measure CPU performance
    double start_time = get_time();
    conv_cpu(h_img, h_kernel, h_output_cpu, width, height);
    double end_time = get_time();
    double cpu_time = end_time - start_time;
    printf("CPU Time: %f seconds\n", cpu_time);

    // measure GPU performance
    double gpu_time = conv_gpu(h_img, h_kernel, h_output_gpu, width, height);
    printf("GPU Time: %f seconds\n", gpu_time);

    // speedup
    printf("Speedup: %fx\n", cpu_time / gpu_time);

    // free host memory
    free(h_img);
    free(h_kernel);
    free(h_output_cpu);
    free(h_output_gpu);

    return 0;
}

#include <stdio.h>

__global__ void whoami(void) {
    int block_id =
        blockIdx.x +    // apartment number
        blockIdx.y * gridDim.x +    // floor number in this building
        blockIdx.z * gridDim.x * gridDim.y;   // building number in this city

    int block_offset =
        block_id * // times our apartment number
        blockDim.x * blockDim.y * blockDim.z; // total threads per block (people per apartment)

    int thread_offset = 
        threadIdx.x +  
        threadIdx.y * blockDim.x +
        threadIdx.z * blockDim.x * blockDim.y;

    int id = block_offset + thread_offset; // global person id in the entire apartment complex

    printf("%04d | Block(%d %d %d) = %3d | Thread(%d %d %d) = %3d\n",
        id,
        blockIdx.x, blockIdx.y, blockIdx.z, block_id,
        threadIdx.x, threadIdx.y, threadIdx.z, thread_offset);
}

int main(int argc, char **argv) {

    //dimensions of the grid
    const int b_x = 2, b_y = 3, b_z = 4;
    //dimensions of each block
    const int t_x = 4, t_y = 4, t_z = 4; // the max warp size is 32
    // so we will get 2 warps of 32 threads per block

    int blocks_per_grid = b_x * b_y * b_z;
    int threads_per_block = t_x * t_y * t_z;

    printf("%d blocks/grid\n", blocks_per_grid);
    printf("%d threads/block\n", threads_per_block);
    printf("%d total threads\n", blocks_per_grid * threads_per_block);

    dim3 blocksPerGrid(b_x, b_y, b_z); // 3d cube of shape 2*3*4 = 24
    dim3 threadsPerBlock(t_x, t_y, t_z); // 3d cube of shape 4*4*4 = 64

    whoami<<<blocksPerGrid, threadsPerBlock>>>();
    cudaDeviceSynchronize();
}
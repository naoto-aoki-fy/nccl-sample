#include <mpi.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <vector>

namespace {
int rank = -1;

[[noreturn]] void fatal(const char* call, const char* message) {
    std::fprintf(stderr, "Rank %d: %s failed: %s\n", rank, call, message);
    std::fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void check_mpi(int result, const char* call) {
    if (result != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(result, message, &length);
        fatal(call, message);
    }
}

void check_cuda(cudaError_t result, const char* call) {
    if (result != cudaSuccess) fatal(call, cudaGetErrorString(result));
}

void check_nccl(ncclResult_t result, const char* call) {
    if (result != ncclSuccess) fatal(call, ncclGetErrorString(result));
}

#define MPI_CHECK(call) check_mpi((call), #call)
#define CUDA_CHECK(call) check_cuda((call), #call)
#define NCCL_CHECK(call) check_nccl((call), #call)

int run() {
    int world_size = 0;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));

    MPI_Comm local_comm = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                  MPI_INFO_NULL, &local_comm));
    MPI_CHECK(MPI_Comm_set_errhandler(local_comm, MPI_ERRORS_RETURN));
    int local_rank = 0;
    int local_size = 0;
    MPI_CHECK(MPI_Comm_rank(local_comm, &local_rank));
    MPI_CHECK(MPI_Comm_size(local_comm, &local_size));

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    // A scheduler may expose just one distinct GPU to each process.
    const int device = device_count == 1 ? 0 : local_rank;
    if (device_count == 0 || device >= device_count) {
        char message[256];
        std::snprintf(message, sizeof(message),
                      "local rank %d of %d sees %d CUDA devices; expose all local "
                      "GPUs in the same order, or one distinct GPU per process",
                      local_rank, local_size, device_count);
        fatal("GPU mapping", message);
    }
    CUDA_CHECK(cudaSetDevice(device));

    // Visible device indices can differ across processes; compare GPU UUIDs.
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    constexpr int uuid_bytes = sizeof(properties.uuid.bytes);
    std::vector<char> uuids(static_cast<std::size_t>(local_size) * uuid_bytes);
    MPI_CHECK(MPI_Allgather(properties.uuid.bytes, uuid_bytes, MPI_BYTE,
                            uuids.data(), uuid_bytes, MPI_BYTE, local_comm));
    for (int other = 0; other < local_size; ++other) {
        if (other != local_rank &&
            std::memcmp(properties.uuid.bytes,
                        uuids.data() + static_cast<std::size_t>(other) * uuid_bytes,
                        uuid_bytes) == 0) {
            char message[256];
            std::snprintf(message, sizeof(message),
                          "local ranks %d and %d selected the same GPU; assign "
                          "one distinct GPU to each process", local_rank, other);
            fatal("GPU mapping", message);
        }
    }
    MPI_CHECK(MPI_Comm_free(&local_comm));

    ncclUniqueId id{};
    if (rank == 0) NCCL_CHECK(ncclGetUniqueId(&id));
    MPI_CHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
    ncclComm_t comm = nullptr;
    NCCL_CHECK(ncclCommInitRank(&comm, world_size, id, rank));

    constexpr std::size_t count = 1024;
    constexpr std::size_t bytes = count * sizeof(float);
    std::vector<float> input(count, static_cast<float>(rank + 1));
    std::vector<float> output(count);
    float* device_input = nullptr;
    float* device_output = nullptr;
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_input), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_output), bytes));
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMemcpy(device_input, input.data(), bytes, cudaMemcpyHostToDevice));
    NCCL_CHECK(ncclAllReduce(device_input, device_output, count, ncclFloat,
                             ncclSum, comm, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(output.data(), device_output, bytes, cudaMemcpyDeviceToHost));

    const double expected = static_cast<double>(world_size) *
                            (static_cast<double>(world_size) + 1.0) / 2.0;
    const double tolerance = 1e-5 * std::max(1.0, std::abs(expected));
    std::size_t first_bad = count;
    for (std::size_t i = 0; i < count; ++i) {
        if (!std::isfinite(output[i]) ||
            std::abs(static_cast<double>(output[i]) - expected) > tolerance) {
            if (first_bad == count) first_bad = i;
        }
    }
    const bool passed = first_bad == count;
    const std::size_t reported_index = passed ? 0 : first_bad;
    std::printf("Rank %d: CUDA device %d, expected %.9g, actual[%zu] %.9g: %s\n",
                rank, device, expected, reported_index,
                static_cast<double>(output[reported_index]), passed ? "PASS" : "FAIL");
    std::fflush(stdout);

    CUDA_CHECK(cudaFree(device_output));
    CUDA_CHECK(cudaFree(device_input));
    CUDA_CHECK(cudaStreamDestroy(stream));
    NCCL_CHECK(ncclCommDestroy(comm));
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
}  // namespace

int main(int argc, char** argv) {
    // MPI error reporting/abort is only available after successful initialization.
    const int init_result = MPI_Init(&argc, &argv);
    if (init_result != MPI_SUCCESS) {
        std::fprintf(stderr, "Rank unknown: MPI_Init failed with code %d\n", init_result);
        return EXIT_FAILURE;
    }
    MPI_CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    int status = EXIT_FAILURE;
    try {
        status = run();
    } catch (const std::exception& error) {
        fatal("host operation", error.what());
    } catch (...) {
        fatal("host operation", "unknown exception");
    }
    const int finalize_result = MPI_Finalize();
    if (finalize_result != MPI_SUCCESS) {
        std::fprintf(stderr, "Rank %d: MPI_Finalize failed with code %d\n",
                     rank, finalize_result);
        return EXIT_FAILURE;
    }
    return status;
}

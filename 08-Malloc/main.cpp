#include "Pod.h"
#include "Ticktock.h"
#include <iostream>
#include <vector>

constexpr size_t n = 1 << 26;

void malloc_vector() {
    std::vector<int> arr(n);

    TICK(vector_write_first);
    for (size_t i = 0; i < n; ++i) {
        arr[i] = 1;
    }
    TOCK(vector_write_first);

    TICK(vector_write_second);
    for (size_t i = 0; i < n; i++) {
        arr[i] = 1;
    }
    TOCK(vector_write_second);

    std::cout << std::endl;
}

void malloc_malloc() {
    int *arr = (int *) malloc(n * sizeof(int));

    TICK(malloc_write_first);
    for (size_t i = 0; i < n; ++i) {
        arr[i] = 1;
    }
    TOCK(malloc_write_first);

    TICK(malloc_write_second);
    for (size_t i = 0; i < n; i++) {
        arr[i] = 1;
    }
    TOCK(malloc_write_second);

    free(arr);

    std::cout << std::endl;
}

void malloc_new() {
    int *arr = new int[n];

    TICK(new_write_first);
    for (size_t i = 0; i < n; ++i) {
        arr[i] = 1;
    }
    TOCK(new_write_first);

    TICK(new_write_second);
    for (size_t i = 0; i < n; i++) {
        arr[i] = 1;
    }
    TOCK(new_write_second);

    delete[] arr;

    std::cout << std::endl;
}

void malloc_new_0() {
    int *arr = new int[n]{};

    TICK(new0_write_first);
    for (size_t i = 0; i < n; ++i) {
        arr[i] = 1;
    }
    TOCK(new0_write_first);

    TICK(new0_write_second);
    for (size_t i = 0; i < n; i++) {
        arr[i] = 1;
    }
    TOCK(new0_write_second);

    delete[] arr;

    std::cout << std::endl;
}

void malloc_huge() {
    const size_t sz = 1ull << 31;
    int *arr = new int[sz];

    TICK(huge_write);
    for (size_t i = 0; i < 1024; ++i) {
        arr[i] = 1;
    }
    TOCK(huge_write);

    delete[] arr;

    std::cout << std::endl;
}

template<typename T>
struct NoInit {
    T value;

    NoInit() { /* Do nothing! */ }
};

void malloc_huge_vector() {
    const size_t sz = 1ull << 31;
    std::vector<NoInit<int>> arr(sz);

    TICK(huge_vector_write);
    for (size_t i = 0; i < 1024; ++i) {
        arr[i].value = 1;
    }
    TOCK(huge_vector_write);

    std::cout << std::endl;
}

void malloc_huge_vector_pod() {
    const size_t sz = 1ull << 31;
    std::vector<Pod<int>> arr(sz);

    TICK(huge_vector_pod_write);
    for (size_t i = 0; i < 1024; ++i) {
        arr[i] = 1;
    }
    TOCK(huge_vector_pod_write);

    std::cout << std::endl;
}

int main() {
    malloc_vector();
    malloc_malloc();
    malloc_new();
    malloc_new_0();
    malloc_huge();
    malloc_huge_vector();
    malloc_huge_vector_pod();

    return 0;
}

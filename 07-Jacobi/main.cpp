#include <cmath>
#include <iostream>
#include <string_view>
#include <vector>

#include "Ticktock.h"

constexpr size_t n = 1 << 26;

std::vector<float> a(n);
std::vector<float> b(n);

void prefill() {
#pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        a[i] = std::sin(i * 1.0f);
    }

    std::fill(b.begin(), b.end(), 0.f);
}

void calculate_loss(std::string_view name) {
    float loss = 0;

#pragma omp parallel for reduction(+ : loss)
    for (size_t i = 1; i < n - 1; ++i) {
        loss += std::pow(a[i - 1] + a[i + 1] - a[i] * 2, 2);
    }
    loss = std::sqrt(loss);
    std::cout << "loss for " << name << ": " << loss << "\n";
    std::cout << std::endl;
}

void Jacobi_original() {
    prefill();
    const size_t steps = 32;

    TICK(iter_original);
    for (int step = 0; step < steps; ++step) {
#pragma omp parallel for
        for (size_t i = 1; i < n - 1; ++i) {
            b[i] = (a[i - 1] + a[i + 1]) * 0.5f;
        }
        std::swap(a, b);
    }
    TOCK(iter_original);

    calculate_loss("iter_original");
}

void Jacobi_2_steps() {
    prefill();
    const size_t steps = 32 / 2;

    TICK(iter_2_steps);
    for (int step = 0; step < steps; step++) {
#pragma omp parallel for
        for (size_t i = 2; i < n - 2; i++) {
            b[i] = (a[i - 2] + a[i + 2]) * 0.25f + a[i] * 0.5f;
        }
        std::swap(a, b);
    }
    TOCK(iter_2_steps);

    calculate_loss("iter_2_steps");
}

void Jacobi_16_steps() {
    prefill();
    const size_t steps = 32 / 16;

    TICK(iter_16_steps);
    for (int step = 0; step < steps; ++step) {
#pragma omp parallel for
        for (size_t ibase = 16; ibase < n - 16; ibase += 32) {
            float ta[32 + 16 * 2], tb[32 + 16 * 2];
            for (intptr_t i = -16; i < 32 + 16; ++i) {
                ta[16 + i] = a[ibase + i];
            }

            for (intptr_t s = 1; s < 16; s += 2) {
                for (intptr_t i = -16 + s; i < 32 + 16 - s; i++) {
                    tb[16 + i] = (ta[16 + i - 1] + ta[16 + i + 1]) * 0.5f;
                }
                for (intptr_t i = -16 + s + 1; i < 32 + 16 - s - 1; i++) {
                    ta[16 + i] = (tb[16 + i - 1] + tb[16 + i + 1]) * 0.5f;
                }
            }

            for (intptr_t i = 0; i < 32; i++) {
                b[ibase + i] = tb[16 + i];
            }
        }

        std::swap(a, b);
    }
    TOCK(iter_16_steps);

    calculate_loss("iter_16_steps");
}

int main() {
    Jacobi_original();
    Jacobi_2_steps();
    Jacobi_16_steps();

    return 0;
}

#ifndef CACHE_BENCHMARKS_ALIGNEDALLOC_H
#define CACHE_BENCHMARKS_ALIGNEDALLOC_H

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>
#include <utility>

// https://stackoverflow.com/questions/12942548/making-stdvector-allocate-aligned-memory
namespace detail {
    void *allocate_aligned_memory(size_t align, size_t size) {
        return std::aligned_alloc(align, size);
    }

    void deallocate_aligned_memory(void *ptr) noexcept {
        std::free(ptr);
    }
}// namespace detail

template<typename T, size_t Align = 64>
class AlignedAllocator;

template<size_t Align>
class AlignedAllocator<void, Align> {
public:
    using pointer = void *;
    using const_pointer = const void *;
    using value_type = void;

    template<class U>
    struct rebind {
        using other = AlignedAllocator<U, Align>;
    };
};


template<typename T, size_t Align>
class AlignedAllocator {
public:
    using value_type = T;
    using pointer = T *;
    using const_pointer = const T *;
    using reference = T &;
    using const_reference = const T &;
    using size_type = size_t;
    using difference_type = ptrdiff_t;

    using propagate_on_container_move_assignment = std::true_type;

    template<class U>
    struct rebind {
        using other = AlignedAllocator<U, Align>;
    };

public:
    AlignedAllocator() noexcept = default;

    template<class U>
    explicit AlignedAllocator(const AlignedAllocator<U, Align> &) noexcept {}

    [[nodiscard]] size_type max_size() const noexcept {
        return (size_type(~0) - size_type(Align)) / sizeof(T);
    }

    pointer address(reference x) const noexcept {
        return std::addressof(x);
    }

    const_pointer address(const_reference x) const noexcept {
        return std::addressof(x);
    }

    pointer allocate(size_type n, typename AlignedAllocator<void, Align>::const_pointer = 0) {
        const auto alignment = static_cast<size_type>(Align);
        void *ptr = detail::allocate_aligned_memory(alignment, n * sizeof(T));
        if (ptr == nullptr) {
            throw std::bad_alloc();
        }

        return reinterpret_cast<pointer>(ptr);
    }

    void deallocate(pointer p, size_type) noexcept {
        return detail::deallocate_aligned_memory(p);
    }

    template<class U, class... Args>
    void construct(U *p, Args &&...args) {
        ::new (reinterpret_cast<void *>(p)) U(std::forward<Args>(args)...);
    }

    void destroy(pointer p) {
        p->~T();
    }
};


template<typename T, size_t Align>
class AlignedAllocator<const T, Align> {
public:
    using value_type = T;
    using pointer = const T *;
    using const_pointer = const T *;
    using reference = const T &;
    using const_reference = const T &;
    using size_type = size_t;
    using difference_type = ptrdiff_t;

    using propagate_on_container_move_assignment = std::true_type;

    template<class U>
    struct rebind {
        using other = AlignedAllocator<U, Align>;
    };

public:
    AlignedAllocator() noexcept = default;

    template<class U>
    explicit AlignedAllocator(const AlignedAllocator<U, Align> &) noexcept {}

    [[nodiscard]] size_type max_size() const noexcept {
        return (size_type(~0) - size_type(Align)) / sizeof(T);
    }

    const_pointer address(const_reference x) const noexcept {
        return std::addressof(x);
    }

    pointer allocate(size_type n, typename AlignedAllocator<void, Align>::const_pointer = 0) {
        const auto alignment = static_cast<size_type>(Align);
        void *ptr = detail::allocate_aligned_memory(alignment, n * sizeof(T));
        if (ptr == nullptr) {
            throw std::bad_alloc();
        }

        return reinterpret_cast<pointer>(ptr);
    }

    void deallocate(pointer p, size_type) noexcept {
        return detail::deallocate_aligned_memory(p);
    }

    template<class U, class... Args>
    void construct(U *p, Args &&...args) {
        ::new (reinterpret_cast<void *>(p)) U(std::forward<Args>(args)...);
    }

    void destroy(pointer p) {
        p->~T();
    }
};

template<typename T, size_t TAlign, typename U, size_t UAlign>
inline bool operator==(const AlignedAllocator<T, TAlign> &, const AlignedAllocator<U, UAlign> &) noexcept {
    return TAlign == UAlign;
}

template<typename T, size_t TAlign, typename U, size_t UAlign>
inline bool operator!=(const AlignedAllocator<T, TAlign> &, const AlignedAllocator<U, UAlign> &) noexcept {
    return TAlign != UAlign;
}

#endif//CACHE_BENCHMARKS_ALIGNEDALLOC_H

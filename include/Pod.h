#ifndef CACHE_BENCHMARKS_POD_H
#define CACHE_BENCHMARKS_POD_H

#include <new>
#include <type_traits>
#include <utility>

template<typename T>
class Pod {
private:
    T m_t;

public:
    Pod() = default;

    Pod(Pod &&src) noexcept(std::is_nothrow_move_constructible_v<T>) : m_t(std::move(src.m_t)) {}

    Pod(const Pod &src) : m_t(src.m_t) {}

    Pod &operator=(Pod &&other) noexcept(std::is_nothrow_move_assignable_v<T>) {
        if (this != &other) {
            m_t = std::move(other.m_t);
        }
        return *this;
    }

    Pod &operator=(const Pod &other) {
        if (this != &other) {
            m_t = other.m_t;
        }
        return *this;
    }

    explicit Pod(T &&t) noexcept(std::is_nothrow_move_constructible_v<T>) : m_t(std::move(t)) {}

    explicit Pod(const T &t) : m_t(t) {}

    Pod &operator=(T &&t) noexcept(std::is_nothrow_move_assignable_v<T>) {
        m_t = std::move(t);
        return *this;
    }

    Pod &operator=(const T &t) {
        m_t = t;
        return *this;
    }

    explicit operator const T &() const {
        return m_t;
    }

    explicit operator T &() {
        return m_t;
    }

    const T &get() const {
        return m_t;
    }

    T &get() {
        return m_t;
    }

    template<typename... Ts>
    Pod &emplace(Ts &&...ts) {
        static_assert(std::is_destructible_v<T>, "T must be destructible");
        static_assert(std::is_constructible_v<T, Ts &&...>, "T must be constructible with the given arguments");

        destroy();// optional
        ::new (static_cast<void *>(&m_t)) T(std::forward<Ts>(ts)...);
        return *this;
    }

    void destroy() {
        m_t.~T();
    }
};

#endif// CACHE_BENCHMARKS_POD_H

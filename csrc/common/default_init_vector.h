#pragma once

#include "common/compat.h"

#include <memory>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

namespace hnw {

// Allocator whose no-argument construct() default-initializes. Large scratch
// buffers that a parallel loop fully overwrites then skip the serial zero-fill
// (and its single-thread first touch) that std::vector<T>(n) would perform.
template <typename T, typename Base = std::allocator<T>> class DefaultInitAllocator : public Base {
    using traits = std::allocator_traits<Base>;

public:
    template <typename U> struct rebind {
        using other = DefaultInitAllocator<U, typename traits::template rebind_alloc<U>>;
    };

    using Base::Base;

    template <typename U>
    void construct(U* ptr) noexcept(std::is_nothrow_default_constructible<U>::value) {
        ::new (static_cast<void*>(ptr)) U;
    }

    template <typename U, typename... Args> void construct(U* ptr, Args&&... args) {
        traits::construct(static_cast<Base&>(*this), ptr, std::forward<Args>(args)...);
    }
};

template <typename T> using DefaultInitVector = std::vector<T, DefaultInitAllocator<T>>;

} // namespace hnw

//
// Created by 秋鱼头 on 2022/4/18.
//

#pragma once

#include <cuda.h>
#include "Defines.hpp"

namespace Agate {

template<typename T>
struct BufferView
{
    CUdeviceptr data                CONST_STATIC_INIT(0);
    unsigned int count              CONST_STATIC_INIT(0);
    unsigned short byte_stride      CONST_STATIC_INIT(0);
    unsigned short elmt_byte_size   CONST_STATIC_INIT(0);

    AGATE_CPUGPU bool isValid() const
    {
        return static_cast<bool>(data);
    }

    AGATE_CPUGPU explicit operator bool() const { return isValid(); }

    AGATE_CPUGPU const T& operator[](unsigned int idx) const
    {
        return *reinterpret_cast<T*>(data + idx * (byte_stride ? byte_stride : sizeof(T)));
    }
};

using GenericBufferView = BufferView<unsigned int>;

} // namespace Agate
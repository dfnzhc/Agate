//
// Created by 秋鱼头 on 2022/3/26.
//

#pragma once
#if defined(_MSC_VER)
/* Disable some warnings on MSVC++ */
#pragma warning(disable : 4127 4702 4100 4515 4800 4146 4512)
#define WIN32_LEAN_AND_MEAN     /* Don't ever include MFC on Windows */
#define NOMINMAX                /* Don't override min/max */
#endif

#include <iostream>
#include <memory>
#include <utility>
#include <algorithm>
#include <functional>
#include <fstream>
#include <execution>
#include <filesystem>
#include <string>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <stdexcept>
#include <queue>
#include <stack>
#include <array>
#include <numeric>
#include <map>
#include <Windows.h>
#include <winnt.h>
#include <windowsx.h>
#include <wrl.h>
#include <limits>
#include <set>
#include <ranges>
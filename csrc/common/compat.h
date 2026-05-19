#pragma once

// MSVC does not define ssize_t (POSIX type).
#ifdef _MSC_VER
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

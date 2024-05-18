#include <TH/THGeneral.h>

#ifdef __cplusplus
#include <c10/core/CPUAllocator.h>
#endif

#ifndef TH_HAVE_THREAD
#define __thread
#elif _MSC_VER
#define __thread __declspec( thread )
#endif

#if (defined(__unix) || defined(_WIN32))
  #if defined(__FreeBSD__)
    #include <malloc_np.h>
  #else
    #include <malloc.h>
  #endif
#elif defined(__APPLE__)
#include <malloc/malloc.h>
#endif

/* Torch Error Handling */
static void defaultErrorHandlerFunction(const char *msg, void *data)
{
  throw std::runtime_error(msg);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static THErrorHandlerFunction defaultErrorHandler = defaultErrorHandlerFunction;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static void *defaultErrorHandlerData;
// NOLINTNEXTLINE(modernize-use-nullptr,cppcoreguidelines-avoid-non-const-global-variables)
static __thread THErrorHandlerFunction threadErrorHandler = NULL;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static __thread void *threadErrorHandlerData;

void _THError(const char *file, const int line, const char *fmt, ...)
{
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,cppcoreguidelines-avoid-magic-numbers)
  char msg[2048];
  va_list args;

  /* vasprintf not standard */
  /* vsnprintf: how to handle if does not exists? */
  va_start(args, fmt);
  int n = vsnprintf(msg, 2048, fmt, args);
  va_end(args);

  if(n < 2048) {
    snprintf(msg + n, 2048 - n, " at %s:%d", file, line);
  }

  if (threadErrorHandler)
    (*threadErrorHandler)(msg, threadErrorHandlerData);
  else
    (*defaultErrorHandler)(msg, defaultErrorHandlerData);
  TH_UNREACHABLE;
}

void _THAssertionFailed(const char *file, const int line, const char *exp, const char *fmt, ...) {
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,cppcoreguidelines-avoid-magic-numbers)
  char msg[1024];
  va_list args;
  va_start(args, fmt);
  vsnprintf(msg, 1024, fmt, args);
  va_end(args);
  _THError(file, line, "Assertion `%s' failed. %s", exp, msg);
}

void THSetErrorHandler(THErrorHandlerFunction new_handler, void *data)
{
  threadErrorHandler = new_handler;
  threadErrorHandlerData = data;
}

void THSetDefaultErrorHandler(THErrorHandlerFunction new_handler, void *data)
{
  if (new_handler)
    defaultErrorHandler = new_handler;
  else
    defaultErrorHandler = defaultErrorHandlerFunction;
  defaultErrorHandlerData = data;
}

/* Torch Arg Checking Handling */
static void defaultArgErrorHandlerFunction(int argNumber, const char *msg, void *data)
{
  std::stringstream new_error;
  new_error << "invalid argument " << argNumber << ": " << msg;
  throw std::runtime_error(new_error.str());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static THArgErrorHandlerFunction defaultArgErrorHandler = defaultArgErrorHandlerFunction;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static void *defaultArgErrorHandlerData;
// NOLINTNEXTLINE(modernize-use-nullptr,cppcoreguidelines-avoid-non-const-global-variables)
static __thread THArgErrorHandlerFunction threadArgErrorHandler = NULL;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static __thread void *threadArgErrorHandlerData;

void _THArgCheck(const char *file, int line, int condition, int argNumber, const char *fmt, ...)
{
  if(!condition) {
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,cppcoreguidelines-avoid-magic-numbers)
    char msg[2048];
    va_list args;

    /* vasprintf not standard */
    /* vsnprintf: how to handle if does not exists? */
    va_start(args, fmt);
    int n = vsnprintf(msg, 2048, fmt, args);
    va_end(args);

    if(n < 2048) {
      snprintf(msg + n, 2048 - n, " at %s:%d", file, line);
    }

    if (threadArgErrorHandler)
      (*threadArgErrorHandler)(argNumber, msg, threadArgErrorHandlerData);
    else
      (*defaultArgErrorHandler)(argNumber, msg, defaultArgErrorHandlerData);
    TH_UNREACHABLE;
  }
}

void THSetArgErrorHandler(THArgErrorHandlerFunction new_handler, void *data)
{
  threadArgErrorHandler = new_handler;
  threadArgErrorHandlerData = data;
}

void THSetDefaultArgErrorHandler(THArgErrorHandlerFunction new_handler, void *data)
{
  if (new_handler)
    defaultArgErrorHandler = new_handler;
  else
    defaultArgErrorHandler = defaultArgErrorHandlerFunction;
  defaultArgErrorHandlerData = data;
}

// NOLINTNEXTLINE(modernize-use-nullptr,cppcoreguidelines-avoid-non-const-global-variables)
static __thread void (*torchGCFunction)(void *data) = NULL;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static __thread void *torchGCData;

/* Optional hook for integrating with a garbage-collected frontend.
 *
 * If torch is running with a garbage-collected frontend (e.g. Lua),
 * the GC isn't aware of TH-allocated memory so may not know when it
 * needs to run. These hooks trigger the GC to run in two cases:
 *
 * (1) When a memory allocation (malloc, realloc, ...) fails
 * (2) When the total TH-allocated memory hits a dynamically-adjusted
 *     soft maximum.
 */
void THSetGCHandler( void (*torchGCFunction_)(void *data), void *data )
{
  torchGCFunction = torchGCFunction_;
  torchGCData = data;
}

void* THAlloc(ptrdiff_t size)
{
  if(size < 0)
    THError("$ Torch: invalid memory size -- maybe an overflow?");

  return c10::alloc_cpu(size);
}

void* THRealloc(void *ptr, ptrdiff_t size)
{
  if(!ptr)
    return(THAlloc(size));

  if(size == 0)
  {
    THFree(ptr);
    // NOLINTNEXTLINE(modernize-use-nullptr)
    return NULL;
  }

  if(size < 0)
    THError("$ Torch: invalid memory size -- maybe an overflow?");

  // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
  void *newptr = realloc(ptr, size);

  if(!newptr && torchGCFunction) {
    torchGCFunction(torchGCData);
    // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
    newptr = realloc(ptr, size);
  }

  if(!newptr)
    THError("$ Torch: not enough memory: you tried to reallocate %dGB. Buy new RAM!", size/1073741824);

  return newptr;
}

void THFree(void *ptr)
{
  c10::free_cpu(ptr);
}

THDescBuff _THSizeDesc(const int64_t *size, const int64_t ndim) {
  const int L = TH_DESC_BUFF_LEN;
  THDescBuff buf;
  char *str = buf.str;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int64_t i;
  int64_t n = 0;
  n += snprintf(str, L-n, "[");

  for (i = 0; i < ndim; i++) {
    if (n >= L) break;
    n += snprintf(str+n, L-n, "%" PRId64, size[i]);
    if (i < ndim-1) {
      n += snprintf(str+n, L-n, " x ");
    }
  }

  if (n < L - 2) {
    snprintf(str+n, L-n, "]");
  } else {
    snprintf(str+L-5, 5, "...]");
  }

  return buf;
}

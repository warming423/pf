#include <pybind11/pybind11.h>
#include <torch/csrc/cuda/Stream.h>
#include <torch/csrc/cuda/Module.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/THP.h>

#include <c10/cuda/CUDAGuard.h>

#include <structmember.h>
#include <cuda_runtime_api.h>

PyObject *THCPStreamClass = nullptr;

static PyObject * THCPStream_pynew(
  PyTypeObject *type, PyObject *args, PyObject *kwargs) {
  HANDLE_TH_ERRORS

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int current_device;
  THCudaCheck(cudaGetDevice(&current_device));

  int priority = 0;
  uint64_t cdata = 0;

  // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
  static char *kwlist[] = {"priority", "_cdata", nullptr};
  if (!PyArg_ParseTupleAndKeywords(
      args, kwargs, "|iK", kwlist, &priority, &cdata)) {
    return nullptr;
  }

  THPObjectPtr ptr(type->tp_alloc(type, 0));
  if (!ptr) {
    return nullptr;
  }

  at::cuda::CUDAStream stream =
    cdata ?
    at::cuda::CUDAStream::unpack(cdata) :
    at::cuda::getStreamFromPool(
      /* isHighPriority */ priority < 0 ? true : false);

  THCPStream* self = (THCPStream *)ptr.get();
  self->cdata = stream.pack();
  new (&self->cuda_stream) at::cuda::CUDAStream(stream);

  return (PyObject *)ptr.release();
  END_HANDLE_TH_ERRORS
}

static void THCPStream_dealloc(THCPStream *self) {
  self->cuda_stream.~CUDAStream();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject * THCPStream_get_device(THCPStream *self, void *unused) {
  HANDLE_TH_ERRORS
  return THPDevice_New(self->cuda_stream.device());
  END_HANDLE_TH_ERRORS
}

static PyObject * THCPStream_get_cuda_stream(THCPStream *self, void *unused) {
  HANDLE_TH_ERRORS
  return PyLong_FromVoidPtr(self->cuda_stream.stream());
  END_HANDLE_TH_ERRORS
}

static PyObject * THCPStream_get_priority(THCPStream *self, void *unused) {
  HANDLE_TH_ERRORS
  return THPUtils_packInt64(self->cuda_stream.priority());
  END_HANDLE_TH_ERRORS
}

static PyObject * THCPStream_priority_range(PyObject *_unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int least_priority, greatest_priority;
  std::tie(least_priority, greatest_priority) =
    at::cuda::CUDAStream::priority_range();
  return Py_BuildValue("(ii)", least_priority, greatest_priority);
  END_HANDLE_TH_ERRORS
}

static PyObject * THCPStream_query(PyObject *_self, PyObject *noargs) {
  HANDLE_TH_ERRORS
  auto self = (THCPStream*)_self;
  return PyBool_FromLong(self->cuda_stream.query());
  END_HANDLE_TH_ERRORS
}

static PyObject * THCPStream_synchronize(PyObject *_self, PyObject *noargs) {
  HANDLE_TH_ERRORS
  {
    pybind11::gil_scoped_release no_gil;
    auto self = (THCPStream*)_self;
    self->cuda_stream.synchronize();
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject * THCPStream_eq(PyObject *_self, PyObject *_other) {
  HANDLE_TH_ERRORS
  auto self = (THCPStream*)_self;
  auto other = (THCPStream*)_other;
  return PyBool_FromLong(self->cuda_stream == other->cuda_stream);
  END_HANDLE_TH_ERRORS
}

// NOLINTNEXTLINE(modernize-avoid-c-arrays, cppcoreguidelines-avoid-non-const-global-variables, cppcoreguidelines-avoid-c-arrays)
static struct PyMemberDef THCPStream_members[] = {
  {nullptr}
};

// NOLINTNEXTLINE(modernize-avoid-c-arrays, cppcoreguidelines-avoid-non-const-global-variables, cppcoreguidelines-avoid-c-arrays)
static struct PyGetSetDef THCPStream_properties[] = {
  {"cuda_stream",
    (getter)THCPStream_get_cuda_stream, nullptr, nullptr, nullptr},
  {"priority", (getter)THCPStream_get_priority, nullptr, nullptr, nullptr},
  {nullptr}
};

// NOLINTNEXTLINE(modernize-avoid-c-arrays, cppcoreguidelines-avoid-non-const-global-variables, cppcoreguidelines-avoid-c-arrays)
static PyMethodDef THCPStream_methods[] = {
  {(char*)"query", THCPStream_query, METH_NOARGS, nullptr},
  {(char*)"synchronize",
    THCPStream_synchronize, METH_NOARGS, nullptr},
  {(char*)"priority_range",
    THCPStream_priority_range, METH_STATIC | METH_NOARGS, nullptr},
  {(char*)"__eq__", THCPStream_eq, METH_O, nullptr},
  {nullptr}
};

PyTypeObject THCPStreamType = {
  PyVarObject_HEAD_INIT(nullptr, 0)
  "torch._C._CudaStreamBase",            /* tp_name */
  sizeof(THCPStream),                    /* tp_basicsize */
  0,                                     /* tp_itemsize */
  (destructor)THCPStream_dealloc,        /* tp_dealloc */
  0,                                     /* tp_vectorcall_offset */
  // NOLINTNEXTLINE(modernize-use-nullptr)
  0,                                     /* tp_getattr */
  // NOLINTNEXTLINE(modernize-use-nullptr)
  0,                                     /* tp_setattr */
  // NOLINTNEXTLINE(modernize-use-nullptr)
  0,                                     /* tp_reserved */
  // NOLINTNEXTLINE(modernize-use-nullptr)
  0,                                     /* tp_repr */
  // NOLINTNEXTLINE(modernize-use-nullptr)
  0,                                     /* tp_as_number */
  // NOLINTNEXTLINE(modernize-use-nullptr)
  0,                                     /* tp_as_sequence */
  // NOLINTNEXTLINE(modernize-use-nullptr)
  0,                                     /* tp_as_mapping */
  // NOLINTNEXTLINE(modernize-use-nullptr)
  0,                                     /* tp_hash  */
  // NOLINTNEXTLINE(modernize-use-nullptr)
  0,                                     /* tp_call */
  // NOLINTNEXTLINE(modernize-use-nullptr)
  0,                                     /* tp_str */
  // NOLINTNEXTLINE(modernize-use-nullptr)
  0,                                     /* tp_getattro */
  // NOLINTNEXTLINE(modernize-use-nullptr)
  0,                                     /* tp_setattro */
  // NOLINTNEXTLINE(modernize-use-nullptr)
  0,                                     /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
  nullptr,                                  /* tp_doc */
  // NOLINTNEXTLINE(modernize-use-nullptr)
  0,                                     /* tp_traverse */
  // NOLINTNEXTLINE(modernize-use-nullptr)
  0,                                     /* tp_clear */
  // NOLINTNEXTLINE(modernize-use-nullptr)
  0,                                     /* tp_richcompare */
  0,                                     /* tp_weaklistoffset */
  // NOLINTNEXTLINE(modernize-use-nullptr)
  0,                                     /* tp_iter */
  // NOLINTNEXTLINE(modernize-use-nullptr)
  0,                                     /* tp_iternext */
  THCPStream_methods,                    /* tp_methods */
  THCPStream_members,                    /* tp_members */
  THCPStream_properties,                 /* tp_getset */
  // NOLINTNEXTLINE(modernize-use-nullptr)
  0,                                     /* tp_base */
  // NOLINTNEXTLINE(modernize-use-nullptr)
  0,                                     /* tp_dict */
  // NOLINTNEXTLINE(modernize-use-nullptr)
  0,                                     /* tp_descr_get */
  // NOLINTNEXTLINE(modernize-use-nullptr)
  0,                                     /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  // NOLINTNEXTLINE(modernize-use-nullptr)
  0,                                     /* tp_init */
  // NOLINTNEXTLINE(modernize-use-nullptr)
  0,                                     /* tp_alloc */
  THCPStream_pynew,                      /* tp_new */
};


void THCPStream_init(PyObject *module)
{
  Py_INCREF(THPStreamClass);
  THCPStreamType.tp_base = THPStreamClass;
  THCPStreamClass = (PyObject*)&THCPStreamType;
  if (PyType_Ready(&THCPStreamType) < 0) {
    throw python_error();
  }
  Py_INCREF(&THCPStreamType);
  if (PyModule_AddObject(
      module, "_CudaStreamBase", (PyObject *)&THCPStreamType) < 0) {
    throw python_error();
  }
}

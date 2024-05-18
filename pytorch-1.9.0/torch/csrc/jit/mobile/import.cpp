#include <torch/csrc/jit/mobile/import.h>

#include <ATen/core/ivalue.h>
#include <c10/util/ScopeExit.h>
#include <caffe2/serialize/inline_container.h>
#include <torch/csrc/jit/api/compilation_unit.h>
#include <torch/csrc/jit/mobile/interpreter.h>
#include <torch/csrc/jit/mobile/observer.h>
#include <torch/csrc/jit/runtime/instruction.h>
#include <torch/csrc/jit/serialization/import_export_constants.h>
#include <torch/csrc/jit/serialization/import_read.h>
#include <torch/custom_class.h>

#include <exception>
#include <fstream>
#include <string>
#include <vector>

// The import process to serialize the bytecode package.
// An example for bytecode.pkl of a small mobile_module looks like:
// (4,  # model version number (caffe2::serialize::kProducedBytecodeVersion)
//  # first method
//  (
//   # function name
//   '__torch__.m.forward',
//   # code
//   (('instructions',
//     (('STOREN', 1, 2),
//      ('DROPR', 1, 0),
//      ('MOVE', 2, 0),
//      ('OP', 0, 0),
//      ('RET', 0, 0))),
//    ('operators', (('aten::Int', 'Tensor'),)),
//    ('constants', ()),
//    ('types', ()),
//    ('register_size', 2)),
//   # schema -- optional (forward-compatible addition to version 4)
//   (('arguments',
//     ((('name', 'x'), ('type', 'Tensor'), ('default_value', 13)),
//      ...)),  # more args follow here
//    ('returns',
//     ((('name', ''), ('type', 'Tensor'), ('default_value', None)),
//      ...)),  # more return values follow here
//   )),
//  # more methods follow here
//  ...)

// In addition, the module debugging information can be saved
// in mobile_debug_handles.pkl. An example for it looks like:
// (4,
//  ('__torch__.m.forward',
//   (('module_debug_handles', 10))))
//   Here 10 is the debug handle.
// We also store separately and optionally callstack_debug_map.
// This serializes inlined callstack (InlinedCallStack data structure)
// corresponding to the debug handles.
// Callstack_debug_map serializes tuples of
// (int64_t(debug_handle), int64_t(source_range_tag), InlinedCallStack)
// source_range_tag maps to .debug_pkl files where this tag maps it to
// source range.
// InlinedCallStack is serialized as:
// IValue(InlinedCallStack) = {IValue(ModuleInstanceInfo),
// int64_t(source_range_tag), IValue(InlinedCallStack)} ModuleInstanceInfo is
// serialized as a tuple of (class_type_name, instance_name)

// Note that currently the backward compatibility is not supported by bytecode.
// This format and process need to be revisited and redesigned if we want to
// support backward compatibility in future.

// Note that the following function-schema fields are not supported:
//  - Argument::{known_length_,kwarg_only_}
//  - FunctionSchema::{overload_name_, is_vararg_, is_varret_}

namespace c10 {
// std::string serializeType(const Type &t);
TypePtr parseType(const std::string& pythonStr);
} // namespace c10

namespace torch {
namespace jit {
using caffe2::serialize::IStreamAdapter;
using caffe2::serialize::PyTorchStreamReader;
using caffe2::serialize::ReadAdapterInterface;

OpCode parseOpCode(const char* str);

IValue expect_field(
    IValue tup,
    const std::string& expected_name,
    size_t entry) {
  auto row = tup.toTuple()->elements().at(entry).toTuple();
  TORCH_INTERNAL_ASSERT(
      row->elements().at(0).toStringRef() == expected_name,
      "Expected ",
      expected_name,
      " found ",
      row->elements().at(0).toStringRef());
  return row->elements().at(1);
}

std::string operator_str(
    const std::string& name,
    const std::string& overloadname) {
  std::string result = name;
  if (!overloadname.empty()) {
    result += "." + overloadname;
  }
  return result;
}

TypePtr resolveTypeNameMobile(
    const c10::QualifiedName& qn,
    std::shared_ptr<CompilationUnit> compilation_unit) {
  // HACK: first we check whether the name starts with special prefix to
  // tell if it's a supported pytorch class type. There are two special
  // prefixes. "__torch__" for nn module, and "torch.jit" from to_backend.
  // This is a reliable
  // check today, but there is no guarantee that this is the case. The
  // real solution is to merge type parsers so we can share class
  // resolution logic.
  static const c10::QualifiedName torchPrefix = "__torch__";
  static const c10::QualifiedName jitPrefix = "torch.jit";
  if (torchPrefix.isPrefixOf(qn) || jitPrefix.isPrefixOf(qn)) {
    if (compilation_unit->get_class(qn) == nullptr) {
      auto typeptr = ClassType::create(qn, compilation_unit, true);
      compilation_unit->register_type(typeptr);
    }
    return compilation_unit->get_class(qn);
  } else {
    return c10::parseType(qn.qualifiedName());
  }
}

c10::StrongTypePtr typeResolverMobile(
    const c10::QualifiedName& qn,
    std::shared_ptr<CompilationUnit> compilation_unit) {
  return c10::StrongTypePtr(
      compilation_unit, resolveTypeNameMobile(qn, compilation_unit));
}

c10::intrusive_ptr<c10::ivalue::Object> objLoaderMobile(
    at::StrongTypePtr type,
    IValue input,
    std::shared_ptr<mobile::CompilationUnit> mobile_compilation_unit) {
  auto cls = type.type_->expect<at::ClassType>();
  auto qn = cls->name();
  c10::QualifiedName method_name(qn.value(), "__setstate__");
  auto setstate = mobile_compilation_unit->find_function(method_name);
  auto find_custom_class_with_setstate = [&qn]() -> c10::ClassTypePtr {
    auto custom_class_type = torch::jit::getCustomClass(qn->qualifiedName());
    if (custom_class_type && custom_class_type->findMethod("__setstate__")) {
      return custom_class_type;
    }
    return nullptr;
  };
  if (setstate) {
    auto obj = c10::ivalue::Object::create(type, 0);
    Stack stack({obj, input});
    setstate->run(stack);
    return obj;
  } else if (auto custom_class_type = find_custom_class_with_setstate()) {
    auto obj = c10::ivalue::Object::create(
        c10::StrongTypePtr(nullptr, custom_class_type), 1);
    Stack stack({obj, input});
    custom_class_type->getMethod("__setstate__").run(stack);
    return obj;
  } else {
    auto dict = std::move(input).toGenericDict();
    size_t ndict = dict.size();
    auto obj = c10::ivalue::Object::create(type, ndict);
    auto it = dict.begin();
    for (size_t i = 0; i < ndict; ++i) {
      std::stringstream name;
      name << it->key();
      cls->addOrCheckAttribute(name.str(), it->key().type());
      obj->setSlot(i, it->value());
      ++it;
    }
    return obj;
  }
}

bool isTensorInBytecodeArchive(
    caffe2::serialize::PyTorchStreamReader& stream_reader) {
  auto records = stream_reader.getAllRecords();
  for (const auto& record : records) {
    if (record.find("bytecode/") != std::string::npos) {
      return true;
    }
  }
  return false;
}

namespace {
void print_unsupported_ops_and_throw(
    const std::unordered_set<std::string>& unsupported_ops) {
  std::string error_message("{");
  for (const auto& op_name : unsupported_ops) {
    error_message += op_name + ", ";
  }
  error_message += "}";
  TORCH_CHECK(
      false,
      "Following ops cannot be found. ",
      "Check fburl.com/missing_ops for the fix.",
      error_message);
}

// The deserializer class which loads the bytecode package from bc files.
class BytecodeDeserializer final {
 public:
  explicit BytecodeDeserializer(
      std::unique_ptr<PyTorchStreamReader> reader,
      uint64_t module_load_options = 0);
  mobile::Module deserialize(c10::optional<at::Device> device);
  mobile::Module deserialize(
      c10::optional<at::Device> device,
      ExtraFilesMap& extra_files);
  std::unordered_map<std::string, std::string> deserializeMetadata(
      c10::optional<at::Device> device);
  void deserialize_only_extra(
      c10::optional<at::Device> device,
      ExtraFilesMap& extra_files);

 private:
  TypePtr resolveTypeName(const c10::QualifiedName& qn);
  void parseMethods(
      const std::vector<IValue>& vals,
      const c10::optional<std::vector<IValue>>& debug_handles,
      mobile::CompilationUnit& mcu);
  c10::IValue readArchive(
      const std::string& archive_name,
      std::shared_ptr<mobile::CompilationUnit> mcu);
  std::unordered_map<std::string, std::string> readMobileMetadata(
      std::shared_ptr<mobile::CompilationUnit> mcu);
  /**
   * Loads operators by looking them up in the Dispatcher and returns
   * the set of operator names (with overload) that are not supported
   * by the current runtime.
   */
  std::unordered_set<std::string> load_and_find_unsupported_operator_names(
      const std::vector<IValue>& ops_list,
      mobile::Function* function,
      int64_t model_version) const;
  std::shared_ptr<CompilationUnit> compilation_unit_;
  std::unordered_set<std::string> imported_libs_;
  std::unique_ptr<PyTorchStreamReader> reader_{};
  c10::optional<at::Device> device_;
  uint64_t module_load_options_;
};

BytecodeDeserializer::BytecodeDeserializer(
    std::unique_ptr<PyTorchStreamReader> reader,
    uint64_t module_load_options)
    : compilation_unit_(std::make_shared<CompilationUnit>()),
      reader_(std::move(reader)),
      module_load_options_(module_load_options) {}

std::unordered_set<std::string> BytecodeDeserializer::
    load_and_find_unsupported_operator_names(
        const std::vector<IValue>& ops_list,
        mobile::Function* function,
        int64_t model_version) const {
  std::unordered_set<std::string> unsupported_op_names;
  // ops_list is the list of operator names that were read in from
  // bytecode.plk for the method that is currently being processed.
  for (const auto& op : ops_list) {
    auto op_item = op.toTuple()->elements();
    TORCH_CHECK(
        op_item.size() >= 2,
        "There should be either two parts (name and overload name), ",
        "or three parts (name, overload name and number of specified args) ",
        "for an operator");
    c10::optional<int> num_args;
    if (op_item.size() > 2) {
      num_args = op_item[2].toInt();
    }
    auto op_found = function->append_operator(
        op_item[0].toString()->string(),
        op_item[1].toString()->string(),
        num_args,
        model_version);
    if (!op_found) {
      unsupported_op_names.emplace(operator_str(
          op_item[0].toString()->string(), op_item[1].toString()->string()));
    }
  }
  return unsupported_op_names;
}

TypePtr BytecodeDeserializer::resolveTypeName(const c10::QualifiedName& qn) {
  return resolveTypeNameMobile(qn, compilation_unit_);
}

void BytecodeDeserializer::parseMethods(
    const std::vector<IValue>& vals,
    const c10::optional<std::vector<IValue>>& debug_handles,
    mobile::CompilationUnit& mcu) {
  TORCH_CHECK(vals.size() > 0, "Bytecode has no elements. ");
  // Initialized with the version number when kProducedBytecodeVersion was
  // introduced. The old models (some of them already in production) without
  // version number don't have to be re-generated.
  int64_t model_version = 0x3L;
  size_t method_i_start = 0;
  if (vals[0].isInt()) {
    model_version = vals[0].toInt();
    method_i_start = 1;
  }
  TORCH_CHECK(
      // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
      caffe2::serialize::kMinSupportedBytecodeVersion <= model_version &&
          // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
          model_version <= caffe2::serialize::kProducedBytecodeVersion,
      "Lite Interpreter verson number does not match. ",
      "The model version must be between ",
      caffe2::serialize::kMinSupportedBytecodeVersion,
      " and ",
      caffe2::serialize::kProducedBytecodeVersion,
      "But the model version is ",
      model_version);

  bool has_debug_handles = debug_handles.has_value();
  if (has_debug_handles) {
    TORCH_CHECK(
        debug_handles->size() == vals.size(),
        "The numbers of bytecode values and debug info values do not match.");
  }

  // Process all methods in this mobile module.
  for (size_t i = method_i_start; i < vals.size(); ++i) {
    const auto& element = vals[i];
    const auto& m_tuple = element.toTuple()->elements();
    const std::string& function_name = m_tuple[0].toStringRef();
    IValue codeTable = m_tuple[1];
    auto schemaTable = // older files do not store function schema
        (model_version > 0x4L || (model_version == 0x4L && m_tuple.size() >= 3))
        ? at::optional<IValue>{m_tuple[2]}
        : at::nullopt;

    // NOLINTNEXTLINE(modernize-make-unique)
    auto function = std::unique_ptr<mobile::Function>(
        new mobile::Function(c10::QualifiedName(function_name)));

    const auto& ins_list =
        expect_field(codeTable, "instructions", BYTECODE_INDEX_INSTRUCTION)
            .toTuple()
            ->elements();
    const auto& ops_list =
        expect_field(codeTable, "operators", BYTECODE_INDEX_OPERATOR)
            .toTuple()
            ->elements();
    const auto& consts_list =
        expect_field(codeTable, "constants", BYTECODE_INDEX_CONSTANT)
            .toTuple()
            ->elements();
    const auto& types_list =
        expect_field(codeTable, "types", BYTECODE_INDEX_TYPE)
            .toTuple()
            ->elements();
    const auto& register_size =
        expect_field(codeTable, "register_size", BYTECODE_INDEX_REGISTER_SIZE)
            .toInt();

    std::vector<IValue> debug_handles_list;
    if (has_debug_handles) {
      const auto& debug_handles_element = (*debug_handles)[i];
      const auto& debug_handles_m_tuple =
          debug_handles_element.toTuple()->elements();
      const std::string& debug_info_function_name =
          debug_handles_m_tuple[0].toStringRef();
      TORCH_CHECK(
          debug_info_function_name == function_name,
          "The function names in the bytecode table and the debug info table do not match.");
      IValue debug_handles_table = debug_handles_m_tuple[1];
      debug_handles_list = (expect_field(
                                debug_handles_table,
                                "function_debug_handles",
                                BYTECODE_INDEX_MODULE_DEBUG_HANDLES)
                                .toTuple()
                                ->elements())[0]
                               .toList()
                               .vec();
      TORCH_CHECK(
          debug_handles_list.size() == ins_list.size(),
          "The numbers of instructions and debug handles strings do not match.");
    }

    for (size_t i = 0; i < ins_list.size(); ++i) {
      auto ins_item = ins_list[i].toTuple()->elements();
      TORCH_CHECK(
          ins_item.size() == 3,
          "There should be three parts in an instruction. The function name is ",
          function_name);
      OpCode op_code = parseOpCode(ins_item[0].toString()->string().c_str());
      int X = ins_item[1].toInt();
      int N = ins_item[2].toInt();
      if (has_debug_handles) {
        int64_t debug_handle = debug_handles_list[i].toInt();
        function->append_instruction(op_code, X, N, debug_handle);
      } else {
        function->append_instruction(op_code, X, N);
      }
    }

    std::unordered_set<std::string> unsupported_op_names =
        load_and_find_unsupported_operator_names(
            ops_list, function.get(), model_version);
    if ((module_load_options_ & MobileModuleLoadOptions::OPERATOR_CHECK) &&
        !unsupported_op_names.empty()) {
      print_unsupported_ops_and_throw(unsupported_op_names);
    }

    for (const auto& constant : consts_list) {
      function->append_constant(constant);
    }

    static const c10::QualifiedName classPrefix = "__torch__.torch.classes";
    for (const auto& t : types_list) {
      c10::QualifiedName qn(t.toStringRef());
      if (classPrefix.isPrefixOf(qn)) {
        auto classType = getCustomClass(qn.qualifiedName());
        TORCH_CHECK(
            classType,
            "The implementation of class ",
            qn.qualifiedName(),
            " cannot be found.");
        function->append_type(classType);
      } else {
        function->append_type(c10::parseType(t.toStringRef()));
      }
    }

    function->set_register_size(register_size);

    // function schema
    if (schemaTable) { // (schema is optional for back compat)
      auto parseArgList = [this](const std::vector<IValue>& argTables) {
        std::vector<c10::Argument> args;
        for (auto&& argTable : argTables) {
          auto name =
              expect_field(argTable, "name", BYTECODE_INDEX_ARGUMENT_NAME)
                  .toStringRef();
          const auto& type = resolveTypeName(
              (expect_field(argTable, "type", BYTECODE_INDEX_ARGUMENT_TYPE))
                  .toStringRef());
          auto default_value = expect_field(
                                   argTable,
                                   "default_value",
                                   BYTECODE_INDEX_ARGUMENT_DEFAULT_VALUE)
                                   .toIValue();
          auto arg =
              c10::Argument(name, type, c10::nullopt /*N*/, default_value);
          args.emplace_back(std::move(arg));
        }
        return args;
      };
      const auto& arg_list =
          expect_field(
              *schemaTable, "arguments", BYTECODE_INDEX_SCHEMA_ARGUMENTS)
              .toTuple()
              ->elements();
      const auto& ret_list =
          expect_field(*schemaTable, "returns", BYTECODE_INDEX_SCHEMA_RETURNS)
              .toTuple()
              ->elements();
      c10::FunctionSchema schema(
          function_name,
          "" /*overload_name*/,
          parseArgList(arg_list),
          parseArgList(ret_list),
          false /*is_varargs*/,
          false /*is_varret*/);
      function->setSchema(std::move(schema));
    }

    mcu.register_function(std::move(function));
  }
}

std::unordered_map<std::string, std::string> BytecodeDeserializer::
    deserializeMetadata(c10::optional<at::Device> device) {
  device_ = device;
  auto mcu = std::make_shared<mobile::CompilationUnit>();
  return readMobileMetadata(mcu);
}

void BytecodeDeserializer::deserialize_only_extra(
    c10::optional<at::Device> device,
    ExtraFilesMap& extra_files) {
  device_ = device;
  for (const auto& kv : extra_files) {
    const std::string& key = "extra/" + kv.first;
    if (reader_->hasRecord(key)) {
      at::DataPtr meta_ptr;
      size_t meta_size = 0;
      std::tie(meta_ptr, meta_size) = reader_->getRecord(key);
      extra_files[kv.first] =
          std::string(static_cast<char*>(meta_ptr.get()), meta_size);
    }
  }
}

mobile::Module BytecodeDeserializer::deserialize(
    c10::optional<at::Device> device,
    ExtraFilesMap& extra_files) {
  deserialize_only_extra(device, extra_files);
  return deserialize(device);
}

mobile::Module BytecodeDeserializer::deserialize(
    c10::optional<at::Device> device) {
  device_ = device;
  auto mcu = std::make_shared<mobile::CompilationUnit>();

  // bvals can have 2 possible formats:
  //
  // 1. Old format: bvals is an array (Tuple) of N elements, each element being
  // itself a Tuple(method_name, method_table).
  //
  // 2. New format: bvals is an array (Tuple) of 1+N elements. The first element
  // being a Tuple (int, table), and the integer stands for the bytecode version
  // number. The rest of the elements are the same as before.
  //
  auto bvals = readArchive("bytecode", mcu).toTuple()->elements();

  c10::optional<std::vector<IValue>> debug_handles;
  if (reader_->hasRecord("mobile_debug_handles.pkl")) {
    debug_handles =
        readArchive("mobile_debug_handles", mcu).toTuple()->elements();
  }
  parseMethods(bvals, debug_handles, *mcu);
  auto meta_dict = readMobileMetadata(mcu);
  auto m = mobile::Module(readArchive("data", mcu).toObject(), meta_dict, mcu);
#if defined(SYMBOLICATE_MOBILE_DEBUG_HANDLE)
  MobileDebugTable debug_table = MobileDebugTable(reader_, compilation_unit_);
  m.setDebugTable(std::move(debug_table));
#endif
  return m;
}

std::unordered_map<std::string, std::string> BytecodeDeserializer::
    readMobileMetadata(std::shared_ptr<mobile::CompilationUnit> mcu) {
  std::unordered_map<std::string, std::string> res;
  if (!reader_->hasRecord("metadata.pkl")) {
    return res;
  }
  auto ivalue_dict = readArchive("metadata", mcu).toGenericDict();
  for (const auto& it : ivalue_dict) {
    const auto key = it.key().toString()->string();
    const auto value = it.value().toString()->string();
    res[key] = value;
  }
  return res;
}

c10::IValue BytecodeDeserializer::readArchive(
    const std::string& archive_name,
    std::shared_ptr<mobile::CompilationUnit> mcu) {
  auto type_resolver = [this](const c10::QualifiedName& qn) {
    return typeResolverMobile(qn, compilation_unit_);
  };

  auto obj_loader = [&](at::StrongTypePtr type, IValue input) {
    return objLoaderMobile(type, input, mcu);
  };

  bool bytecode_tensor_in_constants_archive =
      (archive_name == "bytecode" &&
       !isTensorInBytecodeArchive(*reader_.get()));

  auto ivalues = torch::jit::readArchiveAndTensors(
      archive_name,
      /*pickle_prefix=*/"",
      /*tensor_prefix=*/
      bytecode_tensor_in_constants_archive ? "constants/" : "",
      type_resolver,
      obj_loader,
      device_,
      *reader_.get());
  return ivalues;
}

} // namespace

// Forward declare so that _load_for_mobile() overloads can
// call this method directly.
mobile::Module _load_for_mobile_impl(
    std::unique_ptr<ReadAdapterInterface> rai,
    c10::optional<c10::Device> device,
    ExtraFilesMap& extra_files,
    uint64_t module_load_options);

mobile::Module _load_for_mobile(
    std::istream& in,
    c10::optional<at::Device> device) {
  ExtraFilesMap extra_files;
  return _load_for_mobile(in, device, extra_files);
}

mobile::Module _load_for_mobile(
    const std::string& filename,
    c10::optional<at::Device> device) {
  ExtraFilesMap extra_files;
  return _load_for_mobile(filename, device, extra_files);
}

mobile::Module _load_for_mobile(
    std::unique_ptr<ReadAdapterInterface> rai,
    c10::optional<c10::Device> device) {
  ExtraFilesMap extra_files;
  return _load_for_mobile(std::move(rai), device, extra_files);
}

mobile::Module _load_for_mobile(
    std::istream& in,
    c10::optional<at::Device> device,
    ExtraFilesMap& extra_files) {
  std::unique_ptr<IStreamAdapter> rai = std::make_unique<IStreamAdapter>(&in);
  auto module = _load_for_mobile(std::move(rai), device, extra_files);
  return module;
}

mobile::Module _load_for_mobile(
    const std::string& filename,
    c10::optional<at::Device> device,
    ExtraFilesMap& extra_files) {
  std::unique_ptr<FileAdapter> rai = std::make_unique<FileAdapter>(filename);
  auto module = _load_for_mobile(std::move(rai), device, extra_files);
  return module;
}

mobile::Module _load_for_mobile(
    const std::string& filename,
    c10::optional<at::Device> device,
    ExtraFilesMap& extra_files,
    uint64_t module_load_options) {
  std::unique_ptr<FileAdapter> rai = std::make_unique<FileAdapter>(filename);
  auto module = _load_for_mobile_impl(
      std::move(rai), device, extra_files, module_load_options);
  return module;
}

mobile::Module _load_for_mobile(
    std::unique_ptr<ReadAdapterInterface> rai,
    c10::optional<c10::Device> device,
    ExtraFilesMap& extra_files) {
  auto module = _load_for_mobile_impl(
      std::move(rai), device, extra_files, _default_mobile_module_load_options);
  return module;
}

mobile::Module _load_for_mobile_impl(
    std::unique_ptr<ReadAdapterInterface> rai,
    c10::optional<c10::Device> device,
    ExtraFilesMap& extra_files,
    uint64_t module_load_options) {
  auto observer = torch::observerConfig().getModuleObserver();
  // NOLINTNEXTLINE(clang-analyzer-security.insecureAPI.rand)
  auto instance_key = std::rand();
  if (observer) {
    observer->onEnterLoadModel(instance_key);
  }
  const size_t model_size = rai != nullptr ? rai->size() : 0;
  auto reader = torch::make_unique<PyTorchStreamReader>(std::move(rai));
  BytecodeDeserializer deserializer(std::move(reader), module_load_options);
  std::string error_message;
  auto guard = c10::make_scope_exit([&]() {
    if (!observer) {
      return;
    }

    observer->onFailLoadModel(
        instance_key,
        error_message.empty() ? "Unknown exception" : error_message.c_str(),
        deserializer.deserializeMetadata(device));
  });

  try {
    mobile::Module result = deserializer.deserialize(device, extra_files);
    std::unordered_map<std::string, std::string> copied_metadata =
        result.metadata();
    if (result.metadata().find("model_name") == result.metadata().end()) {
      copied_metadata["model_name"] = result.name();
    }
    copied_metadata["model_size"] = c10::guts::to_string(model_size);
    if (observer) {
      observer->onExitLoadModel(instance_key, copied_metadata);
    }
    guard.release();
    return result;
  } catch (c10::Error& error) {
    error_message = error.what();
    TORCH_RETHROW(error);
  }
}

void _load_extra_only_for_mobile(
    const std::string& filename,
    c10::optional<at::Device> device,
    ExtraFilesMap& extra_files) {
  std::unique_ptr<FileAdapter> rai = std::make_unique<FileAdapter>(filename);
  auto observer = torch::observerConfig().getModuleObserver();
  // NOLINTNEXTLINE(clang-analyzer-security.insecureAPI.rand)
  auto instance_key = std::rand();
  if (observer) {
    observer->onEnterLoadModel(instance_key);
  }
  auto reader = torch::make_unique<PyTorchStreamReader>(std::move(rai));
  BytecodeDeserializer deserializer(std::move(reader));
  deserializer.deserialize_only_extra(device, extra_files);
}

namespace mobile {

std::set<std::string> _export_operator_list(
    torch::jit::mobile::Module& module) {
  std::set<std::string> operator_list;
  for (Method func : module.get_methods()) {
    const Function& function = func.function();
    const std::shared_ptr<Code> cptr = function.get_code();
    // op_names below isn't a list of unique operator names. In fact
    // it can contain the same operator name many many times, so we need
    // to de-dup the list by adding all the operator names into
    // an std::set<std::string>.
    std::vector<c10::OperatorName> const& op_names = cptr->op_names_;
    for (auto& op_name : op_names) {
      operator_list.insert(toString(op_name));
    }
  }
  return operator_list;
}

} // namespace mobile
} // namespace jit
} // namespace torch

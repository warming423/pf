#include <gtest/gtest.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/runtime/static/fusion.h>
#include <torch/csrc/jit/runtime/static/impl.h>
#include <torch/csrc/jit/runtime/static/passes.h>
#include "deep_wide_pt.h"
#include "test_scripts.h"

using namespace caffe2;
using namespace torch;
using namespace torch::jit;
using c10::IValue;

namespace {
static at::Tensor getTensor(const at::IValue& ival) {
  if (ival.isTensor()) {
    return ival.toTensor();
  } else if (ival.isTensorList()) {
    auto tensor_vec = ival.toTensorVector();
    TORCH_CHECK(tensor_vec.size() == 1);
    return tensor_vec[0];
  } else if (ival.isTuple()) {
    auto tuple = ival.toTuple();
    auto ivalue_vec = tuple->elements();
    TORCH_CHECK(ivalue_vec.size() == 1);
    return ivalue_vec[0].toTensor();
  } else {
    CAFFE_THROW("Unknown input IValue");
  }
}

void compareTensorLists(
    const std::vector<IValue>& l, /* expects */
    const std::vector<IValue>& r /* values */) {
  EXPECT_TRUE(l.size() == r.size());
  for (int i = 0; i < l.size(); ++i) {
    ASSERT_TRUE(l[i].isTensor());
    ASSERT_TRUE(r[i].isTensor());
    VLOG(2) << "expect " << i << ": \n" << l[i] << std::endl;
    VLOG(2) << "output " << i << ": \n" << r[i] << std::endl;
    if (!l[i].toTensor().defined()) {
      EXPECT_TRUE(!r[i].toTensor().defined());
    } else {
      EXPECT_TRUE(l[i].toTensor().equal(r[i].toTensor()));
    }
  }
}

void compareTensorLists(
    const std::vector<at::Tensor>& l, /* expects */
    const std::vector<at::Tensor>& r /* values */) {
  EXPECT_TRUE(l.size() == r.size());
  for (int i = 0; i < l.size(); ++i) {
    VLOG(2) << "expect " << i << ": \n" << l[i] << std::endl;
    VLOG(2) << "output " << i << ": \n" << r[i] << std::endl;
    if (!l[i].defined()) {
      EXPECT_TRUE(!r[i].defined());
    } else {
      EXPECT_TRUE(l[i].equal(r[i]));
    }
  }
}

// Given a model/function in jit script, run the model/function
// with the jit interpreter and static runtime, and compare the results
void testStaticRuntime(
    const std::string& jit_script,
    const std::vector<IValue>& args) {
  script::Module module("module");
  module.define(jit_script);

  std::vector<IValue> args_tensors, args_copy;
  for (const auto& ival : args) {
    if (ival.isTensor()) {
      args_tensors.emplace_back(ival);
      const at::Tensor& t = ival.toTensor();
      args_copy.emplace_back(t.clone());
    }
  }

  auto expect = module.forward(args);

  torch::jit::StaticModule smodule(module);
  auto actual = smodule(args, {});
  smodule.runtime().check_for_memory_leak();

  if (expect.isTuple()) {
    compareTensorLists(
        expect.toTuple()->elements(), actual.toTuple()->elements());
  } else if (expect.isList()) {
    compareTensorLists(expect.toTensorVector(), actual.toTensorVector());
  } else {
    VLOG(2) << "expect " << expect.toTensor() << std::endl;
    VLOG(2) << "output " << actual.toTensor() << std::endl;
    EXPECT_TRUE(expect.toTensor().equal(actual.toTensor()));
  }
  // make sure inputs were not modified
  compareTensorLists(args_tensors, args_copy);
}

bool testHasInplaceOp(const std::string& jit_script) {
  script::Module module("module");
  module.define(jit_script);

  Method method = module.get_method("forward");
  auto graph = module.get_method("forward").graph();

  torch::jit::AliasDb alias_db(graph);
  return torch::jit::HasInplaceOp(graph, alias_db);
}
} // namespace

TEST(StaticRuntime, InPlace) {
  EXPECT_TRUE(testHasInplaceOp(reshape_inplace_script));
  EXPECT_TRUE(testHasInplaceOp(sigmoid_inplace_script));
  EXPECT_FALSE(testHasInplaceOp(sigmoid_out_script));
}

TEST(StaticRuntime, UnaryOps) {
  auto a = at::randn({2, 3});

  std::vector<IValue> args{a};

  // sum
  testStaticRuntime(aten_sum, args);
  testStaticRuntime(aten_sum_0, args);
  testStaticRuntime(aten_sum_1, args);
  testStaticRuntime(aten_sum_0_true, args);
  testStaticRuntime(aten_sum_1_true, args);
}

TEST(StaticRuntime, Clone) {
  auto a = at::randn({2, 3});
  auto b = at::empty_strided({3, 2}, {1, 3});

  std::vector<IValue> args_0{b, c10::MemoryFormat::Contiguous};
  std::vector<IValue> args_1{b, c10::MemoryFormat::Preserve};

  testStaticRuntime(clone_script_0, {a});
  testStaticRuntime(clone_script_1, args_0);
  testStaticRuntime(clone_script_1, args_1);
}

TEST(StaticRuntime, Clamp) {
  auto a = at::randn({2, 3});
  auto max_t = at::full_like(a, 1);
  auto min_t = at::full_like(a, -1);

  testStaticRuntime(clamp_script_1, {a, -1, 1});
  testStaticRuntime(clamp_script_2, {a, min_t, max_t});
}

TEST(StaticRuntime, Logit) {
  auto a = at::ones({2, 3});
  double b = 1e-6;
  std::vector<IValue> args_1{a};
  std::vector<IValue> args_2({a, b});

  // logit
  testStaticRuntime(logit_script_1, args_1);
  testStaticRuntime(logit_script_2, args_1);
  testStaticRuntime(logit_script_3, args_2);
}

TEST(StaticRuntime, EmbeddingBag) {
  at::Tensor weight = torch::randn({3, 11}, at::ScalarType::Float);
  at::Tensor input = torch::tensor({0, 1, 0, 2});
  at::Tensor offset = torch::tensor({0, 2, 4});

  std::vector<IValue> args{weight, input, offset};

  testStaticRuntime(embedding_bag_default, args);
  testStaticRuntime(embedding_bag_mean, args);
  testStaticRuntime(embedding_bag_max, args);
  testStaticRuntime(embedding_bag_sum_last_offset, args);
  testStaticRuntime(embedding_bag_mean_last_offset, args);
  testStaticRuntime(embedding_bag_max_last_offset, args);
}

TEST(StaticRuntime, LayerNorm) {
  const auto input = torch::rand({20, 10, 10, 10});
  for (int normalized_size: {2, 3}) {
      std::vector<int64_t> normalized_shape(normalized_size, 10);
      const auto weight = torch::rand(normalized_shape);
      const auto bias = torch::rand(normalized_shape);
      std::vector<IValue> args{input, normalized_shape, weight, bias};
      testStaticRuntime(layer_norm_with_weights, args);
      args = {input, normalized_shape};
      testStaticRuntime(layer_norm_without_weights, args);
  }
}

TEST(StaticRuntime, IndividualOps_Binary) {
  auto a = at::randn({2, 3});
  auto b = at::ones({2, 3});

  std::vector<IValue> args{a, b};

  testStaticRuntime(add_script, args);
  testStaticRuntime(list_construct_script, args);
  testStaticRuntime(list_construct_script_2, args);
  testStaticRuntime(list_construct_script_3, args);
  testStaticRuntime(list_unpack_script, args);
  testStaticRuntime(list_unpack_script_2, args);
  testStaticRuntime(tuple_construct_script, args);
  testStaticRuntime(tuple_construct_script_2, args);
}

TEST(StaticRuntime, IndividualOps_Binary_MatMul) {
  // 1-D, 1-D
  std::vector<IValue> args{at::randn({3}), at::randn({3})};
  testStaticRuntime(aten_matmul, args);
  // 2-D, 2-D
  args = {at::randn({3, 2}), at::randn({2, 3})};
  testStaticRuntime(aten_matmul, args);
  // 1-D, 2-D
  args = {at::randn({3}), at::randn({3, 5})};
  testStaticRuntime(aten_matmul, args);
  // 2-D, 1-D
  args = {at::randn({3, 5}), at::randn({5})};
  testStaticRuntime(aten_matmul, args);
  // > 2-D , > 2-D
  args = {at::randn({3, 1, 4, 5}), at::randn({2, 5, 6})};
  testStaticRuntime(aten_matmul, args);
}

TEST(StaticRuntime, IndividualOps_Div) {
  auto a = at::randn({2, 3});
  auto b = at::randn({2, 3});

  std::vector<IValue> args0{a, b};
  testStaticRuntime(div_tensor, args0);

  std::vector<IValue> args1{a, 3};
  testStaticRuntime(div_scalar, args1);

  std::vector<IValue> args2{a, b, "floor"};
  testStaticRuntime(div_tensor_mode, args2);

  std::vector<IValue> args3{a, 2.3, "trunc"};
  testStaticRuntime(div_scalar_mode, args3);
}

TEST(StaticRuntime, IndividualOps_Sub) {
  auto a = at::randn({2, 3});
  auto b = at::randn({2, 3});

  std::vector<IValue> args0{a, b};
  testStaticRuntime(sub_tensor, args0);

  std::vector<IValue> args1{a, 3};
  testStaticRuntime(sub_scalar, args1);

  std::vector<IValue> args2{a, b, 2.3};
  testStaticRuntime(sub_tensor_alpha, args2);

  std::vector<IValue> args3{a, 2.3, 4};
  testStaticRuntime(sub_scalar_alpha, args3);
}

TEST(StaticRuntime, IndividualOps_Norm) {
  auto a = at::randn({2, 3});
  auto dim = std::vector<int64_t>({1});
  auto dtype = at::ScalarType::Float;

  std::vector<IValue> args2{a, 2};
  testStaticRuntime(norm_2arg, args2);

  std::vector<IValue> args3{a, 2, dtype};
  testStaticRuntime(norm_3arg, args3);

  std::vector<IValue> args4{a, 3, dim, false};
  testStaticRuntime(norm_4arg, args4);

  std::vector<IValue> args5{a, 4, dim, true, dtype};
  testStaticRuntime(norm_5arg, args5);

}

TEST(StaticRuntime, IndividualOps_Reshape) {
  auto a = at::randn({2, 3});
  auto b = std::vector<int64_t>({3, 2});
  std::vector<IValue> args{a, b};

  testStaticRuntime(reshape_script_1, args);
  testStaticRuntime(reshape_script_2, args);
  testStaticRuntime(reshape_script_3, args);
  testStaticRuntime(reshape_script_4, args);
  testStaticRuntime(reshape_script_5, args);
  testStaticRuntime(reshape_inplace_script, args);
  testStaticRuntime(reshape_incontiguous_script, args);
}

TEST(StaticRuntime, IndividualOps_Repeat) {
  auto a = at::randn({2, 3});
  auto b = std::vector<int64_t>({1, 2});
  auto c = std::vector<int64_t>({2, 3});
  std::vector<IValue> args1{a, b};
  std::vector<IValue> args2{a, c};

  testStaticRuntime(repeat, args1);
  testStaticRuntime(repeat, args2);
}

TEST(StaticRuntime, IndividualOps_flatten) {
  auto test_flatten =
      [](std::vector<int64_t> shape, int64_t start_dim, int64_t end_dim) {
        auto a = at::randn(shape);
        std::vector<IValue> args{a, start_dim, end_dim};
        testStaticRuntime(flatten_script_1, args);
        if (shape.size() > 2) {
          testStaticRuntime(flatten_script_2, args);
        }
      };

  test_flatten({2, 3}, 0, 1);
  test_flatten({2, 1, 3}, 1, 2);
  test_flatten({0, 1, 3, 0}, 1, 2);
  test_flatten({2, 3}, 1, 1);
  test_flatten({}, 0, 0);
}

TEST(StaticRuntime, IndividualOps_pow) {
  auto a = at::randn({2, 3});
  auto b = at::randn({2, 3});

  std::vector<IValue> args0{a, 4};
  testStaticRuntime(pow_script_ten_sca, args0);

  std::vector<IValue> args1{at::abs(a), b};
  testStaticRuntime(pow_script_ten_ten, args1);

  std::vector<IValue> args2{5, b};
  testStaticRuntime(pow_script_sca_ten, args2);
}

TEST(StaticRuntime, IndividualOps_to) {
  auto test_to = [](at::ScalarType b, bool c, bool d, c10::MemoryFormat e) {
    auto a = at::randn({2, 3});
    auto other = at::randn({2, 3}, b);
    std::vector<IValue> args0{a, b, c, d, e};
    std::vector<IValue> args1{a, b, c, d};
    std::vector<IValue> args2{a, other, c, d, e};
    testStaticRuntime(to_script_0, args0);
    testStaticRuntime(to_script_1, args1);
    testStaticRuntime(to_script_2, args2);
  };

  test_to(at::ScalarType::Float, true, true, c10::MemoryFormat::Contiguous);
  test_to(at::ScalarType::Half, true, false, c10::MemoryFormat::Preserve);
  test_to(at::ScalarType::Float, false, false, c10::MemoryFormat::Contiguous);
  test_to(at::ScalarType::Half, false, true, c10::MemoryFormat::Preserve);
}

TEST(StaticRuntime, LongModel) {
  torch::jit::Module mod = getLongScriptModel();
  auto a = torch::randn({2, 2});
  auto b = torch::randn({2, 2});
  auto c = torch::randn({2, 2});

  // run jit graph executor
  std::vector<at::IValue> input_ivalues({a, b, c});
  at::Tensor output_1 = mod.forward(input_ivalues).toTensor();

  // run static runtime
  std::vector<at::Tensor> input_tensors({a, b, c});
  torch::jit::StaticModule smod(mod);
  at::Tensor output_2 = smod(input_tensors)[0];
  smod.runtime().check_for_memory_leak();
  EXPECT_TRUE(torch::allclose(output_1, output_2, 1e-6));
}

TEST(StaticRuntime, TrivialModel) {
  torch::jit::Module mod = getTrivialScriptModel();
  auto a = torch::randn({2, 2});
  auto b = torch::randn({2, 2});
  auto c = torch::randn({2, 2});

  // run jit graph executor
  std::vector<at::IValue> input_ivalues({a, b, c});
  at::Tensor output_1 = mod.forward(input_ivalues).toTensor();

  // run static runtime
  std::vector<at::Tensor> input_tensors({a, b, c});
  torch::jit::StaticModule smod(mod);
  at::Tensor output_2 = smod(input_tensors)[0];
  smod.runtime().check_for_memory_leak();
  EXPECT_TRUE(torch::allclose(output_1, output_2, 1e-6));
}

TEST(StaticRuntime, LeakyReLU) {
  torch::jit::Module mod = getLeakyReLUConstScriptModel();
  auto inputs = torch::randn({2, 2});

  // run jit graph executor
  std::vector<at::IValue> input_ivalues({inputs});
  at::Tensor output_1 = mod.forward(input_ivalues).toTensor();

  // run static runtime
  std::vector<at::Tensor> input_tensors({inputs});
  torch::jit::StaticModule smod(mod);
  at::Tensor output_2 = smod(input_tensors)[0];
  smod.runtime().check_for_memory_leak();
  EXPECT_TRUE(torch::allclose(output_1, output_2, 1e-6));
}

TEST(StaticRuntime, DeepWide) {
  const int embedding_size = 32;
  const int num_features = 50;
  torch::jit::Module mod = getDeepAndWideSciptModel();
  torch::jit::StaticModule smod(mod);

  for (int batch_size : {1, 8, 32}) {
    for (int i = 0; i < 2; ++i) {
      auto ad_emb_packed = torch::randn({batch_size, 1, embedding_size});
      auto user_emb = torch::randn({batch_size, 1, embedding_size});
      auto wide = torch::randn({batch_size, num_features});

      // run jit graph executor
      std::vector<at::IValue> inputs({ad_emb_packed, user_emb, wide});
      auto output_1 = getTensor(mod.forward(inputs));

      // run static runtime
      std::vector<at::Tensor> input_tensors({ad_emb_packed, user_emb, wide});
      at::Tensor output_2 = smod(input_tensors)[0];
      smod.runtime().check_for_memory_leak();
      EXPECT_TRUE(torch::allclose(output_1, output_2, 1e-6));
    }
  }
}

TEST(StaticRuntime, KWargsAPI_1) {
  const int embedding_size = 32;
  const int num_features = 50;
  auto module = getDeepAndWideSciptModel();
  torch::jit::StaticModule smod(module);

  for (int batch_size : {1, 8, 32}) {
    for (int i = 0; i < 2; ++i) {
      auto ad_emb_packed = torch::randn({batch_size, 1, embedding_size});
      auto user_emb = torch::randn({batch_size, 1, embedding_size});
      auto wide = torch::randn({batch_size, num_features});
      {
        std::vector<at::IValue> inputs({ad_emb_packed, user_emb, wide});

        // run jit graph executor
        at::Tensor output_1 = getTensor(module.forward(inputs));

        // run static runtime
        c10::IValue output_ivalue = smod(inputs, {});
        smod.runtime().check_for_memory_leak();

        at::Tensor output_2 = getTensor(output_ivalue);
        EXPECT_TRUE(torch::allclose(output_1, output_2, 1e-6));

        // check for output aliasing
        EXPECT_EQ(output_ivalue.use_count(), 1);
        output_ivalue = IValue();

        EXPECT_EQ(output_2.getIntrusivePtr().use_count(), 1);
      }

      // check for input aliasing (deep & wide does not have ops
      // that create aliases of input tensors)
      EXPECT_EQ(ad_emb_packed.getIntrusivePtr().use_count(), 1);
      EXPECT_EQ(user_emb.getIntrusivePtr().use_count(), 1);
      EXPECT_EQ(wide.getIntrusivePtr().use_count(), 1);
    }
  }
}

TEST(StaticRuntime, KWargsAPI_2) {
  const int embedding_size = 32;
  const int num_features = 50;
  auto module = getDeepAndWideSciptModel();
  torch::jit::StaticModule smod(module);

  for (int batch_size : {1, 8, 32}) {
    for (int i = 0; i < 2; ++i) {
      auto ad_emb_packed = torch::randn({batch_size, 1, embedding_size});
      auto user_emb = torch::randn({batch_size, 1, embedding_size});
      auto wide = torch::randn({batch_size, num_features});
      {
        // run jit graph executor
        std::vector<at::IValue> args({ad_emb_packed, user_emb, wide});
        at::Tensor output_1 = getTensor(module.forward(args));

        std::unordered_map<std::string, c10::IValue> kwargs(
            {{"ad_emb_packed", ad_emb_packed},
             {"user_emb", user_emb},
             {"wide", wide}});

        // run static runtime
        c10::IValue output_ivalue = smod({}, kwargs);
        smod.runtime().check_for_memory_leak();

        at::Tensor output_2 = getTensor(output_ivalue);
        EXPECT_TRUE(torch::allclose(output_1, output_2, 1e-6));

        // check for output aliasing
        EXPECT_EQ(output_ivalue.use_count(), 1);
        output_ivalue = IValue();

        EXPECT_EQ(output_2.getIntrusivePtr().use_count(), 1);
      }

      EXPECT_EQ(ad_emb_packed.getIntrusivePtr().use_count(), 1);
      EXPECT_EQ(user_emb.getIntrusivePtr().use_count(), 1);
      EXPECT_EQ(wide.getIntrusivePtr().use_count(), 1);
    }
  }
}

TEST(StaticRuntime, CleanUpMemory) {
  const int embedding_size = 32;
  const int num_features = 50;
  torch::jit::Module mod = getDeepAndWideSciptModel();

  for (auto cleanup_activations : {true, false}) {
    for (auto enable_out_variant : {true, false}) {
      for (auto optimize_memory : {true, false}) {
        for (auto optimize_graph_output_memory : {true, false}) {
          if (optimize_graph_output_memory && !optimize_memory) {
            // when optimize_graph_output_memory is enabled, optimize_memory
            // must be enabled too
            continue;
          }
          if (optimize_memory && !enable_out_variant) {
            // when optimize_memory is enabled, enable_out_variant must be
            // enabled too
            continue;
          }
          VLOG(1) << "cleanup_activations: " << cleanup_activations
                  << ", enable_out_variant: " << enable_out_variant
                  << ", optimize_memory: " << optimize_memory
                  << ", optimize_graph_output_memory: "
                  << optimize_graph_output_memory;
          torch::jit::StaticModuleOptions opts{
              cleanup_activations,
              enable_out_variant,
              optimize_memory,
              optimize_graph_output_memory};
          torch::jit::StaticModule smod(mod, opts);

          for (int batch_size : {1, 8, 32}) {
            for (int i = 0; i < 2; ++i) {
              auto ad_emb_packed =
                  torch::randn({batch_size, 1, embedding_size});
              auto user_emb = torch::randn({batch_size, 1, embedding_size});
              auto wide = torch::randn({batch_size, num_features});

              // run jit graph executor
              std::vector<at::IValue> inputs({ad_emb_packed, user_emb, wide});
              auto output_1 = getTensor(mod.forward(inputs));

              // run static runtime
              std::vector<at::Tensor> input_tensors(
                  {ad_emb_packed, user_emb, wide});
              at::Tensor output_2 = smod(input_tensors)[0];
              smod.runtime().check_for_memory_leak();
              EXPECT_TRUE(torch::allclose(output_1, output_2, 1e-6));
            }
          }
        }
      }
    }
  }
}

TEST(StaticRuntime, FusionPass) {
  const int embedding_size = 32;
  const int num_features = 50;
  for (int batch_size : {1, 8, 32}) {
    for (int i = 0; i < 2; ++i) {
      torch::jit::Module module = getDeepAndWideSciptModel();
      auto ad_emb_packed = torch::randn({batch_size, 1, embedding_size});
      auto user_emb = torch::randn({batch_size, 1, embedding_size});
      auto wide = torch::randn({batch_size, num_features});

      // run jit graph executor
      std::vector<at::IValue> inputs({ad_emb_packed, user_emb, wide});
      auto output_1 = getTensor(module.forward(inputs));

      Method method = module.get_method("forward");
      auto graph = method.graph();
      fuseStaticSubgraphs(graph, 2);
      bool hit = false;
      for (const auto& n : module.get_method("forward").graph()->nodes()) {
        if (n->kind() == torch::jit::prim::StaticSubgraph) {
          hit = true;
        }
      }
      EXPECT_TRUE(hit);
      auto output_2 = getTensor(module.forward(inputs));
      EXPECT_TRUE(torch::allclose(output_1, output_2, 1e-6));
    }
  }
}

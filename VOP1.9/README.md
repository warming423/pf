# pf
system of operator performace visualization 

```
pf
├─ .vscode
│  └─ settings.json
├─ LICENSE
├─ README.md
├─ VOP
│  ├─ Stubs
│  │  ├─ _profiler.pyi
│  │  └─ profiler.pyi
│  ├─ __pycache__
│  │  ├─ aaa.cpython-38.pyc
│  │  └─ profiler.cpython-38.pyc
│  ├─ aaa.py
│  ├─ aaa.pyi
│  ├─ p.pyi
│  ├─ ppp.py
│  ├─ ppp.pyi
│  ├─ profiler.py
│  └─ profiler.pyi
├─ VisualOP
│  ├─ Stubs
│  │  └─ _profiler.pyi
│  └─ profiler.py
├─ main.py
├─ paddle-profiler
│  ├─ __init__.py
│  ├─ profiler.py
│  ├─ profiler_statistic.py
│  ├─ statistic_helper.py
│  ├─ timer.py
│  └─ utils.py
├─ pytorch-profiler
│  ├─ _C
│  │  ├─ _VariableFunctions.pyi.in
│  │  ├─ __init__.pyi.in
│  │  ├─ _aoti.pyi
│  │  ├─ _autograd.pyi
│  │  ├─ _cpu.pyi
│  │  ├─ _cudnn.pyi
│  │  ├─ _distributed_autograd.pyi
│  │  ├─ _distributed_c10d.pyi
│  │  ├─ _distributed_rpc.pyi
│  │  ├─ _distributed_rpc_testing.pyi
│  │  ├─ _dynamo
│  │  │  ├─ __init__.pyi
│  │  │  ├─ compiled_autograd.pyi
│  │  │  ├─ eval_frame.pyi
│  │  │  └─ guards.pyi
│  │  ├─ _functions.pyi
│  │  ├─ _functorch.pyi
│  │  ├─ _itt.pyi
│  │  ├─ _lazy.pyi
│  │  ├─ _lazy_ts_backend.pyi
│  │  ├─ _monitor.pyi
│  │  ├─ _nn.pyi.in
│  │  ├─ _nvtx.pyi
│  │  ├─ _onnx.pyi
│  │  ├─ _profiler.pyi
│  │  ├─ _verbose.pyi
│  │  ├─ build.bzl
│  │  └─ return_types.pyi.in
│  ├─ __init__.py
│  ├─ _memory_profiler.py
│  ├─ _pattern_matcher.py
│  ├─ _utils.py
│  ├─ autograd
│  │  ├─ __init__.py
│  │  ├─ _functions
│  │  │  ├─ __init__.py
│  │  │  ├─ replace.vim
│  │  │  ├─ tensor.py
│  │  │  └─ utils.py
│  │  ├─ anomaly_mode.py
│  │  ├─ forward_ad.py
│  │  ├─ function.py
│  │  ├─ functional.py
│  │  ├─ grad_mode.py
│  │  ├─ gradcheck.py
│  │  ├─ graph.py
│  │  ├─ profiler.py
│  │  ├─ profiler_legacy.py
│  │  ├─ profiler_util.py
│  │  └─ variable.py
│  ├─ init.cpp
│  ├─ itt.py
│  ├─ kineto-main
│  │  ├─ CODE_OF_CONDUCT.md
│  │  ├─ CONTRIBUTING.md
│  │  ├─ LICENSE
│  │  ├─ README.md
│  │  ├─ libkineto
│  │  │  ├─ CMakeLists.txt
│  │  │  ├─ README.md
│  │  │  ├─ include
│  │  │  │  ├─ AbstractConfig.h
│  │  │  │  ├─ ActivityProfilerInterface.h
│  │  │  │  ├─ ActivityTraceInterface.h
│  │  │  │  ├─ ActivityType.h
│  │  │  │  ├─ ClientInterface.h
│  │  │  │  ├─ Config.h
│  │  │  │  ├─ GenericTraceActivity.h
│  │  │  │  ├─ IActivityProfiler.h
│  │  │  │  ├─ ILoggerObserver.h
│  │  │  │  ├─ ITraceActivity.h
│  │  │  │  ├─ LoggingAPI.h
│  │  │  │  ├─ ThreadUtil.h
│  │  │  │  ├─ TraceSpan.h
│  │  │  │  ├─ libkineto.h
│  │  │  │  ├─ output_base.h
│  │  │  │  └─ time_since_epoch.h
│  │  │  ├─ ipcfabric
│  │  │  │  ├─ CMakeLists.txt
│  │  │  │  ├─ Endpoint.h
│  │  │  │  ├─ FabricManager.h
│  │  │  │  ├─ README.md
│  │  │  │  └─ Utils.h
│  │  │  ├─ libkineto_defs.bzl
│  │  │  ├─ sample_programs
│  │  │  │  ├─ README.md
│  │  │  │  ├─ build-cu.sh
│  │  │  │  ├─ build.sh
│  │  │  │  ├─ kineto_cupti_profiler.cpp
│  │  │  │  ├─ kineto_playground.cpp
│  │  │  │  ├─ kineto_playground.cu
│  │  │  │  └─ kineto_playground.cuh
│  │  │  ├─ src
│  │  │  │  ├─ AbstractConfig.cpp
│  │  │  │  ├─ ActivityBuffers.h
│  │  │  │  ├─ ActivityLoggerFactory.h
│  │  │  │  ├─ ActivityProfilerController.cpp
│  │  │  │  ├─ ActivityProfilerController.h
│  │  │  │  ├─ ActivityProfilerProxy.cpp
│  │  │  │  ├─ ActivityProfilerProxy.h
│  │  │  │  ├─ ActivityTrace.h
│  │  │  │  ├─ ActivityType.cpp
│  │  │  │  ├─ Config.cpp
│  │  │  │  ├─ ConfigLoader.cpp
│  │  │  │  ├─ ConfigLoader.h
│  │  │  │  ├─ CudaDeviceProperties.cpp
│  │  │  │  ├─ CudaDeviceProperties.h
│  │  │  │  ├─ CudaUtil.cpp
│  │  │  │  ├─ CudaUtil.h
│  │  │  │  ├─ CuptiActivity.cpp
│  │  │  │  ├─ CuptiActivity.h
│  │  │  │  ├─ CuptiActivityApi.cpp
│  │  │  │  ├─ CuptiActivityApi.h
│  │  │  │  ├─ CuptiActivityBuffer.h
│  │  │  │  ├─ CuptiActivityProfiler.cpp
│  │  │  │  ├─ CuptiActivityProfiler.h
│  │  │  │  ├─ CuptiCallbackApi.cpp
│  │  │  │  ├─ CuptiCallbackApi.h
│  │  │  │  ├─ CuptiCallbackApiMock.h
│  │  │  │  ├─ CuptiEventApi.cpp
│  │  │  │  ├─ CuptiEventApi.h
│  │  │  │  ├─ CuptiMetricApi.cpp
│  │  │  │  ├─ CuptiMetricApi.h
│  │  │  │  ├─ CuptiNvPerfMetric.cpp
│  │  │  │  ├─ CuptiNvPerfMetric.h
│  │  │  │  ├─ CuptiRangeProfiler.cpp
│  │  │  │  ├─ CuptiRangeProfiler.h
│  │  │  │  ├─ CuptiRangeProfilerApi.cpp
│  │  │  │  ├─ CuptiRangeProfilerApi.h
│  │  │  │  ├─ CuptiRangeProfilerConfig.cpp
│  │  │  │  ├─ CuptiRangeProfilerConfig.h
│  │  │  │  ├─ DaemonConfigLoader.cpp
│  │  │  │  ├─ DaemonConfigLoader.h
│  │  │  │  ├─ Demangle.cpp
│  │  │  │  ├─ Demangle.h
│  │  │  │  ├─ EventProfiler.cpp
│  │  │  │  ├─ EventProfiler.h
│  │  │  │  ├─ EventProfilerController.cpp
│  │  │  │  ├─ EventProfilerController.h
│  │  │  │  ├─ GenericTraceActivity.cpp
│  │  │  │  ├─ ILoggerObserver.cpp
│  │  │  │  ├─ InvariantViolations.h
│  │  │  │  ├─ IpcFabricConfigClient.cpp
│  │  │  │  ├─ IpcFabricConfigClient.h
│  │  │  │  ├─ Logger.cpp
│  │  │  │  ├─ Logger.h
│  │  │  │  ├─ LoggerCollector.h
│  │  │  │  ├─ LoggingAPI.cpp
│  │  │  │  ├─ RoctracerActivity.h
│  │  │  │  ├─ RoctracerActivityApi.cpp
│  │  │  │  ├─ RoctracerActivityApi.h
│  │  │  │  ├─ RoctracerActivity_inl.h
│  │  │  │  ├─ RoctracerLogger.cpp
│  │  │  │  ├─ RoctracerLogger.h
│  │  │  │  ├─ SampleListener.h
│  │  │  │  ├─ ScopeExit.h
│  │  │  │  ├─ ThreadUtil.cpp
│  │  │  │  ├─ WeakSymbols.cpp
│  │  │  │  ├─ cuda_call.h
│  │  │  │  ├─ cupti_call.h
│  │  │  │  ├─ cupti_strings.cpp
│  │  │  │  ├─ cupti_strings.h
│  │  │  │  ├─ init.cpp
│  │  │  │  ├─ libkineto_api.cpp
│  │  │  │  ├─ output_csv.cpp
│  │  │  │  ├─ output_csv.h
│  │  │  │  ├─ output_json.cpp
│  │  │  │  ├─ output_json.h
│  │  │  │  └─ output_membuf.h
│  │  │  ├─ stress_test
│  │  │  │  ├─ kineto_stress_test.cpp
│  │  │  │  ├─ random_ops_stress_test.cu
│  │  │  │  ├─ random_ops_stress_test.cuh
│  │  │  │  ├─ run_multiproc_stress_test.sh
│  │  │  │  ├─ stress_test_dense.json
│  │  │  │  ├─ stress_test_uvm_nccl.json
│  │  │  │  ├─ tensor_cache.cu
│  │  │  │  ├─ tensor_cache.cuh
│  │  │  │  └─ utils.h
│  │  │  ├─ test
│  │  │  │  ├─ CMakeLists.txt
│  │  │  │  ├─ ConfigTest.cpp
│  │  │  │  ├─ CuptiActivityProfilerTest.cpp
│  │  │  │  ├─ CuptiCallbackApiTest.cpp
│  │  │  │  ├─ CuptiProfilerApiTest.cu
│  │  │  │  ├─ CuptiRangeProfilerApiTest.cpp
│  │  │  │  ├─ CuptiRangeProfilerConfigTest.cpp
│  │  │  │  ├─ CuptiRangeProfilerTest.cpp
│  │  │  │  ├─ CuptiRangeProfilerTestUtil.h
│  │  │  │  ├─ CuptiStringsTest.cpp
│  │  │  │  ├─ EventProfilerTest.cpp
│  │  │  │  ├─ LoggerObserverTest.cpp
│  │  │  │  ├─ MockActivitySubProfiler.cpp
│  │  │  │  ├─ MockActivitySubProfiler.h
│  │  │  │  └─ PidInfoTest.cpp
│  │  │  └─ third_party
│  │  │     ├─ dynolog
│  │  │     ├─ fmt
│  │  │     └─ googletest
│  │  └─ tb_plugin
│  │     ├─ .flake8
│  │     ├─ .pre-commit-config.yaml
│  │     ├─ LICENSE
│  │     ├─ README.md
│  │     ├─ ci_scripts
│  │     │  └─ install_env.sh
│  │     ├─ docs
│  │     │  ├─ gpu_utilization.md
│  │     │  └─ images
│  │     │     ├─ control_panel.PNG
│  │     │     ├─ diff_view.png
│  │     │     ├─ distributed_view.PNG
│  │     │     ├─ kernel_view.PNG
│  │     │     ├─ kernel_view_group_by_properties_and_op.PNG
│  │     │     ├─ lightning_view.png
│  │     │     ├─ memory_view.PNG
│  │     │     ├─ module_view.png
│  │     │     ├─ operator_view.PNG
│  │     │     ├─ operator_view_group_by_inputshape.PNG
│  │     │     ├─ overall_view.PNG
│  │     │     ├─ time_breakdown_priority.PNG
│  │     │     ├─ trace_view.PNG
│  │     │     ├─ trace_view_fwd_bwd_correlation.PNG
│  │     │     ├─ trace_view_gpu_utilization.PNG
│  │     │     ├─ trace_view_launch.PNG
│  │     │     ├─ trace_view_one_step.PNG
│  │     │     └─ vscode_stack.PNG
│  │     ├─ examples
│  │     │  ├─ resnet50_autograd_api.py
│  │     │  ├─ resnet50_ddp_profiler.py
│  │     │  └─ resnet50_profiler_api.py
│  │     ├─ fe
│  │     │  ├─ README.md
│  │     │  ├─ index.html
│  │     │  ├─ package.json
│  │     │  ├─ prettier.json
│  │     │  ├─ scripts
│  │     │  │  ├─ add_header.py
│  │     │  │  ├─ build.sh
│  │     │  │  └─ setup.sh
│  │     │  ├─ src
│  │     │  │  ├─ api
│  │     │  │  │  ├─ README.md
│  │     │  │  │  ├─ generated
│  │     │  │  │  │  ├─ api.ts
│  │     │  │  │  │  ├─ configuration.ts
│  │     │  │  │  │  ├─ custom.d.ts
│  │     │  │  │  │  └─ index.ts
│  │     │  │  │  ├─ index.ts
│  │     │  │  │  ├─ mock.ts
│  │     │  │  │  └─ openapi.yaml
│  │     │  │  ├─ app.tsx
│  │     │  │  ├─ components
│  │     │  │  │  ├─ DataLoading.tsx
│  │     │  │  │  ├─ DiffOverview.tsx
│  │     │  │  │  ├─ DistributedView.tsx
│  │     │  │  │  ├─ FullCircularProgress.tsx
│  │     │  │  │  ├─ GpuInfoTable.tsx
│  │     │  │  │  ├─ Kernel.tsx
│  │     │  │  │  ├─ MemoryView.tsx
│  │     │  │  │  ├─ ModuleView.tsx
│  │     │  │  │  ├─ Operator.tsx
│  │     │  │  │  ├─ Overview.tsx
│  │     │  │  │  ├─ TextListItem.tsx
│  │     │  │  │  ├─ TooltipDescriptions.ts
│  │     │  │  │  ├─ TraceView.tsx
│  │     │  │  │  ├─ charts
│  │     │  │  │  │  ├─ AntTableChart.tsx
│  │     │  │  │  │  ├─ AreaChart.tsx
│  │     │  │  │  │  ├─ ColumnChart.tsx
│  │     │  │  │  │  ├─ LineChart.tsx
│  │     │  │  │  │  ├─ PieChart.tsx
│  │     │  │  │  │  ├─ SteppedAreaChart.tsx
│  │     │  │  │  │  └─ TableChart.tsx
│  │     │  │  │  ├─ helpers.tsx
│  │     │  │  │  ├─ tables
│  │     │  │  │  │  ├─ CallFrameList.tsx
│  │     │  │  │  │  ├─ CallStackTable.tsx
│  │     │  │  │  │  ├─ ExpandIcon.tsx
│  │     │  │  │  │  ├─ MemoryStatsTable.tsx
│  │     │  │  │  │  ├─ NavToCodeButton.tsx
│  │     │  │  │  │  ├─ OperationTable.tsx
│  │     │  │  │  │  ├─ common.tsx
│  │     │  │  │  │  └─ transform.ts
│  │     │  │  │  └─ transform.ts
│  │     │  │  ├─ constants
│  │     │  │  │  └─ groupBy.ts
│  │     │  │  ├─ gstatic.d.ts
│  │     │  │  ├─ index.tsx
│  │     │  │  ├─ setup.tsx
│  │     │  │  ├─ styles.css
│  │     │  │  └─ utils
│  │     │  │     ├─ binarysearch.ts
│  │     │  │     ├─ debounce.ts
│  │     │  │     ├─ def.ts
│  │     │  │     ├─ index.ts
│  │     │  │     ├─ resize.ts
│  │     │  │     ├─ search.ts
│  │     │  │     ├─ top.ts
│  │     │  │     ├─ type.ts
│  │     │  │     └─ vscode.ts
│  │     │  ├─ tsconfig.json
│  │     │  ├─ update-static.js
│  │     │  ├─ webpack.config.js
│  │     │  └─ yarn.lock
│  │     ├─ packaging
│  │     │  └─ torch_tb_profiler
│  │     │     └─ meta.yaml
│  │     ├─ samples
│  │     │  ├─ resnet50_num_workers_0
│  │     │  │  ├─ worker0.1623143089861.pt.trace.json.gz
│  │     │  │  └─ worker0.1623143566756.pt.trace.json.gz
│  │     │  └─ resnet50_num_workers_4
│  │     │     ├─ worker0.1623212756351.pt.trace.json.gz
│  │     │     └─ worker0.1623213129365.pt.trace.json.gz
│  │     ├─ setup.py
│  │     ├─ test
│  │     │  ├─ gpu_metrics_expected.json
│  │     │  ├─ gpu_metrics_input.json
│  │     │  ├─ result_check_file.txt
│  │     │  ├─ test_compare_with_autograd.py
│  │     │  ├─ test_diffrun.py
│  │     │  ├─ test_profiler.py
│  │     │  ├─ test_ranges.py
│  │     │  └─ test_tensorboard_end2end.py
│  │     └─ torch_tb_profiler
│  │        ├─ __init__.py
│  │        ├─ consts.py
│  │        ├─ io
│  │        │  ├─ __init__.py
│  │        │  ├─ azureblob.py
│  │        │  ├─ base.py
│  │        │  ├─ cache.py
│  │        │  ├─ file.py
│  │        │  ├─ gs.py
│  │        │  ├─ hdfs.py
│  │        │  └─ utils.py
│  │        ├─ multiprocessing.py
│  │        ├─ plugin.py
│  │        ├─ profiler
│  │        │  ├─ __init__.py
│  │        │  ├─ communication.py
│  │        │  ├─ data.py
│  │        │  ├─ diffrun
│  │        │  │  ├─ __init__.py
│  │        │  │  ├─ contract.py
│  │        │  │  ├─ operator.py
│  │        │  │  └─ tree.py
│  │        │  ├─ event_parser.py
│  │        │  ├─ gpu_metrics_parser.py
│  │        │  ├─ kernel_parser.py
│  │        │  ├─ loader.py
│  │        │  ├─ memory_parser.py
│  │        │  ├─ module_op.py
│  │        │  ├─ node.py
│  │        │  ├─ op_agg.py
│  │        │  ├─ op_tree.py
│  │        │  ├─ overall_parser.py
│  │        │  ├─ range_utils.py
│  │        │  ├─ run_generator.py
│  │        │  ├─ tensor_core.py
│  │        │  ├─ tensor_cores_parser.py
│  │        │  └─ trace.py
│  │        ├─ run.py
│  │        ├─ static
│  │        │  ├─ index.html
│  │        │  ├─ index.js
│  │        │  ├─ trace_embedding.html
│  │        │  └─ trace_viewer_full.html
│  │        └─ utils.py
│  ├─ profiler.py
│  ├─ python_tracer.py
│  └─ torch.autograd.profiler.py
├─ tensorflow-profiler
│  └─ profiler-master
│     ├─ BUILD
│     ├─ LICENSE
│     ├─ README.md
│     ├─ WORKSPACE
│     ├─ defs
│     │  ├─ BUILD.bazel
│     │  └─ defs.bzl
│     ├─ demo
│     │  ├─ events.out.tfevents.1583461681.localhost.profile-empty
│     │  └─ plugins
│     │     └─ profile
│     │        ├─ Cloud-TPU
│     │        │  ├─ 10.240.1.50.cluster_while_body_41431_321794125038409836__.9413.formatting.memory_viewer.json
│     │        │  ├─ 10.240.1.50.input_pipeline.json
│     │        │  ├─ 10.240.1.50.op_profile.json
│     │        │  ├─ 10.240.1.50.overview_page.json
│     │        │  ├─ 10.240.1.50.pod_viewer.json
│     │        │  ├─ 10.240.1.50.tensorflow_stats.pb
│     │        │  └─ 10.240.1.50.trace
│     │        ├─ GPU-compute-bound
│     │        │  ├─ localhost.input_pipeline.pb
│     │        │  └─ localhost.overview_page.pb
│     │        ├─ GPU-input-bound
│     │        │  ├─ localhost.input_pipeline.pb
│     │        │  ├─ localhost.kernel_stats.pb
│     │        │  ├─ localhost.overview_page.pb
│     │        │  ├─ localhost.tensorflow_stats.pb
│     │        │  └─ localhost.trace.json.gz
│     │        └─ GPU-multi-worker
│     │           ├─ 34.83.195.179_6003.xplane.pb
│     │           └─ 34.83.195.180_6003.xplane.pb
│     ├─ docs
│     │  ├─ images
│     │  │  ├─ capture_profile.png
│     │  │  └─ overview_page.png
│     │  └─ profile_multi_gpu.md
│     ├─ frontend
│     │  ├─ BUILD
│     │  ├─ app
│     │  │  ├─ BUILD
│     │  │  ├─ app.ng.html
│     │  │  ├─ app.scss
│     │  │  ├─ app.ts
│     │  │  ├─ app_module.ts
│     │  │  ├─ common
│     │  │  │  ├─ angular
│     │  │  │  │  └─ BUILD
│     │  │  │  ├─ constants
│     │  │  │  │  ├─ BUILD
│     │  │  │  │  ├─ constants.ts
│     │  │  │  │  ├─ enums.ts
│     │  │  │  │  └─ testing.ts
│     │  │  │  ├─ interfaces
│     │  │  │  │  ├─ BUILD
│     │  │  │  │  ├─ buffer_allocation_info.ts
│     │  │  │  │  ├─ capture_profile.ts
│     │  │  │  │  ├─ chart.ts
│     │  │  │  │  ├─ data_table.ts
│     │  │  │  │  ├─ diagnostics.ts
│     │  │  │  │  ├─ graph_viewer.ts
│     │  │  │  │  ├─ heap_object.ts
│     │  │  │  │  ├─ hlo.jsonpb_decls.d.ts.gz
│     │  │  │  │  ├─ memory_profile.jsonpb_decls.d.ts.gz
│     │  │  │  │  ├─ memory_viewer_preprocess.jsonpb_decls.d.ts.gz
│     │  │  │  │  ├─ navigation_event.ts
│     │  │  │  │  ├─ op_metrics.jsonpb_decls.d.ts
│     │  │  │  │  ├─ op_profile.jsonpb_decls.d.ts
│     │  │  │  │  ├─ summary_info.ts
│     │  │  │  │  ├─ tool.ts
│     │  │  │  │  ├─ window.ts
│     │  │  │  │  └─ xla_data.jsonpb_decls.d.ts.gz
│     │  │  │  └─ utils
│     │  │  │     ├─ BUILD
│     │  │  │     ├─ testing.ts
│     │  │  │     └─ utils.ts
│     │  │  ├─ components
│     │  │  │  ├─ capture_profile
│     │  │  │  │  ├─ BUILD
│     │  │  │  │  ├─ capture_profile.ng.html
│     │  │  │  │  ├─ capture_profile.scss
│     │  │  │  │  ├─ capture_profile.ts
│     │  │  │  │  ├─ capture_profile_dialog
│     │  │  │  │  │  ├─ BUILD
│     │  │  │  │  │  ├─ capture_profile_dialog.ng.html
│     │  │  │  │  │  ├─ capture_profile_dialog.scss
│     │  │  │  │  │  ├─ capture_profile_dialog.ts
│     │  │  │  │  │  └─ capture_profile_dialog_module.ts
│     │  │  │  │  └─ capture_profile_module.ts
│     │  │  │  ├─ chart
│     │  │  │  │  ├─ BUILD
│     │  │  │  │  ├─ category_diff_table_data_processor.ts
│     │  │  │  │  ├─ category_table_data_processor.ts
│     │  │  │  │  ├─ chart.ts
│     │  │  │  │  ├─ chart_options.ts
│     │  │  │  │  ├─ dashboard
│     │  │  │  │  │  ├─ BUILD
│     │  │  │  │  │  └─ dashboard.ts
│     │  │  │  │  ├─ default_data_provider.ts
│     │  │  │  │  ├─ filter_data_processor.ts
│     │  │  │  │  ├─ org_chart
│     │  │  │  │  │  ├─ BUILD
│     │  │  │  │  │  ├─ org_chart.ng.html
│     │  │  │  │  │  ├─ org_chart.scss
│     │  │  │  │  │  ├─ org_chart.ts
│     │  │  │  │  │  └─ org_chart_module.ts
│     │  │  │  │  ├─ table
│     │  │  │  │  │  ├─ BUILD
│     │  │  │  │  │  ├─ table.ng.html
│     │  │  │  │  │  ├─ table.scss
│     │  │  │  │  │  ├─ table.ts
│     │  │  │  │  │  └─ table_module.ts
│     │  │  │  │  ├─ table_utils.ts
│     │  │  │  │  └─ xy_table_data_processor.ts
│     │  │  │  ├─ controls
│     │  │  │  │  ├─ category_filter
│     │  │  │  │  │  ├─ BUILD
│     │  │  │  │  │  ├─ category_filter.ng.html
│     │  │  │  │  │  ├─ category_filter.scss
│     │  │  │  │  │  ├─ category_filter.ts
│     │  │  │  │  │  └─ category_filter_module.ts
│     │  │  │  │  ├─ export_as_csv
│     │  │  │  │  │  ├─ BUILD
│     │  │  │  │  │  ├─ export_as_csv.ng.html
│     │  │  │  │  │  ├─ export_as_csv.scss
│     │  │  │  │  │  ├─ export_as_csv.ts
│     │  │  │  │  │  └─ export_as_csv_module.ts
│     │  │  │  │  └─ string_filter
│     │  │  │  │     ├─ BUILD
│     │  │  │  │     ├─ string_filter.ng.html
│     │  │  │  │     ├─ string_filter.scss
│     │  │  │  │     ├─ string_filter.ts
│     │  │  │  │     └─ string_filter_module.ts
│     │  │  │  ├─ dcn_collective_stats
│     │  │  │  │  ├─ BUILD
│     │  │  │  │  ├─ dcn_collective_stats.ng.html
│     │  │  │  │  ├─ dcn_collective_stats.scss
│     │  │  │  │  ├─ dcn_collective_stats.ts
│     │  │  │  │  └─ dcn_collective_stats_module.ts
│     │  │  │  ├─ diagnostics_view
│     │  │  │  │  ├─ BUILD
│     │  │  │  │  ├─ diagnostics_view.ng.html
│     │  │  │  │  ├─ diagnostics_view.scss
│     │  │  │  │  ├─ diagnostics_view.ts
│     │  │  │  │  └─ diagnostics_view_module.ts
│     │  │  │  ├─ empty_page
│     │  │  │  │  ├─ BUILD
│     │  │  │  │  ├─ empty_page.ng.html
│     │  │  │  │  ├─ empty_page.scss
│     │  │  │  │  ├─ empty_page.ts
│     │  │  │  │  └─ empty_page_module.ts
│     │  │  │  ├─ graph_viewer
│     │  │  │  │  ├─ BUILD
│     │  │  │  │  ├─ graph_config
│     │  │  │  │  │  ├─ BUILD
│     │  │  │  │  │  ├─ graph_config.ng.html
│     │  │  │  │  │  ├─ graph_config.scss
│     │  │  │  │  │  ├─ graph_config.ts
│     │  │  │  │  │  └─ graph_config_module.ts
│     │  │  │  │  ├─ graph_viewer.ng.html
│     │  │  │  │  ├─ graph_viewer.scss
│     │  │  │  │  ├─ graph_viewer.ts
│     │  │  │  │  └─ graph_viewer_module.ts
│     │  │  │  ├─ input_pipeline
│     │  │  │  │  ├─ BUILD
│     │  │  │  │  ├─ analysis_summary
│     │  │  │  │  │  ├─ BUILD
│     │  │  │  │  │  ├─ analysis_summary.ng.html
│     │  │  │  │  │  ├─ analysis_summary.scss
│     │  │  │  │  │  ├─ analysis_summary.ts
│     │  │  │  │  │  └─ analysis_summary_module.ts
│     │  │  │  │  ├─ device_side_analysis_detail
│     │  │  │  │  │  ├─ BUILD
│     │  │  │  │  │  ├─ device_side_analysis_detail.ng.html
│     │  │  │  │  │  ├─ device_side_analysis_detail.scss
│     │  │  │  │  │  ├─ device_side_analysis_detail.ts
│     │  │  │  │  │  ├─ device_side_analysis_detail_data_provider.ts
│     │  │  │  │  │  └─ device_side_analysis_detail_module.ts
│     │  │  │  │  ├─ host_side_analysis_detail
│     │  │  │  │  │  ├─ BUILD
│     │  │  │  │  │  ├─ host_side_analysis_detail.ng.html
│     │  │  │  │  │  ├─ host_side_analysis_detail.scss
│     │  │  │  │  │  ├─ host_side_analysis_detail.ts
│     │  │  │  │  │  ├─ host_side_analysis_detail_module.ts
│     │  │  │  │  │  └─ host_side_analysis_detail_table_data_provider.ts
│     │  │  │  │  ├─ input_pipeline.ng.html
│     │  │  │  │  ├─ input_pipeline.scss
│     │  │  │  │  ├─ input_pipeline.ts
│     │  │  │  │  ├─ input_pipeline_common.ts
│     │  │  │  │  └─ input_pipeline_module.ts
│     │  │  │  ├─ kernel_stats
│     │  │  │  │  ├─ BUILD
│     │  │  │  │  ├─ kernel_stats.ng.html
│     │  │  │  │  ├─ kernel_stats.scss
│     │  │  │  │  ├─ kernel_stats.ts
│     │  │  │  │  ├─ kernel_stats_adapter.ts
│     │  │  │  │  ├─ kernel_stats_module.ts
│     │  │  │  │  └─ kernel_stats_table
│     │  │  │  │     ├─ BUILD
│     │  │  │  │     ├─ kernel_stats_table.ng.html
│     │  │  │  │     ├─ kernel_stats_table.scss
│     │  │  │  │     ├─ kernel_stats_table.ts
│     │  │  │  │     └─ kernel_stats_table_module.ts
│     │  │  │  ├─ main_page
│     │  │  │  │  ├─ BUILD
│     │  │  │  │  ├─ main_page.ng.html
│     │  │  │  │  ├─ main_page.scss
│     │  │  │  │  ├─ main_page.ts
│     │  │  │  │  └─ main_page_module.ts
│     │  │  │  ├─ memory_profile
│     │  │  │  │  ├─ BUILD
│     │  │  │  │  ├─ memory_breakdown_table
│     │  │  │  │  │  ├─ BUILD
│     │  │  │  │  │  ├─ memory_breakdown_table.ng.html
│     │  │  │  │  │  ├─ memory_breakdown_table.scss
│     │  │  │  │  │  ├─ memory_breakdown_table.ts
│     │  │  │  │  │  └─ memory_breakdown_table_module.ts
│     │  │  │  │  ├─ memory_profile.ng.html
│     │  │  │  │  ├─ memory_profile.scss
│     │  │  │  │  ├─ memory_profile.ts
│     │  │  │  │  ├─ memory_profile_base.ts
│     │  │  │  │  ├─ memory_profile_common.scss
│     │  │  │  │  ├─ memory_profile_module.ts
│     │  │  │  │  ├─ memory_profile_summary
│     │  │  │  │  │  ├─ BUILD
│     │  │  │  │  │  ├─ memory_profile_summary.ng.html
│     │  │  │  │  │  ├─ memory_profile_summary.scss
│     │  │  │  │  │  ├─ memory_profile_summary.ts
│     │  │  │  │  │  └─ memory_profile_summary_module.ts
│     │  │  │  │  └─ memory_timeline_graph
│     │  │  │  │     ├─ BUILD
│     │  │  │  │     ├─ memory_timeline_graph.ng.html
│     │  │  │  │     ├─ memory_timeline_graph.scss
│     │  │  │  │     ├─ memory_timeline_graph.ts
│     │  │  │  │     └─ memory_timeline_graph_module.ts
│     │  │  │  ├─ memory_viewer
│     │  │  │  │  ├─ BUILD
│     │  │  │  │  ├─ buffer_details
│     │  │  │  │  │  ├─ BUILD
│     │  │  │  │  │  ├─ buffer_details.ng.html
│     │  │  │  │  │  ├─ buffer_details.scss
│     │  │  │  │  │  ├─ buffer_details.ts
│     │  │  │  │  │  └─ buffer_details_module.ts
│     │  │  │  │  ├─ max_heap_chart
│     │  │  │  │  │  ├─ BUILD
│     │  │  │  │  │  ├─ max_heap_chart.ng.html
│     │  │  │  │  │  ├─ max_heap_chart.scss
│     │  │  │  │  │  ├─ max_heap_chart.ts
│     │  │  │  │  │  └─ max_heap_chart_module.ts
│     │  │  │  │  ├─ memory_usage
│     │  │  │  │  │  ├─ BUILD
│     │  │  │  │  │  └─ memory_usage.ts
│     │  │  │  │  ├─ memory_viewer.ng.html
│     │  │  │  │  ├─ memory_viewer.scss
│     │  │  │  │  ├─ memory_viewer.ts
│     │  │  │  │  ├─ memory_viewer_main
│     │  │  │  │  │  ├─ BUILD
│     │  │  │  │  │  ├─ memory_viewer_main.ng.html
│     │  │  │  │  │  ├─ memory_viewer_main.scss
│     │  │  │  │  │  ├─ memory_viewer_main.ts
│     │  │  │  │  │  └─ memory_viewer_main_module.ts
│     │  │  │  │  ├─ memory_viewer_module.ts
│     │  │  │  │  └─ program_order_chart
│     │  │  │  │     ├─ BUILD
│     │  │  │  │     ├─ program_order_chart.ng.html
│     │  │  │  │     ├─ program_order_chart.scss
│     │  │  │  │     ├─ program_order_chart.ts
│     │  │  │  │     └─ program_order_chart_module.ts
│     │  │  │  ├─ op_profile
│     │  │  │  │  ├─ BUILD
│     │  │  │  │  ├─ op_details
│     │  │  │  │  │  ├─ BUILD
│     │  │  │  │  │  ├─ op_details.ng.html
│     │  │  │  │  │  ├─ op_details.scss
│     │  │  │  │  │  ├─ op_details.ts
│     │  │  │  │  │  └─ op_details_module.ts
│     │  │  │  │  ├─ op_profile.ng.html
│     │  │  │  │  ├─ op_profile.scss
│     │  │  │  │  ├─ op_profile.ts
│     │  │  │  │  ├─ op_profile_base.ts
│     │  │  │  │  ├─ op_profile_common.scss
│     │  │  │  │  ├─ op_profile_data.ts
│     │  │  │  │  ├─ op_profile_module.ts
│     │  │  │  │  ├─ op_table
│     │  │  │  │  │  ├─ BUILD
│     │  │  │  │  │  ├─ op_table.ng.html
│     │  │  │  │  │  ├─ op_table.scss
│     │  │  │  │  │  ├─ op_table.ts
│     │  │  │  │  │  └─ op_table_module.ts
│     │  │  │  │  └─ op_table_entry
│     │  │  │  │     ├─ BUILD
│     │  │  │  │     ├─ op_table_entry.ng.html
│     │  │  │  │     ├─ op_table_entry.scss
│     │  │  │  │     ├─ op_table_entry.ts
│     │  │  │  │     └─ op_table_entry_module.ts
│     │  │  │  ├─ overview
│     │  │  │  │  ├─ BUILD
│     │  │  │  │  ├─ normalized_accelerator_performance_view
│     │  │  │  │  │  ├─ BUILD
│     │  │  │  │  │  ├─ normalized_accelerator_performance_view.ng.html
│     │  │  │  │  │  ├─ normalized_accelerator_performance_view.scss
│     │  │  │  │  │  ├─ normalized_accelerator_performance_view.ts
│     │  │  │  │  │  └─ normalized_accelerator_performance_view_module.ts
│     │  │  │  │  ├─ overview.ng.html
│     │  │  │  │  ├─ overview.scss
│     │  │  │  │  ├─ overview.ts
│     │  │  │  │  ├─ overview_common.ts
│     │  │  │  │  ├─ overview_module.ts
│     │  │  │  │  ├─ performance_summary
│     │  │  │  │  │  ├─ BUILD
│     │  │  │  │  │  ├─ performance_summary.ng.html
│     │  │  │  │  │  ├─ performance_summary.scss
│     │  │  │  │  │  ├─ performance_summary.ts
│     │  │  │  │  │  └─ performance_summary_module.ts
│     │  │  │  │  ├─ recommendation_result_view
│     │  │  │  │  │  ├─ BUILD
│     │  │  │  │  │  ├─ recommendation_result_view.ng.html
│     │  │  │  │  │  ├─ recommendation_result_view.scss
│     │  │  │  │  │  ├─ recommendation_result_view.ts
│     │  │  │  │  │  ├─ recommendation_result_view_common.ts
│     │  │  │  │  │  └─ recommendation_result_view_module.ts
│     │  │  │  │  ├─ run_environment_view
│     │  │  │  │  │  ├─ BUILD
│     │  │  │  │  │  ├─ run_environment_view.ng.html
│     │  │  │  │  │  ├─ run_environment_view.scss
│     │  │  │  │  │  ├─ run_environment_view.ts
│     │  │  │  │  │  └─ run_environment_view_module.ts
│     │  │  │  │  ├─ step_time_graph
│     │  │  │  │  │  ├─ BUILD
│     │  │  │  │  │  ├─ step_time_graph.ng.html
│     │  │  │  │  │  ├─ step_time_graph.scss
│     │  │  │  │  │  ├─ step_time_graph.ts
│     │  │  │  │  │  └─ step_time_graph_module.ts
│     │  │  │  │  └─ top_ops_table
│     │  │  │  │     ├─ BUILD
│     │  │  │  │     ├─ top_ops_table.ng.html
│     │  │  │  │     ├─ top_ops_table.scss
│     │  │  │  │     ├─ top_ops_table.ts
│     │  │  │  │     └─ top_ops_table_module.ts
│     │  │  │  ├─ pod_viewer
│     │  │  │  │  ├─ BUILD
│     │  │  │  │  ├─ pod_viewer.ng.html
│     │  │  │  │  ├─ pod_viewer.scss
│     │  │  │  │  ├─ pod_viewer.ts
│     │  │  │  │  ├─ pod_viewer_common.ts
│     │  │  │  │  ├─ pod_viewer_details
│     │  │  │  │  │  ├─ BUILD
│     │  │  │  │  │  ├─ pod_viewer_details.ng.html
│     │  │  │  │  │  ├─ pod_viewer_details.scss
│     │  │  │  │  │  ├─ pod_viewer_details.ts
│     │  │  │  │  │  └─ pod_viewer_details_module.ts
│     │  │  │  │  ├─ pod_viewer_module.ts
│     │  │  │  │  ├─ stack_bar_chart
│     │  │  │  │  │  ├─ BUILD
│     │  │  │  │  │  ├─ stack_bar_chart.ng.html
│     │  │  │  │  │  ├─ stack_bar_chart.scss
│     │  │  │  │  │  ├─ stack_bar_chart.ts
│     │  │  │  │  │  └─ stack_bar_chart_module.ts
│     │  │  │  │  └─ topology_graph
│     │  │  │  │     ├─ BUILD
│     │  │  │  │     ├─ topology_graph.ng.html
│     │  │  │  │     ├─ topology_graph.scss
│     │  │  │  │     ├─ topology_graph.ts
│     │  │  │  │     └─ topology_graph_module.ts
│     │  │  │  ├─ sidenav
│     │  │  │  │  ├─ BUILD
│     │  │  │  │  ├─ sidenav.ng.html
│     │  │  │  │  ├─ sidenav.scss
│     │  │  │  │  ├─ sidenav.ts
│     │  │  │  │  └─ sidenav_module.ts
│     │  │  │  ├─ tensorflow_stats
│     │  │  │  │  ├─ BUILD
│     │  │  │  │  ├─ flop_rate_chart
│     │  │  │  │  │  ├─ BUILD
│     │  │  │  │  │  ├─ flop_rate_chart.ng.html
│     │  │  │  │  │  ├─ flop_rate_chart.scss
│     │  │  │  │  │  ├─ flop_rate_chart.ts
│     │  │  │  │  │  └─ flop_rate_chart_module.ts
│     │  │  │  │  ├─ model_properties
│     │  │  │  │  │  ├─ BUILD
│     │  │  │  │  │  ├─ model_properties.ng.html
│     │  │  │  │  │  ├─ model_properties.scss
│     │  │  │  │  │  ├─ model_properties.ts
│     │  │  │  │  │  └─ model_properties_module.ts
│     │  │  │  │  ├─ operations_table
│     │  │  │  │  │  ├─ BUILD
│     │  │  │  │  │  ├─ operations_table.ng.html
│     │  │  │  │  │  ├─ operations_table.scss
│     │  │  │  │  │  ├─ operations_table.ts
│     │  │  │  │  │  ├─ operations_table_data_provider.ts
│     │  │  │  │  │  └─ operations_table_module.ts
│     │  │  │  │  ├─ stats_table
│     │  │  │  │  │  ├─ BUILD
│     │  │  │  │  │  ├─ stats_table.ng.html
│     │  │  │  │  │  ├─ stats_table.scss
│     │  │  │  │  │  ├─ stats_table.ts
│     │  │  │  │  │  ├─ stats_table_data_provider.ts
│     │  │  │  │  │  └─ stats_table_module.ts
│     │  │  │  │  ├─ tensorflow_stats.ng.html
│     │  │  │  │  ├─ tensorflow_stats.scss
│     │  │  │  │  ├─ tensorflow_stats.ts
│     │  │  │  │  ├─ tensorflow_stats_adapter.ts
│     │  │  │  │  └─ tensorflow_stats_module.ts
│     │  │  │  ├─ tf_data_bottleneck_analysis
│     │  │  │  │  ├─ BUILD
│     │  │  │  │  ├─ tf_data_bottleneck_analysis.ng.html
│     │  │  │  │  ├─ tf_data_bottleneck_analysis.scss
│     │  │  │  │  ├─ tf_data_bottleneck_analysis.ts
│     │  │  │  │  └─ tf_data_bottleneck_analysis_module.ts
│     │  │  │  └─ trace_viewer
│     │  │  │     ├─ BUILD
│     │  │  │     ├─ trace_viewer.ng.html
│     │  │  │     ├─ trace_viewer.scss
│     │  │  │     ├─ trace_viewer.ts
│     │  │  │     └─ trace_viewer_module.ts
│     │  │  ├─ pipes
│     │  │  │  ├─ BUILD
│     │  │  │  ├─ pipes_module.ts
│     │  │  │  └─ safe_pipe.ts
│     │  │  ├─ services
│     │  │  │  ├─ data_dispatcher
│     │  │  │  │  ├─ BUILD
│     │  │  │  │  ├─ data_dispatcher.ts
│     │  │  │  │  ├─ data_dispatcher_base.ts
│     │  │  │  │  └─ data_request_queue.ts
│     │  │  │  └─ data_service
│     │  │  │     ├─ BUILD
│     │  │  │     ├─ data_service.ts
│     │  │  │     └─ mock_data.ts
│     │  │  ├─ store
│     │  │  │  ├─ BUILD
│     │  │  │  ├─ actions.ts
│     │  │  │  ├─ common_data_store
│     │  │  │  │  ├─ BUILD
│     │  │  │  │  ├─ actions.ts
│     │  │  │  │  ├─ reducers.ts
│     │  │  │  │  ├─ selectors.ts
│     │  │  │  │  └─ state.ts
│     │  │  │  ├─ reducers.ts
│     │  │  │  ├─ selectors.ts
│     │  │  │  ├─ state.ts
│     │  │  │  ├─ store_module.ts
│     │  │  │  ├─ tensorflow_stats
│     │  │  │  │  ├─ BUILD
│     │  │  │  │  ├─ actions.ts
│     │  │  │  │  ├─ reducers.ts
│     │  │  │  │  ├─ selectors.ts
│     │  │  │  │  └─ state.ts
│     │  │  │  └─ types.ts
│     │  │  └─ styles
│     │  │     ├─ BUILD
│     │  │     └─ common.scss
│     │  ├─ index.html
│     │  ├─ main.ts
│     │  ├─ server.py
│     │  └─ styles.scss
│     ├─ install_and_run.py
│     ├─ package.json
│     ├─ plugin
│     │  ├─ BUILD
│     │  ├─ README.rst
│     │  ├─ build_pip_package.sh
│     │  ├─ setup.py
│     │  ├─ tensorboard_plugin_profile
│     │  │  ├─ BUILD
│     │  │  ├─ __init__.py
│     │  │  ├─ __pycache__
│     │  │  │  ├─ __init__.cpython-38.pyc
│     │  │  │  └─ version.cpython-38.pyc
│     │  │  ├─ build_utils
│     │  │  │  ├─ BUILD
│     │  │  │  ├─ profiler_test.bzl
│     │  │  │  ├─ pytype.default.bzl
│     │  │  │  └─ strict.default.bzl
│     │  │  ├─ convert
│     │  │  │  ├─ BUILD
│     │  │  │  ├─ __init__.py
│     │  │  │  ├─ dcn_collective_stats_proto_to_gviz.py
│     │  │  │  ├─ dcn_collective_stats_proto_to_gviz_test.py
│     │  │  │  ├─ diagnostics.py
│     │  │  │  ├─ diagnostics_test.py
│     │  │  │  ├─ input_pipeline_proto_to_gviz.py
│     │  │  │  ├─ input_pipeline_proto_to_gviz_test.py
│     │  │  │  ├─ kernel_stats_proto_to_gviz.py
│     │  │  │  ├─ kernel_stats_proto_to_gviz_test.py
│     │  │  │  ├─ overview_page_proto_to_gviz.py
│     │  │  │  ├─ overview_page_proto_to_gviz_test.py
│     │  │  │  ├─ raw_to_tool_data.py
│     │  │  │  ├─ tf_data_stats_proto_to_gviz.py
│     │  │  │  ├─ tf_data_stats_proto_to_gviz_test.py
│     │  │  │  ├─ tf_stats_proto_to_gviz.py
│     │  │  │  ├─ tf_stats_proto_to_gviz_test.py
│     │  │  │  ├─ trace_events_json.py
│     │  │  │  └─ trace_events_json_test.py
│     │  │  ├─ demo
│     │  │  │  ├─ BUILD
│     │  │  │  ├─ __init__.py
│     │  │  │  ├─ data
│     │  │  │  │  ├─ profile_demo.input_pipeline.json
│     │  │  │  │  ├─ profile_demo.input_pipeline.pb
│     │  │  │  │  ├─ profile_demo.memory_viewer.json
│     │  │  │  │  ├─ profile_demo.op_profile.json
│     │  │  │  │  ├─ profile_demo.overview_page.json
│     │  │  │  │  ├─ profile_demo.overview_page.pb
│     │  │  │  │  ├─ profile_demo.pod_viewer.json
│     │  │  │  │  └─ profile_demo.tensorflow_stats.pb
│     │  │  │  ├─ profile_demo.py
│     │  │  │  └─ profile_demo_data.py
│     │  │  ├─ integration_tests
│     │  │  │  ├─ BUILD
│     │  │  │  ├─ __init__.py
│     │  │  │  ├─ tf_mnist.py
│     │  │  │  ├─ tf_profiler_session.py
│     │  │  │  └─ tpu
│     │  │  │     └─ tensorflow
│     │  │  │        ├─ BUILD
│     │  │  │        ├─ __init__.py
│     │  │  │        └─ tpu_tf2_keras_test.py
│     │  │  ├─ profile_plugin.py
│     │  │  ├─ profile_plugin_loader.py
│     │  │  ├─ profile_plugin_test.py
│     │  │  ├─ profile_plugin_test_utils.py
│     │  │  ├─ protobuf
│     │  │  │  ├─ BUILD
│     │  │  │  ├─ __init__.py
│     │  │  │  ├─ dcn_slack_analysis.proto
│     │  │  │  ├─ diagnostics.proto
│     │  │  │  ├─ input_pipeline.proto
│     │  │  │  ├─ kernel_stats.proto
│     │  │  │  ├─ overview_page.proto
│     │  │  │  ├─ power_metrics.proto
│     │  │  │  ├─ tf_data_stats.proto
│     │  │  │  ├─ tf_stats.proto
│     │  │  │  └─ trace_events.proto
│     │  │  ├─ static
│     │  │  │  ├─ index.html
│     │  │  │  ├─ index.js
│     │  │  │  └─ materialicons.woff2
│     │  │  └─ version.py
│     │  ├─ third_party
│     │  │  ├─ tracing
│     │  │  │  ├─ BUILD
│     │  │  │  └─ trace_viewer_full.html.gz
│     │  │  └─ webcomponentsjs
│     │  │     ├─ BUILD
│     │  │     └─ webcomponents.js
│     │  └─ trace_viewer
│     │     ├─ BUILD
│     │     ├─ tf_trace_viewer
│     │     │  ├─ BUILD
│     │     │  ├─ data
│     │     │  │  ├─ BUILD
│     │     │  │  └─ trace.json
│     │     │  ├─ demo.html
│     │     │  ├─ tf-trace-viewer-helper.js
│     │     │  └─ tf-trace-viewer.html
│     │     ├─ trace_viewer.html
│     │     └─ webcomponentsjs_polyfill
│     │        ├─ BUILD
│     │        └─ webcomponentsjs_polyfill.html
│     ├─ requirements.txt
│     ├─ rollup.config.js
│     ├─ tsconfig.json
│     └─ yarn.lock
└─ tf-profiler
   ├─ BUILD
   ├─ __init__.py
   ├─ integration_test
   │  ├─ BUILD
   │  ├─ mnist_testing_utils.py
   │  └─ profiler_api_test.py
   ├─ internal
   │  ├─ BUILD
   │  ├─ _pywrap_profiler.pyi
   │  ├─ _pywrap_traceme.pyi
   │  ├─ flops_registry.py
   │  ├─ flops_registry_test.py
   │  ├─ model_analyzer_testlib.py
   │  ├─ print_model_analysis_test.py
   │  ├─ profiler_pywrap_impl.cc
   │  ├─ profiler_pywrap_impl.h
   │  ├─ profiler_wrapper.cc
   │  ├─ run_metadata_test.py
   │  └─ traceme_wrapper.cc
   ├─ model_analyzer.py
   ├─ model_analyzer_test.py
   ├─ option_builder.py
   ├─ pprof_profiler.py
   ├─ pprof_profiler_test.py
   ├─ profile_context.py
   ├─ profile_context_test.py
   ├─ profiler.py
   ├─ profiler_client.py
   ├─ profiler_client_test.py
   ├─ profiler_test.py
   ├─ profiler_v2.py
   ├─ profiler_v2_test.py
   ├─ profiler_wrapper_test.py
   ├─ tfprof_logger.py
   ├─ tfprof_logger_test.py
   └─ trace.py

```
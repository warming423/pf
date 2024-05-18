#!/bin/bash

# shellcheck source=./macos-common.sh
source "$(dirname "${BASH_SOURCE[0]}")/macos-common.sh"

git submodule sync --recursive
git submodule update --init --recursive
export CMAKE_PREFIX_PATH=${WORKSPACE_DIR}/miniconda3/

# Build PyTorch
if [ -z "${IN_CI}" ]; then
  export DEVELOPER_DIR=/Applications/Xcode9.app/Contents/Developer
fi

echo "BUILD_LITE_INTERPRETER: ${BUILD_LITE_INTERPRETER}"
if [ "${BUILD_LITE_INTERPRETER}" == 1 ]; then
    echo "Testing libtorch (lite interpreter)."

    CPP_BUILD="$PWD/../cpp-build"
    rm -rf "${CPP_BUILD}"
    mkdir -p "${CPP_BUILD}/caffe2"

    # It looks libtorch need to be built in "${CPP_BUILD}/caffe2"
    # folder.
    BUILD_LIBTORCH_PY=$PWD/tools/build_libtorch.py
    pushd "${CPP_BUILD}/caffe2" || exit
    VERBOSE=1 DEBUG=1 python "${BUILD_LIBTORCH_PY}"
    popd || exit

    "${CPP_BUILD}/caffe2/build/bin/test_lite_interpreter_runtime"
    "${CPP_BUILD}/caffe2/build/bin/test_mobile_nnc"

    # Change the permission manually from 755 to 644 to keep git clean
    chmod 644 "${HOME}/project/.jenkins/pytorch/macos-lite-interpreter-build-test.sh"
    assert_git_not_dirty
else
    echo "Skipping libtorch (lite interpreter)."
fi

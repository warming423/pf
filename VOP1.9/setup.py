import codecs
import os
import re
import setuptools
from setuptools import Extension

here = os.path.abspath(os.path.dirname(__file__))

def read(*parts):
    with codecs.open(os.path.join(here, *parts), "r") as fp:
        return fp.read()

def find_version(*file_paths):
    " find version information in prof.__init__.py "
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

long_description = read("README.md")
version = find_version("prof", "__init__.py")


extensions = []
main_compile_args = []
main_link_args = []
main_libraries = ['torch_python']
main_sources = ["csrc/stub.c"]
library_dirs=[]

lib_path = os.path.join(here, "torch", "lib")
library_dirs.append(lib_path)

C = Extension(  "torch._C",    # 模块名，编译生成的模块名称
                libraries=main_libraries,  # 需要链接的库
                sources=main_sources,      # 源文件列表，包括扩展模块的源代码文件
                language='c', 
                #extra_compile_args=main_compile_args + extra_compile_args,  # 额外的编译选项
                include_dirs=[],  # 需要包含的头文件路径
                library_dirs=library_dirs, # 需要链接的库文件路径
                #extra_link_args=extra_link_args + main_link_args + make_relative_rpath_args('lib') # 额外的链接选项
            ) 
extensions.append(C)


setuptools.setup(
    name="VOP",
    version=version,
    author="wangm",
    author_email="2435403746@qq.com",
    description="Operators Performance Visualization System ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/awwong1/torchprof",
    packages=setuptools.find_packages(), #####
    ext_modules=extensions,
    license="MIT",
    install_requires=["torch>=1.9.0,<2.3"], ######
    package_data={
        "stubs":['/*.pyi']
    },
    
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)

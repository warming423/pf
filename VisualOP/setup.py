import codecs
import os
import re
import setuptools

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

setuptools.setup(
    name="VisualOP",
    version=version,
    author="wangm",
    author_email="2435403746@qq.com",
    description="Operators Performance Visualization System ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/awwong1/torchprof",
    packages=setuptools.find_packages(), 
    license="MIT",
    install_requires=["torch>=1.9.0,<2.3"], 
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
    ],
)

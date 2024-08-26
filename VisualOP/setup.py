import codecs
import os
import re
import setuptools

def read(*parts):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, *parts), "r") as fp:
        return fp.read()

def find_version(*file_paths):
    " find version information in Prof.__init__.py "
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

setuptools.setup(
    name="VisualOP",
    version=find_version("Prof", "__init__.py"),
    author="wangm",
    author_email="2435403746@qq.com",
    description="Operators Performance Visualization System ",
    long_description= read("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/warming423/pf",
    packages=setuptools.find_packages(), 
    license="MIT",
    install_requires=["torch>=1.9.0,<2.3"], 
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
    ],
)

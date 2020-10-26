import setuptools

VERSION = "0.0.1"

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="pruning",
    version=VERSION,
    author="Xiaohu Tang",
    author_email="tigertang.zju@outlook.com",
    description="A tool to prune neural networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tigert1998/pruning",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.5',
)

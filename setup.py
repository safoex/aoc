import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="aoc",
    version="0.0.0",
    author="IIT",
    author_email="evgenii.safronov@iit.it",
    description="Ambiguous Object Classification module",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/safoex/aoc",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

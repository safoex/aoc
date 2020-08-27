import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mitmpose",
    version="0.0.0",
    author="IIT",
    author_email="evgenii.safronov@iit.it",
    description="Panda project: scene identification module",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/safoex/mitmpose",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

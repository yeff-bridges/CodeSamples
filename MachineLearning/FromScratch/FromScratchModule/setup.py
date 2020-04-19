import setuptools

with open("README.md", "r") as file:
    long_description = file.read()

setuptools.setup(
    name="FromScratch",
    version="0.0.1",
    author="Jeremy Frechette",
    author_email="JeremyAFrechette@gmail.com",
    description="Common ML Functions and Algorithms coded from scratch in Numpy",
    url="https://github.com/yeff-bridges/CodeSamples/FromScratch",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy",
        "matplotlib",
        "pandas",
    ],
    python_requires='>=3',
    package_data={
    },
)
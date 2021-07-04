import setuptools

setuptools.setup(
    name="ODELab", 
    version="0.1",
    author="Katharine Long",
    author_email="katharine.long@ttu.edu",
    description="Numerical solution of ordinary differential equations",
    long_description="Numerical solution of ordinary differential equations",
    long_description_content_type="text/markdown",
    url="https://github.com/krlong014/ODELab",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: LGPL License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='patternsum',
    version='1.1',
    packages=setuptools.find_packages(),
    url='https://github.com/SSripilaipong/patternsum',
    author='SSripilaipong',
    author_email='SHSnail@mail.com',
    description='An algorithm to summarize possible patterns covered in a list of strings using Genetic Algorithm.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

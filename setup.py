from setuptools import setup

setup(
    name='patternsum',
    version='1.0',
    packages=['patternsum', 'patternsum.core', 'patternsum.pattern', 'patternsum.optimizer'],
    url='https://github.com/SSripilaipong/string-pattern-summ',
    license='MIT',
    author='SSripilaipong',
    author_email='SHSnail@mail.com',
    description='An algorithm to summarize possible patterns covered in a list of strings using Genetic Algorithm.'
)

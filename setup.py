from setuptools import setup, find_packages

setup(
    name='hyperalignment',
    version='0.1',
    packages=find_packages(),
    scripts=['bin/npls', 'bin/rmdirs'],
)

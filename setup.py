from setuptools import setup, find_packages

setup(
    name='ect',
    version='0.1',
    packages=find_packages(),
    description='A python package for computing the Euler Characteristic Transform',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Liz Munch',
    author_email='muncheli@msu.edu',
    url='https://github.com/MunchLab/ect',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    install_requires=[
        # list of your package dependencies
        'networkx',
        'numpy',
        'numba',
        'pathlib',
        'argparse',
        'matplotlib'
    ],
)
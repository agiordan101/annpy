from setuptools import setup

setup(
    name='annpy',
    version='1.0.0',
    author='agiordan',
    description='Homemade machine learning library',
    keywords='lib',
    packages=[
        'annpy'
    ],
    long_description=open('README.md').read(),
    install_requires=[
        'numpy==1.19.3',
        'numba==0.43.1'
    ]
)

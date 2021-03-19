from setuptools import setup

setup(
    name='annpy',
    version='0.1.0',
    author='agiordan',
    description='homemade deeplearning library',
    keywords='lib',
    packages=[
        'annpy',
        # 'annpy.models',
        # 'annpy.layers',
        # 'annpy.activations',
        # 'annpy.losses',
        # 'annpy.optimizers',
        # 'annpy.callbacks',
        # 'annpy.initializers',
        # 'annpy.layers.connectors',
        # 'annpy.metrics',
        # 'annpy.utils',
    ],
    long_description=open('README.md').read(),
    install_requires=[
        'numpy==1.19.3',
        'numba==0.43.1'
    ]
)

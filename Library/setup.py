from setuptools import setup, find_packages

setup(
    name='risk_lib',
    version='1.0',
    description='A module for risk management',
    author='Suxiang Li',
    author_email='sl787@duke.edu',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scipy',
    ],
)
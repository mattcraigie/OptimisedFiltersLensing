from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    # ... other setup configuration ...
    install_requires=requirements,
)

setup(
    name='ostlensing',
    version='1.0.0',
    packages=['ostlensing'],
)


# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='thesisRI',
    version='0.1.0',
    description='Bachelor thesis Oleg Yarin',
    long_description=readme,
    author='Oleg Yarin',
    author_email='yarinolega@gmail.com',
    url='https://github.com/monapasan/bachelor-thesis',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

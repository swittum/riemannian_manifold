from setuptools import setup, find_packages

setup(
    name='riemannian_manifold',
    version='0.2',
    packages=find_packages(),
    author='Simon Wittum',
    author_email='simonwittum@gmx.de',
    description='A package for working with Riemannian manifolds in Python',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/swittum/riemannian_manifold',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)

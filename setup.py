from setuptools import setup, find_packages

setup(
    name='riemann_manifold',
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
    install_requires=[
        'numpy==1.26.4',
        'scipy==1.12.0',
        'sympy==1.12'
    ]p
)
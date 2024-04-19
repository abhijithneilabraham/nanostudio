# setup.py
from setuptools import setup, find_packages

setup(
    name='nanostudio',
    version='0.1.0',
    author='Abhijith Neil Abraham',
    author_email='abhijithneilabrahampk@gmail.com',
    description='A lightweight deep learning framework for training and evaluating GPT models',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/abhijithneilabraham/nanostudio',
    packages=find_packages(),
    install_requires=[
        'torch>=1.7.1',
        'numpy',
        'transformers'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.7',
    entry_points={
        'console_scripts': [
            'nanostudio-train=nanostudio.examples.train_example:main',
            'nanostudio-eval=nanostudio.examples.eval_example:main',
        ],
    }
)

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    README = readme_file.read()

setup(
  name = 'aft_pytorch',
  packages = find_packages(exclude=[]),
  version = '0.2.2',
  license='MIT',
  description = 'Attention Free Transformer - Pytorch',
  long_description_content_type="text/markdown",
  long_description=README,
  author = 'Rishabh Anand',
  author_email = 'mail.rishabh.anand@gmail.com',
  url = 'https://github.com/rish-16/aft-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'attention free transformer',
    'self-attention',
    'transformer',
    'natural language processing'
  ],
  install_requires=[
    'torch>=1.6'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
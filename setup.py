from setuptools import setup

requires = ['astropy', 'numpy', 'matplotlib']

setup(name='gsspy',
      version='0.1.0',
      description='A package to deal with interface with GSSP.',
      author='Kevin Gullikson',
      author_email='kevin.gullikson@gmail.com',
      license='MIT',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python',
          'Topic :: Scientific/Engineering :: Astronomy',
      ],
      packages=['gsspy'],
      requires=requires)

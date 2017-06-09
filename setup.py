from setuptools import setup

setup(name='waview',
      version='0.1',
      description='View audio files in the terminal',
      url='https://github.com/richardmitic/waview',
      author='Richard Mitic',
      license='MIT',
      install_requires=[
          'scipy',
      ],
      packages=['waview'],
      scripts=['bin/waview'])

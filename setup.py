from setuptools import setup, find_packages
import logging

# By default the root logger is set to WARNING and all loggers you define
# inherit that value. Here we set the root logger to NOTSET. This logging
# level is automatically inherited by all existing and new sub-loggers
# that do not set a less verbose level.
logging.root.setLevel(logging.NOTSET)

# The following line sets the root logger level as well.
# It's equivalent to both previous statements combined:
logging.basicConfig(level=logging.NOTSET)

with open('requirements.txt') as f:
    required = f.read().splitlines()


setup(name="deep_nilmtk",
      version="0.1.1",
      packages=find_packages(),
      description="deep_nilmtk",
      install_requires= required,
      zip_safe=False)





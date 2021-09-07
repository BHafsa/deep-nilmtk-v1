"""
 Pre-processing module is a module responsible of analysing the raw energy data as provided from the NILMtk
 API. It contains a pre-processing sub-module that defines the different data transformations to be applied 
 to the input data (e.g., data normalisation). It focuses on the input data, while the output data is included 
 in the data loader as some models requires states generation. 
"""

from .pre_processing import *
from .states import *
from .threshold import *
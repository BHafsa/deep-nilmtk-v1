"""
This package contains the definition of an abstract disaggregation class compatible with all PyTorch models defined in the config module and compatible with the API 
of NILMtk. It thus relieves the user from the burden of data formatting and interfacing with NILMtk. 
"""

from .nilm_experiment import NILMExperiment
# -*- coding: utf-8 -*-
"""
    This package contain a set of map-style PyTorch datasets for the different models implemented in the tool. 
    Through the use of this class,  Deep-NILMtk implements a generic and memory-efficient data model that allows
    feeding batches of data to the deep neural network during model training. It thus offers a set of ready-to-use 
    data generators with the goal of speeding up the research process and promoting re-usability of code.
"""
from .bertdataset import *
from .generaldataset import *
from .tempooldataset import *
from .wavenetdataset import *
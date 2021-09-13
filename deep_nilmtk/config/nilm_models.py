# -*- coding: utf-8 -*-
"""
This variable defines the models and their corresponding data loaders as well as extra-parameters
when needed. The mapping specified here will be directly used during disaggregation.


.. list-table:: Liste of available models
   :widths: 25 25 50
   :header-rows: 1

   * - Model name
     - DataLoader
     - Link to original paper
   * - :ref:`Seq-to-Seq<seqseq>`
     - :ref:`General data loader<generaldataset>`
     - `<https://dl.acm.org/doi/pdf/10.1145/3360322.3360844>`_
   * - :ref:`DAE<dae>`
     - :ref:`General data loader<generaldataset>`
     - `<https://dl.acm.org/doi/pdf/10.1145/3360322.3360844>`_
   * - :ref:`Seq-to-Point<seqpoint>`
     - :ref:`General data loader<generaldataset>`
     - `<https://dl.acm.org/doi/pdf/10.1145/3360322.3360844>`_
   * - :ref:`RNN<rnn>`
     - :ref:`General data loader<generaldataset>`
     - `<https://dl.acm.org/doi/pdf/10.1145/3360322.3360844>`_
   * - :ref:`WindowGRU<gru>`
     - :ref:`General data loader<generaldataset>`
     - `<https://dl.acm.org/doi/pdf/10.1145/3360322.3360844>`_
   * - :ref:`UNET<unet>`
     - :ref:`General data loader<generaldataset>`
     - `<https://dl.acm.org/doi/pdf/10.1145/3427771.3427859>`_
   * - :ref:`BERT4NILM<bert>`
     - :ref:`Bert data loader<bertdataset>`
     - `<https://dl.acm.org/doi/pdf/10.1145/3427771.3429390>`_
   * - :ref:`Temp-Pooling<ptp>`
     - :ref:`Temporal-Pooling data loader<ptpdataset>`
     - `<https://arxiv.org/pdf/2010.16050.pdf>`_
   * - :ref:`WAVENILM<wavenilm>`
     - :ref:`WaveNet data loader<wavenetdataset>`
     - `<https://dl.acm.org/doi/pdf/10.1145/3441300>`_
"""
from ..loader import *
from ..model import *
from ..preprocessing import *

NILM_MODELS = {
    # =============================================================================
    #     UNETNILM with uncertainty estimation
    # =============================================================================
    'UNETNiLMS2P': {
        'model': UNETNILM,
        'loader': generalDataLoader,
        'extra_params':{}
    },
    
    'UNETNiLMSeq2Quantile': {
        'model': UNETNILMSeq2Quantile,
        'loader': generalDataLoader,
        'extra_params':{}
    },
  

    # =============================================================================
    #     Temporal Pooling model
    # =============================================================================
    'tempPool':{
        'model': PTPNet,
        'loader': TemPoolLoader,
        'extra_params':{
        }
        },
    # =============================================================================
    #     Bert4NILM implementation from 
    # =============================================================================
    'BERT4NILM': {
        'model': BERT4NILM,
        'loader': BERTDataset,
        'extra_params':{
            
        }
    },
    # =============================================================================
    #     Baseline models from nilmtk-contrib in pytorch
    # =============================================================================
    'Seq2Pointbaseline': {
        'model': Seq2Point,
        'loader': generalDataLoader,
        'extra_params':{}
    },
    'RNNbaseline': {
        'model': RNN,
        'loader': generalDataLoader,
        'extra_params':{}
    },
    
    'WindowGRUbaseline': {
        'model': WindowGRU,
        'loader': generalDataLoader,
        'extra_params':{}
    },
    
    'Seq2Seqbaseline':{
        'model': Seq2Seq,
        'loader': generalDataLoader,
        'extra_params':{}
        },
    
    'DAE':{
        'model': DAE,
        'loader': generalDataLoader,
        'extra_params':{}
        },
    
    'WAVENILM':{
        'model': WaveNet,
        'loader': WaveNetDataLoader,
        'extra_params':{}
        },

    'WaveNetBGRU':{
        'model': WaveNetBGRU,
        'loader': WaveNetDataLoader,
        'extra_params':{}
        },
    
    'WaveNetBGRU_speedup':{
        'model': WaveNetBGRU_speedup,
        'loader': WaveNetDataLoader,
        'extra_params':{}
        },

}
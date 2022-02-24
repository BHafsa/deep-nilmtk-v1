import deep_nilmtk.models.tensorflow as KerasModels
import deep_nilmtk.models.pytorch as TorchModels

import deep_nilmtk.data.loader.tensorflow as KerasLoader
import deep_nilmtk.data.loader.pytorch as TorchLoader

__models__ = {
    'tensorflow': {
        'Seq2Pointbaseline':{
            'model': KerasModels.Seq2Point,
            'loader':None
        }
    },
    'pytorch':{
            # =============================================================================
            #     Baseline models from nilmtk-contrib in pytorch
            # =============================================================================
            'Seq2Pointbaseline': {
                'model': TorchModels.seq2point.Seq2Point,
                'loader': TorchLoader.GeneralDataLoader,
                'extra_params':{
                    'point_position': 'mid_point',
                    'sequence_type':'seq2point'
                }
            },
            'RNNbaseline': {
                'model': TorchModels.seq2point.RNN,
                'loader': TorchLoader.GeneralDataLoader,
                'extra_params':{}
            },

            'WindowGRUbaseline': {
                'model': TorchModels.seq2point.WindowGRU,
                'loader': TorchLoader.GeneralDataLoader,
                'extra_params':{}
            },

            'Seq2Seqbaseline':{
                'model': TorchModels.seq2seq.Seq2Seq,
                'loader': TorchLoader.GeneralDataLoader,
                'extra_params':{}
            },

            'DAE':{
                'model': TorchModels.seq2seq.DAE,
                'loader': TorchLoader.GeneralDataLoader,
                'extra_params':{}
            },


    }
}
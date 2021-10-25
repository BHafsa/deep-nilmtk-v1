Add a New Model
=======================================

Adding a new model would require several steps to be followed so that 
it will be comaptibale with the deep_nilmtk toolkit and take advanatge of all features 
included in the tool. 

1. :Model definition:

The neural network architecture is defined as a usual PyTorch model with two extra 
mandatory function: the step funtion and the prediction. The first one calculates the 
predictions of for a batch of data and return the loss. It used during training. 
The second is the testing loop and uses pre-trained model to generate predictions 
for the testing data.

.. note::
   For Seq-to-Seq models, the aggregation of sequences is performed at the end of the predict function.

.. code-block:: python
    
    import torch.nn as nn
    import torch.nn.functional as F
    
    class NewModel(nn.Module):
        def __init__(self):
            # Model definition

        def step(self, batch):
            x, y  = batch 
            out   = self(x)  # BxCxT

            error = (y - out)
            loss = F.mse_loss(out, y)
            mae = error.abs().data.mean()
            return  loss, mae 

        def predict(self,  model, test_dataloader):
            ...
            results = {"pred":pred}
            return results


2. :DataLoader Definition:

The corresponding dataLoader of the model is defined as standard PyTorch dataset.
The user has the choice of defining a new class or using existing data loaders.

.. code-block:: python
    
    import torch
    
    class NewDataLoader(torch.utils.data.Dataset):
        def __init__(self, inputs,  targets=None,  params= {}):
            # Defines the padding of the daa and the targets Normalization
        def __len__(self):
            # Denotes the total number of samples
        def get_sample(self, index):
            # Defines how samples as generated
        def __getitem__(self, index):
            # Generate a sample of data
        

3. :Adding the model to the configurationof the tool:

The model and the loader should be linked together in the 
config/nilm_models.py file as follows:

.. code-block:: python
   
   NILM_MODELS = {
       'new_model_name':{
            'model': NewModel,
            'loader': NewDataLoader,
            'extra_params':{
            }
        },
   }



At this point, the model is ready for experimenting. It can be tested directly using the
NILMtk-API as the already available baselines.


   
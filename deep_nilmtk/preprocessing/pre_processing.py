import numpy as np
import pandas as pd

def get_differential_power(data):
    """
    Generate differential power := p[t+1]-p[t]

    :param data: The  power data
    :type data: np.array
    :return: The differentiated power
    :rtype: np.array
    """

    return np.ediff1d(data, to_end=None, to_begin=0)   

 
def get_variant_power(data, alpha=0.1):
    """
    Generate variant power which reduce noise that may impose negative influence on  pattern identification
    Parameters
    ----------
    
    data(np.array) :  power signal
    alpha(float) :  reflection rate
    
    Returns
    ----------
    variant_power: The variant power generated
    """
    variant_power = np.zeros(len(data))
    for i in range(1,len(data)):
        d = data[i]-variant_power[i-1]
        variant_power[i] = variant_power[i-1] + alpha*d     
    return  variant_power 

    

def get_percentile(data,p=50):
    """
    Calculates the percentile p of the data
    
    Parameters
    ----------
        data (np.array) : The power data
        p (np.array) : The quantile 
    
    Returns
    ----------
        np.array : The quantile values of the power data
    """
    return np.percentile(data, p, axis=1, interpolation="midpoint")


def over_lapping_sliding_window(data, seq_len = 4, step_size = 1):
    """
    Generates overlappping sequences using the sliding sequence approach.

    Parameters
    ----------
        data (np.array): Power data
        seq_len (int, optional): The length of the sequences. Defaults to 4.
        step_size (int, optional): The step size. Defaults to 1.

    Returns
    ----------
        np.array: An array of the generated sequences.
    """
    units_to_pad = (seq_len//2)*step_size
    new_data = np.pad(data, (units_to_pad,units_to_pad),'constant',constant_values=(0,0))
    sh = (new_data.size - seq_len, seq_len)
    st = new_data.strides * 2
    sequences = np.lib.stride_tricks.as_strided(new_data, strides = st, shape = sh)[0::step_size]
    return sequences.copy()



def quantile_filter(data:np.array, sequence_length:int=10,  p:int=50):
    """ 
    Applies quantile filter on the input data.

    Parameters
    ----------
        data (np.array): The input data power data.
        sequence_length (int): The length of sequence
        p (int, optional): The percentile. Defaults to 50.

    Returns
    ----------
        np.array: array of values for correponding percentile
    """
    new_mains = over_lapping_sliding_window(data, sequence_length)
    new_mains = get_percentile(new_mains, p)
    return new_mains


def get_differential_power(data):
    """
    The differences between consecutive elements of an array.

    Parameters
    ----------
        data (np.array): the input data 

    Returns
    ----------
        np.array: The differences.
    """
    return np.ediff1d(data, to_end=None, to_begin=0)   
    


def get_variant_power(data, alpha=0.1):

    """
    Generate variant power which reduce noise that may impose negative inluence on  pattern identification

    Parameters
    ----------
        data:  power signal
        alpha[0,0.9]:  reflection rate

    Returns
    ----------
        variant_power: The variant power generated
    """


    variant_power = np.zeros(len(data))
    for i in range(1,len(data)):
            d = data[i]-variant_power[i-1]
            variant_power[i] = variant_power[i-1] + alpha*d     
    return  variant_power 


def get_temporal_info(data):
    """
    Generates the temporal information related 
    power consumption

    Parameters
    ----------
        data (list(DatetimeIndex)): a list of all temporal information

    Returns
    ----------
        np.array: Temporal contextual information of the energy data
    """
    
    out_info =[]
    
    for d in data:
        seconds = (d - d.iloc[0]).dt.total_seconds().values / np.max((d - d.iloc[0]).dt.total_seconds().values)
        minutes = d.dt.minute.values / 60
        hod = d.dt.hour.values / 24
        dow = d.dt.dayofweek.values / 7
    
        out_info.append([seconds, minutes, hod, dow])
    
    return  np.transpose(np.array(out_info)).reshape((-1,4))



def data_preprocessing(aggregate, targets=None, 
                       feature_type="vpower",
                       alpha=0.1,
                       normalize=None, 
                       main_mu=329, 
                       main_std=450,
                       q_filter={"q":50, "w":10},
                       main_min=0,
                       main_max = 1500,):
    

    """
    Default pre-processing function. It performs normalization of the input. 
    However, it leaves the target output normlization to the dataloader as 
    some loaders require to also generate the states from the the original data.

    Parameters
    ----------
        aggregate: the aggregate power consumption
        feature_type([str]): the type of input features to derive from the aggregate power, default 'main' 
        alpha ([float]): reflection rate
        normalize ([str]): normalization type
        main_mu ([float]): the mean of the aggregate power data
        main_std ([float]): the std of the aggregate power data
        main_min ([float]): the min of the aggregate power data
        main_max ([float]): the max of the aggregate power data
        q_filter: quantile filters

    Returns
    ----------
        aggregate power, submetered data , submetered data: [description]
    """

    

    if targets is not None: 
        processed_mains_lst = []
        temporal_data = []
        for i in range(len(aggregate)):
          
            main = aggregate[i].values.flatten()
    
            temporal_data.append(aggregate[i].index.to_series())
            #apply quantile filter as proposed in https://dl.acm.org/doi/10.1145/3427771.3427859
            processed_mains_lst.append(pd.DataFrame(main))   
            
        appliance_list = []
        for app_index, (app_name, app_df_lst) in enumerate(targets):
            processed_app_dfs = []
            for app_df in app_df_lst:                    
                new_app_readings = app_df.values.flatten() 
                '''  
                if q_filter is not None:
                    #apply quantile filter as proposed in https://dl.acm.org/doi/10.1145/3427771.3427859
                    new_app_readings  = quantile_filter(data=new_app_readings, sequence_length=q_filter['w'],  p=q_filter['q'])
                '''
                processed_app_dfs.append(pd.DataFrame(new_app_readings))  
            appliance_list.append((app_name, processed_app_dfs))
                    
        sub_meters = []
        sub_aggregate = []
        for app_name, app_dfs in appliance_list:
            app_df = pd.concat(app_dfs, axis=0)
            sub_aggregate.append(app_df.values)
            sub_meters.append((app_name, app_df.values))
            
        sub_aggregate = np.hstack(sub_aggregate)
        main = pd.concat(processed_mains_lst, axis=0).values.flatten()
        main = np.where(main < sub_aggregate.sum(axis=1), sub_aggregate.sum(axis=1), main)

        if q_filter is not None:
            main = quantile_filter(data=main , sequence_length=q_filter['w'],  p=q_filter['q'])
         
        # print(main)
        if normalize is not None:
            if normalize=="lognorm":
                main = np.log1p(main)
            elif normalize == 'min-max':
                main = (main - main_min) / (main_max - main_min)
            else:
                main= (main - main_mu)/main_std
                
            
        if feature_type=='mains':
            main =  main[:, np.newaxis]

        elif feature_type=="diff":
            #aggregate = np.log1p(aggregate)
            main = get_differential_power(main)
            
        elif feature_type=="vpower" and alpha>0:
            #aggregate = np.log1p(aggregate)
            main = get_variant_power.wer(main, alpha)
            
        elif feature_type=="combined" and alpha>0:
            dp = get_differential_power(main)[:,None]
            dv = get_variant_power(main, alpha)[:,None]
            dvp = get_differential_power(dv.flatten())[:,None]
            main = np.concatenate([main[:,None], dv, dp, dvp], axis=1)
        
        elif feature_type=="combined_with_time" and alpha>0:
            dp = get_differential_power(main)[:,None]
            dv = get_variant_power(main, alpha)[:,None]
            dvp = get_differential_power(dv.flatten())[:,None]
            tmp = get_temporal_info(temporal_data)
            main = np.concatenate([tmp, main[:,None],  dv, dp, dvp], axis=1)
            
            
        return main, sub_aggregate,  sub_meters   
    else:
        processed_mains = []
        for main in aggregate: 
            temporal_data = main.index.to_series()
            main = main.values.flatten()
            if q_filter is not None:
                main = quantile_filter(data=main , sequence_length=q_filter['w'],  p=q_filter['q'])  
            
            if normalize is not None:
                if normalize=="lognorm":
                    main = np.log1p(main)
                elif normalize == 'min-max':
                    main = (main - main_min) / (main_max - main_min)
                else:
                    main= (main - main_mu)/main_std
            
            if feature_type=="diff":
                #aggregate = np.log1p(aggregate)
                main = get_differential_power(main)
            
            elif feature_type=="vpower" and alpha>0:
                #aggregate = np.log1p(aggregate)
                main = get_variant_power(main, alpha)
            
            elif feature_type=="combined" and alpha>0:
                dp = get_differential_power(main)[:,None]
                dv = get_variant_power(main, alpha)[:,None]
                dvp = get_differential_power(dv.flatten())[:,None]
                main = np.concatenate([main[:,None], dv, dp, dvp], axis=1) 

            elif feature_type=="combined_with_time" and alpha>0:
                dp = get_differential_power(main)[:,None]
                dv = get_variant_power(main, alpha)[:,None]
                dvp = get_differential_power(dv.flatten())[:,None]
                tmp = get_temporal_info([temporal_data])

                main = np.concatenate([tmp, main[:,None],  dv, dp, dvp], axis=1)              
               
            processed_mains.append(pd.DataFrame(main))
        return processed_mains
        
    


import numpy as np
import pandas as pds

def getData(batch_size=64, string_size=32, 
            normalize=True, shuffle_batches=False, 
            file_location='btcusdt5min.csv'):
    '''
    return batches: (Array) The preprocessed and batched data
                    [num_batches, batch_size, string_size, sample_dim]
    '''
    
    data = pds.read_csv('btcusdt5min.csv')
    
    data = representData(data)
    
    if normalize is True:
        data = normalizeRepresentation(data)
        
    batches = getBatches(data, batch_size=batch_size, string_size=string_size, shuffle=shuffle_batches)
    batches = batches.astype(np.float32)
    
    return batches
    
def representData(data):
    '''
    Takes pandas csv OHLCV candles and returns a numpy representation
    '''
    
    #cast to numpy
    data = data.to_numpy()
    
    #truncate formatted time
    data = np.delete(data, 6, axis=1)
    
    #set chronology to start at 0 miliseconds
    data[:,5] -= data[:,5].min()
    
    #turn green/red indicator into a binary int
    data[:,6] = (data[:,6] == 'green').astype(np.int32)
    
    return data

def normalizeRepresentation(data):
    '''
    Takes OHLCV candle data as a numpy array and regularizes the dimensions
    
    Excludes time and green/red
    '''
    
    for col in range(5):
        #center by subtracting mean
        data[:,col] -= (np.sum(data[:,col])/data.shape[0])
        
        #normalize
        data[:,col] = data[:,col] / data[:,col].max()
        
    return data

def getBatches(data, batch_size=32, string_size=32, shuffle=False):
    '''
    Chunks data into batches, with length along the sample dimension according to string_size
    
    returns (num_samples / batch_size * string_size) batches
    '''
    
    num_samples = data.shape[0]
    sample_dim = data[0].shape[0]
    num_batches = int(num_samples / (batch_size * string_size))
    
    batched = np.reshape(data[:(num_batches*batch_size*string_size)], 
                         [num_batches, batch_size, string_size, sample_dim])
    
    return batched
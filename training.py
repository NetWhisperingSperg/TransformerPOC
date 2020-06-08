import tensorflow as tf
import numpy as np

from transformer import Transformer, MLP

def train_model(model, epochs, data):
    '''
    The function containing the main logic for the training sequence
    
    param model: (tf.keras.Model) the model to be trained
    param epochs: (Int) the number of epochs to train over
    param data: (Tuple) a tuple of the training and validation data
    
    return train_loss: (List) the training loss for each batch
    return val_loss: (List) the validation loss calculated at each checkpoint
    '''
    
    
    
def maskData(data, mask_prob=0.1):
    '''
    Takes input data and applies a zero-mask with probability mask_prob. If time_structured
    the sequence is unveiled one unit at a time
    
    param data: (Array) the incoming data to mask
                    Shape: [batch, seq_length, seq_dim]
    param mask_prob: (Float) the probability of applying a mask
    
    return masked: (Array) the masked data
    return mask: (Array) the mask that was applied
    '''
    
    batch, seq_length, seq_dim = data.shape
    mask = np.zeros(shape=data.shape)
    
    #select indices to mask
    for b in range(batch):
        for l in range(seq_length):
            for d in range(seq_dim):
                if (np.random.uniform() >= mask_prob):
                    mask[b,l,d] = 1
    
    return (data * mask), mask
    
def squaredDistanceLoss(x, y):
    '''
    Takes predictions and their true values and calculates the squared euclidian distance
    
    param x: (Tensor) the predicted values
                Shape: [batch, seq_length, num_dim]
    param y: (Tensor) the true values
                Shape: [batch, seq_lenth, num_dim]
    '''
    
    summed = tf.einsum('bsd->', (x-y)**2)
    
    return summed
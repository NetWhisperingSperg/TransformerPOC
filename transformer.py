import numpy as np
import tensorflow as tf



class Transformer(tf.keras.Model):
    '''
    The full transformer model
    
    param num_layers: (Int) The number of seperate layers in the encoder and decoder
    param d_model: (Int) The dimensionality used for the embedded input and the model itself
    param d_ff: (Int) The number of latent units for the dense layer at the end of each encoding/decoding step
    param num_heads: (Int) The number of attention heads for the multi-head attention algorithm
    param dropout: (Float) The probability of the application for the dropout algorithm
    '''
    
    def __init__(self, num_layers=6, d_model=512, d_ff=2048, num_heads=8, dropout=0.1):
        super(Transformer, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.dropout = dropout
        
        self.embedding = Embedding(embedding_dim=d_model)
        
        self.encoder = Encoder(num_layers=num_layers,
                               d_model=d_model, 
                               num_heads=num_heads, 
                               d_ff=d_ff, 
                               dropout=dropout)
        self.decoder = Decoder(num_layers=num_layers,
                               d_model=d_model, 
                               num_heads=num_heads, 
                               d_ff=d_ff, 
                               dropout=dropout)
        
        
    def call(self, x):
        
        embedding = self.embedding(x)
        
        encoding = self.encoder(embedding)
        
        decoding = self.decoder(embedding, encoding)
        
        return decoding
    
    
class MultiHeadAttention(tf.keras.Model):
    '''
    The multi-head attention algorithm. Applies the attention calculation on the data seperately
    num_heads times and concatenates the final results
    
    param d_model: (Int) The dimensionality used for the embedded input and the model itself
    param num_heads: (Int) The number of attention heads for the multi-head attention algorithm
    param dropout: (Float) The probability of the application for the dropout algorithm
    '''
    
    def __init__(self, d_model=512, num_heads=8, dropout=None):
        super(MultiHeadAttention, self).__init__()
        
        #the model dimensionality must be divisible by the number of attention heads
        assert d_model % num_heads == 0
        
        d_head = d_model // num_heads
        
        #create num_heads attention heads
        self.attentionheads = []
        for head in range(num_heads):
            self.attentionheads.append(AttentionHead(d_model=d_model, d_head=d_head, dropout=dropout))
        
        #linear transformation of the concatenated attention heads
        self.dense = tf.keras.layers.Dense(units=d_model)
        
    def call(self, query, key, value):
        '''
        param query/key/value: (Tensor) values to feed to the query/key/value function
                        Shape: [batch, seq_length, d_model]
                        
        return dense: (Tensor) The attention values for the input
                        Shape: [batch, seq_length, d_model]
        '''
        
        #apply all attention heads and concatenate the output
        attention = []
        for head in self.attentionheads:
            attention.append(head(query, key, value))
            
        attention = tf.concat(attention, axis=0)
        
        #apply the final linear transformation
        dense = self.dense(attention)
        
        return dense
    
class AttentionHead(tf.keras.Model):
    '''
    An individual attention head of the multi-head attention algorithm. Finds query, key and values and then
    applies the attention equation
    
    param d_model: (Int) The dimensionality used for the embedded input and the model itself
    param d_head: (Int) The dimensionality of the attention head
    param dropout: (Float) The probability of the application for the dropout algorithm
    '''
    
    def __init__(self, d_model=512, d_head=64, dropout=None):
        super(AttentionHead, self).__init__()
        
        self.d_head = d_head
        
        self.query = tf.keras.layers.Dense(units=d_head)
        self.key = tf.keras.layers.Dense(units=d_head)
        self.value = tf.keras.layers.Dense(units=d_head)
        
        self.dropout = tf.keras.layers.Dropout(rate=dropout)
        
    def call(self, query, key, value):
        '''
        The attention algorithm for a single head
        
        param query/key/value: (Tensor) values to feed to the query/key/value function
                        Shape: [batch, seq_length, d_model]
                        
        return dense: (Tensor) The attention values for the input
                        Shape: [batch, seq_length, d_head]
        '''
        
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)
        
        #attention(Q,K,V) = softmax(QK^t / root(d_head))V
        
        #matmul QK^T
        scores = tf.einsum('bsd,bsd->bsd', Q, K)
        
        #scale by attention dimensionality
        scores = scores / np.sqrt(self.d_head)
        
        #softmax, dropout and apply value
        attention = tf.nn.softmax(scores)
        attention = self.dropout(attention)
        attention = tf.einsum('bsd,bsd->bsd', attention, V)
        
        return attention
    
class SublayerNorm(tf.keras.Model):
    '''
    Applies layer normalization and residual connection for the encoding and decoding layers
    
    param d_model: (Int) The dimensionality used for the embedded input and the model itself
    param dropout: (Float) The probability of the application for the dropout algorithm
    '''
    
    def __init__(self, d_model=512, dropout=None):
        super(SublayerNorm, self).__init__()
        
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = tf.keras.layers.Dropout(rate=dropout)
        
    def call(self, x, sublayer):
        '''
        Applies layer normalization, dropout, and residual connection
        
        param x: (Tensor) input tensor for the layer
                    Shape: [batch, seq_length, d_model]
        param sublayer: (Tensor) the output tensor of the previous sublayer
                            Shape: [batch, seq_length, d_model]
                            
        return normedconnection: (Tensor) the normed input with residual connection and dropout
                                    Shape: [batch, seq_length, d_model]
        '''
        
        norm = self.norm(sublayer)
        
        normedconnection = self.dropout(norm) + x
        
        return normedconnection
    
class Encoder(tf.keras.Model):
    '''
    The full encoding network for the transformer
    
    param num_layers: (Int) The number of seperate layers in the encoder
    param d_model: (Int) The dimensionality used for the embedded input and the model itself
    param d_ff: (Int) The number of latent units for the dense layer at the end of each encoding
    param num_heads: (Int) The number of attention heads for the multi-head attention algorithm
    param dropout: (Float) The probability of the application for the dropout algorithm
    '''
    
    def __init__(self, num_layers=6, d_model=512, d_ff=2048, num_heads=8, dropout=0.1):
        super(Encoder, self).__init__()
        
        self.encoderlayers = []
        
        for layer in range(num_layers):
            self.encoderlayers.append(EncoderLayer(d_model=d_model, d_ff=d_ff, num_heads=num_heads, dropout=0.1))
            
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            
    def call(self, x):
        '''
        Gives the output for the full encoding network
        
        param x: (Tensor) The embedded sequence
                    Shape: [batch, seq_length, d_model]
        
        return encoding: (Tensor) The now fully-encoded input
                    Shape: [batch, seq_length, d_model]
        '''
        for layer in self.encoderlayers:
            x = layer(x)
        
        encoding = self.norm(x)
        
        return encoding
    
class EncoderLayer(tf.keras.Model):
    '''
    The encoding layers for the full encoder
    
    param d_model: (Int) The dimensionality used for the embedded input and the model itself
    param d_ff: (Int) The number of latent units for the dense layer at the end of each encoding
    param num_heads: (Int) The number of attention heads for the multi-head attention algorithm
    param dropout: (Float) The probability of the application for the dropout algorithm
    '''
    
    def __init__(self, d_model=512, d_ff=2048, num_heads=8, dropout=0.1):
        super(EncoderLayer, self).__init__()
        
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.attentionnorm = SublayerNorm(d_model=d_model, dropout=dropout)
        
        self.dense = tf.keras.layers.Dense(units=d_ff)
        self.densenorm = SublayerNorm(d_model=d_model, dropout=dropout)
        
    def call(self, x):
        '''
        param x: (Tensor) The input tensor to be encoded
                    shape: [batch, seq_length, d_model]
                    
        return encoded: (Tensor) The encoded tensor
                        shape: [batch, seq_lenth, d_model]
        '''
        
        attention = self.attention(x, x, x)
        attentionnorm = self.attentionnorm(x, attention)
        
        dense = self.dense(attentionnorm)
        encoded = self.densenorm(dense)
        
        return encoded
    
class Decoder(tf.keras.Model):
    '''
    The full decoding network for the transformer
    
    param num_layers: (Int) The number of seperate layers in the decoder
    param d_model: (Int) The dimensionality used for the embedded input and the model itself
    param d_ff: (Int) The number of latent units for the dense layer at the end of each decoding step
    param num_heads: (Int) The number of attention heads for the multi-head attention algorithm
    param dropout: (Float) The probability of the application for the dropout algorithm
    '''
    
    def __init__(self, num_layers=6, d_model=512, d_ff=2048, num_heads=8, dropout=0.1):
        super(Decoder, self).__init__()
        
        self.decoderlayers = []
        
        for layer in range(num_layers):
            self.decoderlayers.append(DecoderLayer(d_model=d_model, d_ff=d_ff, num_heads=num_heads, dropout=0.1))
            
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        
    def call(self, x, encoding):
        '''
        Gives the final decoded output of the transformer
        
        param x: (Tensor) The sequence to decode
                    Shape: [batch, seq_length, d_model]
        param encoding: (Tensor) The output of the encoding network
                            Shape: [batch, seq_length, d_model]
                            
        return decodedseq: (Tensor) The final decoded output of the transformer
                            Shape: [batch, seq_length, d_model]
        '''
        
        for layer in self.decoderlayers:
            x = layer(x, encoding)
            
        decodedseq = self.norm(x)
        
        return decodedseq
    
class DecoderLayer(tf.keras.Model):
    '''
    The decoding layers for the decoding network
    
    param d_model: (Int) The dimensionality used for the embedded input and the model itself
    param d_ff: (Int) The number of latent units for the dense layer at the end of each decoding step
    param num_heads: (Int) The number of attention heads for the multi-head attention algorithm
    param dropout: (Float) The probability of the application for the dropout algorithm
    '''
    
    def __init__(self, d_model=512, d_ff=2048, num_heads=8, dropout=0.1):
        super(DecoderLayer, self).__init__()
        
        
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.attentionnorm = SublayerNorm(d_model=d_model, dropout=dropout)
        
        self.encoderattention = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.encoderattentionnorm = SublayerNorm(d_model=d_model, dropout=dropout)

        self.dense = tf.keras.layers.Dense(units=d_ff)
        self.densenorm = SublayerNorm(d_model=d_model, dropout=dropout)
        
    def call(self, x, encoding):
        '''
        param x: (Tensor) the input tensor to be decoded
                    Shape: [batch, seq_length, d_model]
        param encoding: (Tensor) the final output of the encoder network
                            Shape: [batch, seq_length, d_model]
                            
        return decoding: (Tensor) the decoded output of the layer
                            Shape: [batch, seq_length, d_model]
        '''
        
        attention = self.attention(x, x, x)
        attentionnorm = self.attentionnorm(x, attention)
        
        encoderattention = self.encoderattention(encoding, encoding, attentionnorm)
        encoderattentionnorm = self.encoderattentionnorm(attentionnorm, encoderattention)
        
        dense = self.dense(encoderattentionnorm)
        decoding = self.densenorm(dense)
        
        return decoding


class Embedding(tf.keras.Model):
    '''
    Takes incoming low-dimensional data and applies a dense-net transformation
    into a higher dimensional embedding space
    
    param embedding_dim: (Int) The dimension to embed into
    param activation: (tf.nn.Function) The activation function for the transformation
    '''
    
    def __init__(self, embedding_dim=512, activation=None):
        super(Embedding,self).__init__()
        self.dense = tf.keras.layers.Dense(units=embedding_dim, activation=activation)
        
    def call(self, x):
        '''
        param x: (Tensor) The value sequence to embed
                    Shape: [batch, seq_length, sample_dim]
        return embedding: (Tensor) A high-dimension embedded vector
                        Shape: [batch, seq_length, embedding_dim]
        '''
        return self.dense(x)
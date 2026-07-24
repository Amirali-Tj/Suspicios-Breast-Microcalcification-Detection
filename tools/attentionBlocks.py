import keras
from keras import layers

def spatial_attention(tensor) :
  AvgPool = keras.ops.mean(tensor  , axis=-1 , keepdims=True)
  MaxPool = keras.ops.max(tensor   , axis=-1 , keepdims=True)
  concat_features    = layers.Concatenate(axis=-1)([AvgPool , MaxPool])
  Conv_attention     = layers.Conv2D(filters=1 , kernel_size=(7 , 7) , padding='same')(concat_features)
  activate_attention = layers.Activation(activation='sigmoid')(Conv_attention)
  return layers.Multiply()([tensor , activate_attention])

def channel_attention(tensor , rr=16) :
  def shared_MLP(input_T) :
    n_channels = input_T.shape[-1]
    hidden_layer = layers.Dense(n_channels//rr , activation='relu' , kernel_initializer='he_normal')(input_T)
    output       = layers.Dense(n_channels)(hidden_layer)
    return output
  #----
  GAP = layers.GlobalAveragePooling2D(keepdims=True)(tensor)
  GMP = layers.GlobalMaxPooling2D(keepdims=True)(tensor)
  #----
  GAP_out = shared_MLP(GAP)
  GMP_out = shared_MLP(GMP)
  #----
  add = layers.Add()([GAP_out , GMP_out])
  activate_attention = layers.Activation(activation='sigmoid')(add)
  return layers.Multiply()([tensor , activate_attention])

def CBAM(tensor) :
  sp_att   = channel_attention(tensor)
  cbam_att = spatial_attention(sp_att)
  return cbam_att
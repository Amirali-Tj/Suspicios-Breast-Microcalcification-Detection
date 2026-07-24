import keras
from keras import layers , models , Model

def insert_layer(base_model , insert_func) :
  
  resnet_add_cbam = models.clone_model(
      base_model ,
      input_tensors=base_model.input ,
      call_function=insert_func
  )

  weight_dict = {}
  for layer in base_model.layers :
    weight_dict[layer.name] = layer.get_weights()
  for layer in resnet_add_cbam.layers :
    try :
      layer_weights = weight_dict[layer.name]
      layer.set_weights(layer_weights)
      layer.trainable = False
    except :
      pass

  return resnet_add_cbam

def top_cls(backbone , hidden_unit=32) :
  output = layers.GlobalAveragePooling2D()(backbone.output)
  output = layers.Dense(hidden_unit , activation='relu' , kernel_initializer='he_normal')(output)
  output = layers.Dense(1 , activation='sigmoid')(output)
  model  = Model(inputs=backbone.input , outputs=output , name='resnet50_cbam')
  return model
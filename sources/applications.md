# Applications

Keras Applications are deep learning models that are made available alongside pre-trained weights.
These models can be used for prediction, feature extraction, and fine-tuning.

Weights are downloaded automatically when instantiating a model. They are stored at `~/.keras/models/`.

## Available models

### Models for image classification with weights trained on ImageNet:

- [Xception](#xception)
- [VGG16](#vgg16)
- [VGG19](#vgg19)
- [ResNet, ResNetV2, ResNeXt](#resnet)
- [InceptionV3](#inceptionv3)
- [InceptionResNetV2](#inceptionresnetv2)
- [MobileNet](#mobilenet)
- [MobileNetV2](#mobilenetv2)
- [DenseNet](#densenet)
- [NASNet](#nasnet)

All of these architectures are compatible with all the backends (TensorFlow, Theano, and CNTK), and upon instantiation the models will be built according to the image data format set in your Keras configuration file at `~/.keras/keras.json`. For instance, if you have set `image_data_format=channels_last`, then any model loaded from this repository will get built according to the TensorFlow data format convention, "Height-Width-Depth".

Note that:
- For `Keras < 2.2.0`, The Xception model is only available for TensorFlow, due to its reliance on `SeparableConvolution` layers.
- For `Keras < 2.1.5`, The MobileNet model is only available for TensorFlow, due to its reliance on `DepthwiseConvolution` layers.

-----

## Usage examples for image classification models

### Classify ImageNet classes with ResNet50

```python
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

model = ResNet50(weights='imagenet')

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])
# Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]
```

### Extract features with VGG16

```python
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

model = VGG16(weights='imagenet', include_top=False)

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)
```

### Extract features from an arbitrary intermediate layer with VGG19

```python
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np

base_model = VGG19(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

block4_pool_features = model.predict(x)
```

### Fine-tune InceptionV3 on a new set of classes

```python
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(200, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# train the model on the new data for a few epochs
model.fit_generator(...)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model.fit_generator(...)
```


### Build InceptionV3 over a custom input tensor

```python
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input

# this could also be the output a different Keras model or layer
input_tensor = Input(shape=(224, 224, 3))  # this assumes K.image_data_format() == 'channels_last'

model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=True)
```

-----

# Documentation for individual models

| Model | Size | Top-1 Accuracy | Top-5 Accuracy | Parameters | Depth |
| ----- | ----: | --------------: | --------------: | ----------: | -----: |
| [Xception](#xception) | 88 MB | 0.790 | 0.945 | 22,910,480 | 126 |
| [VGG16](#vgg16) | 528 MB | 0.713 | 0.901 | 138,357,544 | 23 |
| [VGG19](#vgg19) | 549 MB | 0.713 | 0.900 | 143,667,240 | 26 |
| [ResNet50](#resnet) | 98 MB | 0.749 | 0.921 | 25,636,712 | - |
| [ResNet101](#resnet) | 171 MB | 0.764 | 0.928 | 44,707,176 | - |
| [ResNet152](#resnet) | 232 MB | 0.766 | 0.931 | 60,419,944 | - |
| [ResNet50V2](#resnet) | 98 MB | 0.760 | 0.930 | 25,613,800 | - |
| [ResNet101V2](#resnet) | 171 MB | 0.772 | 0.938 | 44,675,560 | - |
| [ResNet152V2](#resnet) | 232 MB | 0.780 | 0.942 | 60,380,648 | - |
| [ResNeXt50](#resnet) | 96 MB | 0.777 | 0.938 | 25,097,128 | - |
| [ResNeXt101](#resnet) | 170 MB | 0.787 | 0.943 | 44,315,560 | - |
| [InceptionV3](#inceptionv3) | 92 MB | 0.779 | 0.937 | 23,851,784 | 159 |
| [InceptionResNetV2](#inceptionresnetv2) | 215 MB | 0.803 | 0.953 | 55,873,736 | 572 |
| [MobileNet](#mobilenet) | 16 MB | 0.704 | 0.895 | 4,253,864 | 88 |
| [MobileNetV2](#mobilenetv2) | 14 MB | 0.713 | 0.901 | 3,538,984 | 88 |
| [DenseNet121](#densenet) | 33 MB | 0.750 | 0.923 | 8,062,504 | 121 |
| [DenseNet169](#densenet) | 57 MB | 0.762 | 0.932 | 14,307,880 | 169 |
| [DenseNet201](#densenet) | 80 MB | 0.773 | 0.936 | 20,242,984 | 201 |
| [NASNetMobile](#nasnet) | 23 MB | 0.744 | 0.919 | 5,326,716 | - |
| [NASNetLarge](#nasnet) | 343 MB | 0.825 | 0.960 | 88,949,818 | - |

The top-1 and top-5 accuracy refers to the model's performance on the ImageNet validation dataset.

Depth refers to the topological depth of the network. This includes activation layers, batch normalization layers etc.

-----


## Xception


```python
keras.applications.xception.Xception(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
```

Xception V1 model, with weights pre-trained on ImageNet.

On ImageNet, this model gets to a top-1 validation accuracy of 0.790
and a top-5 validation accuracy of 0.945.

Note that this model only supports the data format `'channels_last'` (height, width, channels).

The default input size for this model is 299x299.

### Arguments

- include_top: whether to include the fully-connected layer at the top of the network.
- weights: one of `None` (random initialization) or `'imagenet'` (pre-training on ImageNet).
- input_tensor: 모델의 이미지 인풋으로 사용할 수 있는 선택적 케라스 텐서 (다시말해, `layers.Input()`의 아웃풋).
- input_shape: 선택적 형태 튜플로,
    `include_top`이 `False`일 경우만 특정하십시오.
    (그렇지 않다면 인풋의 형태가 `(229, 229 3)`이어야 합니다).
    인풋 채널이 정확히 3개여야 하며
    넓이와 높이가 71 미만이어서는 안됩니다.
    예시. `(150, 150, 3)`은 유효한 값입니다.
- pooling: Optional pooling mode for feature extraction
    when `include_top` is `False`.
    - `None` means that the output of the model will be
        the 4D tensor output of the
        last convolutional layer.
    - `'avg'` means that global average pooling
        will be applied to the output of the
        last convolutional layer, and thus
        the output of the model will be a 2D tensor.
    - `'max'` means that global max pooling will
        be applied.
- classes: optional number of classes to classify images 
    into, only to be specified if `include_top` is `True`, and 
    if no `weights` argument is specified.

### Returns

A Keras `Model` instance.

### References

- [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)

### License

These weights are trained by ourselves and are released under the MIT license.


-----


## VGG16

```python
keras.applications.vgg16.VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
```

ImageNet에 대해 가중치가 선행학습된 VGG16 모델

이 모델에는 `'channels_first'` 데이터 포맷(채널, 높이, 넓이)과 `'channels_last'` 데이터 포맷(높이, 넓이, 채널) 둘 모두 사용할 수 있습니다.

이 모델의 디폴트 인풋 사이즈는 224x224 입니다.

### 인수

- include_top: 네트워크의 최상단에 3개의 완전 연결 레이어를 넣을지 여부.
- weights: `None` (임의의 초기값 설정) 혹은 `'imagenet'` (ImageNet에 대한 선행 학습) 중 하나.
- input_tensor: 모델의 이미지 인풋으로 사용할 수 있는 선택적 케라스 텐서 (다시말해, `layers.Input()`의 아웃풋).
- input_shape: 선택적 형태 튜플로,
    `include_top`이 `False`일 경우만 특정하십시오.
    (그렇지 않다면 인풋의 형태가 `(224, 224, 3)`이고 `'channels_last'` 데이터 포맷을 취하거나
    혹은 인풋의 형태가 `(3, 224, 224)`이고 `'channels_first'` 데이터 포맷을 취해야 합니다).
    인풋 채널이 정확히 3개여야 하며
    넓이와 높이가 32 미만이어서는 안됩니다.
    예시. `(200, 200, 3)`은 유효한 값입니다.
- pooling: 특성추출을 위한 선택적 풀링 모드로,
    `include_top`이 `False`일 경우 유효합니다.
    - `None`은 모델의 아웃풋이
        마지막 컨볼루션 레이어의
        4D 텐서 아웃풋임을 의미합니다.
    - `'avg'`는 글로벌 평균값 풀링이
        마지막 컨볼루션 레이어의
        아웃풋에 적용되어
        모델의 아웃풋이 2D 텐서가 됨을 의미합니다.
    - `'max'`는 글로벌 최대값 풀링이
        적용됨을 의미합니다.
- classes: 이미지를 분류하기 위한 선택적 클래스의 수로, 
    `include_top`이 `True`일 경우, 
    그리고 `weights` 인수가 따로 정해지지 않은 경우만 특정합니다.

### 반환값

케라스 `Model` 인스텐스

### 참고

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556): 작업에 VGG 모델을 사용하는 경우 이 논문을 인용해주십시오.

### 라이센스

이 가중치는 [Creative Commons Attribution License](https://creativecommons.org/licenses/by/4.0/)에 따라 다음의 가중치에서 복사되었습니다 [released by VGG at Oxford](http://www.robots.ox.ac.uk/~vgg/research/very_deep/). 

-----

## VGG19


```python
keras.applications.vgg19.VGG19(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
```


ImageNet에 대해 가중치가 선행학습된 VGG19 모델.

이 모델에는 `'channels_first'` 데이터 포맷(채널, 높이, 넓이)과 `'channels_last'` 데이터 포맷(높이, 넓이, 채널) 둘 모두 사용할 수 있습니다.

이 모델의 디폴트 인풋 사이즈는 224x224입니다.

### 인수

- include_top: 네트워크의 최상단에 3개의 완전 연결 레이어를 넣을지 여부.
- weights: `None` (임의의 초기값 설정) 혹은 `'imagenet'` (ImageNet에 대한 선행 학습) 중 하나.
- input_tensor: 모델의 이미지 인풋으로 사용할 수 있는 선택적 케라스 텐서 (다시말해, `layers.Input()`의 아웃풋).
- input_shape: 선택적 형태 튜플로,
    `include_top`이 `False`일 경우만 특정하십시오.
    (그렇지 않다면 인풋의 형태가 `(224, 224, 3)`이고 `'channels_last'` 데이터 포맷을 취하거나
    혹은 인풋의 형태가 `(3, 224, 224)`이고 `'channels_first'` 데이터 포맷을 취해야 합니다).
    인풋 채널이 정확히 3개여야 하며
    넓이와 높이가 32 미만이어서는 안됩니다.
    예시. `(200, 200, 3)`은 유효한 값입니다.
- pooling: 특성추출을 위한 선택적 풀링 모드로,
    `include_top`이 `False`일 경우 유효합니다.
    - `None`은 모델의 아웃풋이
        마지막 컨볼루션 레이어의
        4D 텐서 아웃풋임을 의미합니다.
    - `'avg'`는 글로벌 평균값 풀링이
        마지막 컨볼루션 레이어의
        아웃풋에 적용되어
        모델의 아웃풋이 2D 텐서가 됨을 의미합니다.
    - `'max'`는 글로벌 최대값 풀링이
        적용됨을 의미합니다.
- classes: 이미지를 분류하기 위한 선택적 클래스 수로, 
    `include_top`이 `True`일 경우, 
    그리고 `weights` 인수가 따로 정해지지 않은 경우만 특정합니다.

### 반환값

케라스 `Model` 인스턴스


### 참고

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

### 라이센스

이 가중치는 [Creative Commons Attribution License](https://creativecommons.org/licenses/by/4.0/)에 따라 다음의 가중치에서 복사되었습니다 [released by VGG at Oxford](http://www.robots.ox.ac.uk/~vgg/research/very_deep/).

-----

## ResNet


```python
keras.applications.resnet.ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
keras.applications.resnet.ResNet101(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
keras.applications.resnet.ResNet152(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
keras.applications.resnet_v2.ResNet50V2(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
keras.applications.resnet_v2.ResNet101V2(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
keras.applications.resnet_v2.ResNet152V2(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
keras.applications.resnext.ResNeXt50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
keras.applications.resnext.ResNeXt101(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
```


ImageNet에 대해 가중치가 선행학습된 ResNet 모델, ResNetV2 모델, ResNeXt 모델.

이 모델에는 `'channels_first'` 데이터 포맷(채널, 높이, 넓이)과 `'channels_last'` 데이터 포맷(높이, 넓이, 채널) 둘 모두 사용할 수 있습니다.

이 모델의 디폴트 인풋 사이즈는 224x224입니다.


### 인수

- include_top: 네트워크의 최상단에 완전 연결 레이어를 넣을지 여부.
- weights: `None` (임의의 초기값 설정) 혹은 `'imagenet'` (ImageNet에 대한 선행 학습) 중 하나.
- input_tensor: 모델의 이미지 인풋으로 사용할 수 있는 선택적 케라스 텐서 (다시말해, `layers.Input()`의 아웃풋).
- input_shape: 선택적 형태 튜플로,
    `include_top`이 `False`일 경우만 특정하십시오.
    (그렇지 않다면 인풋의 형태가 `(224, 224, 3)`이고 `'channels_last'` 데이터 포맷을 취하거나
    혹은 인풋의 형태가 `(3, 224, 224)`이고 `'channels_first'` 데이터 포맷을 취해야 합니다).
    인풋 채널이 정확히 3개여야 하며
    넓이와 높이가 32 미만이어서는 안됩니다.
    예시. `(200, 200, 3)`은 유효한 값입니다.
- pooling: 특성추출을 위한 선택적 풀링 모드로,
    `include_top`이 `False`일 경우 유효합니다.
    - `None`은 모델의 아웃풋이
        마지막 컨볼루션 레이어의
        4D 텐서 아웃풋임을 의미합니다.
    - `'avg'`는 글로벌 평균값 풀링이
        마지막 컨볼루션 레이어의
        아웃풋에 적용되어
        모델의 아웃풋이 2D 텐서가 됨을 의미합니다.
    - `'max'`는 글로벌 최대값 풀링이
        적용됨을 의미합니다.
- classes: 이미지를 분류하기 위한 선택적 클래스의 수로, 
    `include_top`이 `True`일 경우, 
    그리고 `weights` 인수가 따로 정해지지 않은 경우만 특정합니다.

### 반환값

케라스 `Model` 인스턴스.

### 참고

- `ResNet`: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- `ResNetV2`: [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)
- `ResNeXt`: [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431)

### 라이센스

이 가중치는 다음의 출처에서 복사되었습니다:

- `ResNet`: [The original repository of Kaiming He](https://github.com/KaimingHe/deep-residual-networks) [MIT license](https://github.com/KaimingHe/deep-residual-networks/blob/master/LICENSE).
- `ResNetV2`: [Facebook](https://github.com/facebook/fb.resnet.torch) [BSD license](https://github.com/facebook/fb.resnet.torch/blob/master/LICENSE).
- `ResNeXt`: [Facebook AI Research](https://github.com/facebookresearch/ResNeXt) [BSD license](https://github.com/facebookresearch/ResNeXt/blob/master/LICENSE).

-----

## InceptionV3


```python
keras.applications.inception_v3.InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
```

ImageNet에 대해 가중치가 선행학습된 Inception V3.

이 모델에는 `'channels_first'` 데이터 포맷(채널, 높이, 넓이)과 `'channels_last'` 데이터 포맷(높이, 넓이, 채널) 둘 모두 사용할 수 있습니다.

이 모델의 디폴트 인풋 사이즈는 299x299 입니다.


### 인수

- include_top: 네트워크의 최상단에 완전 연결 레이어를 넣을지 여부.
- weights: `None` (임의의 초기값 설정) 혹은 `'imagenet'` (ImageNet에 대한 선행 학습) 중 하나.
- input_tensor: 모델의 이미지 인풋으로 사용할 수 있는 선택적 케라스 텐서 (다시말해, `layers.Input()`의 아웃풋).
- input_shape: 선택적 형태 튜플로,
    `include_top`이 `False`일 경우만 특정하십시오.
    (그렇지 않다면 인풋의 형태가 `(299, 299, 3)`이고 `'channels_last'` 데이터 포맷을 취하거나
    혹은 인풋의 형태가 `(3, 299, 299)`이고 `'channels_first'` 데이터 포맷을 취해야 합니다).
    인풋 채널이 정확히 3개여야 하며
    넓이와 높이가 75 미만이어서는 안됩니다.
    예시. `(150, 150, 3)`은 유효한 값입니다.
- pooling: 특성추출을 위한 선택적 풀링 모드로,
    `include_top`이 `False`일 경우 유효합니다.
    - `None`은 모델의 아웃풋이
        마지막 컨볼루션 레이어의
        4D 텐서 아웃풋임을 의미합니다.
    - `'avg'`는 글로벌 평균값 풀링이
        마지막 컨볼루션 레이어의
        아웃풋에 적용되어
        모델의 아웃풋이 2D 텐서가 됨을 의미합니다.
    - `'max'`는 글로벌 최대값 풀링이
        적용됨을 의미합니다.
- classes: 이미지를 분류하기 위한 선택적 클래스의 수로, 
    `include_top`이 `True`일 경우, 
    그리고 `weights` 인수가 따로 정해지지 않은 경우만 특정합니다.

### 반환값

케라스 `Model` 인스턴스.

### 참고

- [Rethinking the Inception Architecture for Computer Vision](http://arxiv.org/abs/1512.00567)

### 라이센스

이 가중치는 [the Apache License](https://github.com/tensorflow/models/blob/master/LICENSE)에 따라 배포되었습니다.

-----

## InceptionResNetV2


```python
keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
```

ImageNet에 대해 가중치가 선행학습된 Inception-ResNet V2 모델.

이 모델에는 `'channels_first'` 데이터 포맷(채널, 높이, 넓이)과 `'channels_last'` 데이터 포맷(높이, 넓이, 채널) 둘 모두 사용할 수 있습니다.

이 모델의 디폴트 인풋 사이즈는 299x299 입니다.


### 인수

- include_top: 네트워크의 최상단에 완전 연결 레이어를 넣을지 여부.
- weights: `None` (임의의 초기값 설정) 혹은 `'imagenet'` (ImageNet에 대한 선행 학습) 중 하나.
- input_tensor: 모델의 이미지 인풋으로 사용할 수 있는 선택적 케라스 텐서 (다시말해, `layers.Input()`의 아웃풋).
- input_shape: 선택적 형태 튜플로,
    `include_top`이 `False`일 경우만 특정하십시오.
    (그렇지 않다면 인풋의 형태가 `(299, 299, 3)`이고 `'channels_last'` 데이터 포맷을 취하거나
    혹은 인풋의 형태가 `(3, 299, 299)`이고 `'channels_first'` 데이터 포맷을 취해야 합니다).
    인풋 채널이 정확히 3개여야 하며
    넓이와 높이가 75 미만이어서는 안됩니다.
    예시. `(150, 150, 3)`은 유효한 값입니다.
- pooling: 특성추출을 위한 선택적 풀링 모드로,
    `include_top`이 `False`일 경우 유효합니다.
    - `None`은 모델의 아웃풋이
        마지막 컨볼루션 레이어의
        4D 텐서 아웃풋임을 의미합니다.
    - `'avg'`는 글로벌 평균값 풀링이
        마지막 컨볼루션 레이어의
        아웃풋에 적용되어
        모델의 아웃풋이 2D 텐서가 됨을 의미합니다.
    - `'max'`는 글로벌 최대값 풀링이
        적용됨을 의미합니다.
- classes: 이미지를 분류하기 위한 선택적 클래스의 수로, 
    `include_top`이 `True`일 경우, 
    그리고 `weights` 인수가 따로 정해지지 않은 경우만 특정합니다.

### 반환값

케라스 `Model` 인스턴스.

### 참고

- [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261)

### 라이센스

이 가중치는 [the Apache License](https://github.com/tensorflow/models/blob/master/LICENSE)에 따라 배포되었습니다.

-----

## MobileNet


```python
keras.applications.mobilenet.MobileNet(input_shape=None, alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)
```

ImageNet에 대해 가중치가 선행학습된 MobileNet 모델.

이 모델은 `'channels_last'` 데이터 포맷(높이, 넓이, 채널)만 지원하니 참고하십시오.

이 모델의 디폴트 인풋 사이즈는 224x224 입니다.

### 인수

- input_shape: 선택적 형태 튜플로,
    `include_top`이 `False`일 경우만 특정하십시오.
    (그렇지 않다면 인풋의 형태가 `(224, 224, 3)`이고 `'channels_last'` 데이터 포맷을 취하거나
    혹은 인풋의 형태가 `(3, 224, 224)`이고 `'channels_first'` 데이터 포맷을 취해야 합니다).
    인풋 채널이 정확히 3개여야 하며
    넓이와 높이가 32 미만이어서는 안됩니다.
    예시. `(200, 200, 3)`은 유효한 값입니다.
- alpha: 네트워크의 넓이를 조정합니다.
    - `alpha` < 1.0 인 경우, 그에 비례해서 각 레이어의
        필터 숫자를 감소시킵니다.
    - `alpha` > 1.0 인 경우, 그에 비례해서 각 레이어의
        필터 숫자를 증가시킵니다.
    - `alpha` = 1 인 경우, 각 레이어의 필터의 수가
        참고 논문에 따른 디폴트 값으로 정해집니다.
- depth_multiplier: 깊이별 컨볼루션의 깊이 승수
    (해상도 승수라고도 합니다)
- dropout: 드롭아웃 속도
- include_top: 네트워크의 최상단에
    완전연결 레이어를 넣을지 여부.
- weights: `None` (임의의 초기값 설정) 혹은
    `'imagenet'` (ImageNet에 대한 선행 학습)
- input_tensor: 모델의 이미지 인풋으로 사용할 수 있는
    선택적 케라스 텐서
    (다시말해, `layers.Input()`의 아웃풋).
- pooling: 특성추출을 위한 선택적 풀링 모드로,
    `include_top`이 `False`일 경우 유효합니다.
    - `None`은 모델의 아웃풋이
        마지막 컨볼루션 레이어의
        4D 텐서 아웃풋임을 의미합니다.
    - `'avg'`는 글로벌 평균값 풀링이
        마지막 컨볼루션 레이어의
        아웃풋에 적용되어
        모델의 아웃풋이 
        2D 텐서가 됨을 의미합니다.
    - `'max'`는 글로벌 최대값 풀링이
        적용됨을 의미합니다.
- classes: 이미지를 분류하기 위한 선택적 클래스의 수로, 
    `include_top`이 `True`일 경우, 
    그리고 `weights` 인수가 따로 정해지지 않은 경우만 특정합니다.

### 반환값

케라스 `Model` 인스턴스.

### 참고

- [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf)

### 라이센스

이 가중치는 [the Apache License](https://github.com/tensorflow/models/blob/master/LICENSE)에 따라 배포되었습니다.

-----

## DenseNet


```python
keras.applications.densenet.DenseNet121(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
keras.applications.densenet.DenseNet169(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
keras.applications.densenet.DenseNet201(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
```

ImageNet에 대해 가중치가 선행학습된 DenseNet.

이 모델에는 `'channels_first'` 데이터 포맷(채널, 높이, 넓이)과 `'channels_last'` 데이터 포맷(높이, 넓이, 채널) 둘 모두 사용할 수 있습니다.

이 모델의 디폴트 인풋 사이즈는 224x224 입니다.

### 인수

- blocks: 4개의 밀집 레이어의 빌딩 블록의 수.
- include_top: 네트워크의 최상단에
    완전연결 레이어를 넣을지 여부.
- weights: `None` (임의의 초기값 설정),
    `'imagenet'` (ImageNet에 대한 선행 학습), 혹은
    가중치 파일을 불러올 경로 중 하나.
- input_tensor: 모델의 이미지 인풋으로 사용할 수 있는 선택적 케라스 텐서
    (다시말해, `layers.Input()`의 아웃풋).
- input_shape: 선택적 형태 튜플로,
    `include_top`이 `False`일 경우만 특정하십시오.
    (그렇지 않다면 인풋의 형태가 `(224, 224, 3)`이고 `'channels_last'` 데이터 포맷을 취하거나
    혹은 인풋의 형태가 `(3, 224, 224)`이고 `'channels_first'` 데이터 포맷을 취해야 합니다).
    인풋 채널이 정확히 3개여야 하며
    넓이와 높이가 32 미만이어서는 안됩니다.
    예시. `(200, 200, 3)`은 유효한 값입니다.
- pooling: 특성추출을 위한 선택적 풀링 모드로,
    `include_top`이 `False`일 경우 유효합니다.
    - `None`은 모델의 아웃풋이
        마지막 컨볼루션 레이어의
        4D 텐서 아웃풋임을 의미합니다.
    - `'avg'`는 글로벌 평균값 풀링이
        마지막 컨볼루션 레이어의
        아웃풋에 적용되어
        모델의 아웃풋이 2D 텐서가 됨을 의미합니다.
    - `'max'`는 글로벌 최대값 풀링이
        적용됨을 의미합니다.
- classes: 이미지를 분류하기 위한 선택적 클래스의 수로, 
    `include_top`이 `True`일 경우, 
    그리고 `weights` 인수가 따로 정해지지 않은 경우만 특정합니다.

### 반환값

케라스 `Model` 인스턴스.

### 참고

- [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) (CVPR 2017 Best Paper Award)

### License

이 가중치는 [the BSD 3-clause License](https://github.com/liuzhuang13/DenseNet/blob/master/LICENSE)에 따라 배포되었습니다.

-----

## NASNet


```python
keras.applications.nasnet.NASNetLarge(input_shape=None, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)
keras.applications.nasnet.NASNetMobile(input_shape=None, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)
```

ImageNet에 대해 가중치가 선행학습된 Neural Architecture Search Network (NASNet) 모델.

NASNetLarge 모델의 디폴트 인풋사이즈는 331x331 이고
NASNetMobile 모델은 224x224 입니다.

### 인수

- input_shape: 선택적 형태 튜플로,
    `include_top`이 `False`일 경우만 특정하십시오.
    (그렇지 않다면 NASNetMobile의 경우 인풋의 형태가 `(224, 224, 3)`이고 
    `'channels_last'` 데이터 포맷을 취하거나 혹은 인풋의 형태가 `(3, 224, 224)`이고 
    `'channels_first'` 데이터 포맷을 취해야 하며, NASNetLarge의 경우  
    인풋이 `(331, 331, 3)`에 `'channels_last'` 데이터 포맷, 
    혹은 인풋이 `(3, 331, 331)`에 `'channels_first'` 데이터 포맷이어야 합니다).
    인풋 채널이 정확히 3개여야 하며
    넓이와 높이가 32 미만이어서는 안됩니다.
    예시. `(200, 200, 3)`은 유효한 값입니다.
- include_top: 네트워크의 최상단에
    완전연결 레이어를 넣을지 여부.
- weights: `None` (임의의 초기값 설정) 혹은
    `'imagenet'` (ImageNet에 대한 선행 학습)
- input_tensor: 모델의 이미지 인풋으로 사용할 수 있는
    선택적 케라스 텐서
    (다시말해, `layers.Input()`의 아웃풋).
- pooling: 특성추출을 위한 선택적 풀링 모드로,
    `include_top`이 `False`일 경우 유효합니다.
    - `None`은 모델의 아웃풋이
        마지막 컨볼루션 레이어의
        4D 텐서 아웃풋임을 의미합니다.
    - `'avg'`는 글로벌 평균값 풀링이
        마지막 컨볼루션 레이어의
        아웃풋에 적용되어
        모델의 아웃풋이 
        2D 텐서가 됨을 의미합니다.
    - `'max'`는 글로벌 최대값 풀링이
        적용됨을 의미합니다.
- classes: 이미지를 분류하기 위한 선택적 클래스의 수로, 
    `include_top`이 `True`일 경우, 
    그리고 `weights` 인수가 따로 정해지지 않은 경우만 특정합니다.

### 반환값

케라스 `Model` 인스턴스.

### 참고

- [Learning Transferable Architectures for Scalable Image Recognition](https://arxiv.org/abs/1707.07012)

### 라이센스

이 가중치는 [the Apache License](https://github.com/tensorflow/models/blob/master/LICENSE)에 따라 배포되었습니다.

-----

## MobileNetV2


```python
keras.applications.mobilenet_v2.MobileNetV2(input_shape=None, alpha=1.0, depth_multiplier=1, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)
```

ImageNet에 대해 가중치가 선행학습된 MobileNetV2 model.

이 모델은 `'channels_last'` 데이터 포맷(높이, 넓이, 채널)만 지원하니 참고하십시오.

이 모델의 디폴트 인풋 사이즈는 224x224 입니다.

### 인수

- input_shape: 선택적 형태 튜플로,
    (224, 224, 3)이 아닌 인풋 이미지 해상도를 사용할 경우에만
    특정하십시오.
    인풋 채널이 정확히 3개(224, 224, 3)이어야 합니다.
    input_tensor에서 input_shape을 추론하고 싶다면
    이 옵션을 생략해도 됩니다.
    input_tensor와 input_shape 둘 모두를 활용하는 경우
    둘의 형태가 매치된다면 input_shape이 사용되고,
    그렇지 않다면 에러를 알립니다.
    예시. `(160, 160, 3)`은 유효한 값입니다.
- alpha: 네트워크의 넓이를 조정합니다.
    이는 MobileNetV2 논문에서 넓이 승수로 기술됩니다.
    - `alpha` < 1.0 인 경우, 그에 비례해서 각 레이어의
        필터 숫자를 감소시킵니다.
    - `alpha` > 1.0 인 경우, 그에 비례해서 각 레이어의
        필터 숫자를 증가시킵니다.
    - `alpha` = 1 인 경우, 각 레이어의 필터의 수가
        참고 논문에 따른 디폴트 값으로 정해집니다.
- depth_multiplier: 깊이별 컨볼루션의 깊이 승수
    (해상도 승수라고도 합니다)
- include_top: 네트워크의 최상단에
    완전연결 레이어를 넣을지 여부.
- weights: `None` (임의의 초기값 설정),
    `'imagenet'` (ImageNet에 대한 선행 학습), 혹은
    가중치 파일을 불러올 경로 중 하나.
- input_tensor: 모델의 이미지 인풋으로 사용할 수 있는
    선택적 케라스 텐서
    (다시말해, `layers.Input()`의 아웃풋).
- pooling: 특성추출을 위한 선택적 풀링 모드로,
    `include_top`이 `False`일 경우 유효합니다.
    - `None`은 모델의 아웃풋이
        마지막 컨볼루션 레이어의
        4D 텐서 아웃풋임을 의미합니다.
    - `'avg'`는 글로벌 평균값 풀링이
        마지막 컨볼루션 레이어의
        아웃풋에 적용되어
        모델의 아웃풋이 
        2D 텐서가 됨을 의미합니다.
    - `'max'`는 글로벌 최대값 풀링이
        적용됨을 의미합니다.
- classes: 이미지를 분류하기 위한 선택적 클래스의 수로, 
    `include_top`이 `True`일 경우, 
    그리고 `weights` 인수가 따로 정해지지 않은 경우만 특정합니다.

### 반환값

케라스 `Model` 인스턴스.

### 오류처리

ValueError: `weights`에 유효하지 않은 인수를 넣은 경우,
    혹은 weights='imagenet'일 때 유효하지 않은 형태의 인풋이나 유효하지 않은 depth_multiplier, alpha, rows를 넣은 경우

### 참고

- [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)

### 라이센스

이 가중치는 [the Apache License](https://github.com/tensorflow/models/blob/master/LICENSE)에 따라 배포되었습니다.

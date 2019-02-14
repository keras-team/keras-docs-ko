# 어플리케이션

케라스 어플리케이션은 선행학습된 가중치와 함께 사용할 수 있도록 한 딥러닝 모델입니다.
이 모델로 예측, 특성추출, 파인튜닝을 할 수 있습니다.

가중치는 모델을 인스턴스화 할 때 자동으로 다운로드 됩니다. 이는 `~/.keras/models/`에 저장됩니다.

## 사용 가능한 모델

### ImageNet으로 학습한 가중치를 이용해 이미지 분류를 수행하는 모델:

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

위의 아키텍쳐 전부는 모든 백엔드(TensorFlow, Theano, and CNTK)와 호환가능하고, 인스턴스화 시 `~/.keras/keras.json`의 케라스 구성에 세팅된 이미지 데이터 포멧에 따라 모델이 만들어집니다. 예를 들어, 만약 `image_data_format=channels_last`로 세팅이 되어있다면, 이 리포지토리에서 불러온 모든 모델은 "Height-Width-Depth"의 텐서플로우 데이터 포맷 형식에 따라서 만들어집니다.

참고로:
- `Keras < 2.2.0`의 경우, Xception 모델은 `SeparableConvolution` 레이어를 이용하기 때문에 TensorFlow로만 사용가능합니다.
- `Keras < 2.1.5`의 경우, MobileNet 모델은 `DepthwiseConvolution` 레이어를 이용하기 때문에 TensorFlow로만 사용가능합니다.

-----

## 이미지 분류 모델의 사용법 예시

### ResNet50을 사용한 ImageNet 클래스 분류

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
# 결과를 튜플의 리스트(클래스, 설명, 확률)로 디코딩합니다
# (배치 내 각 샘플 당 하나의 리스트)
print('Predicted:', decode_predictions(preds, top=3)[0])
# 예측결과: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]
```

### VGG16을 사용한 특성추출

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

### VGG19를 사용한 임의의 중간 레이어로부터의 특성추출

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

### 새로운 클래스에 대한 InceptionV3 파인튜닝

```python
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

# 선행학습된 기준모델을 만듭니다
base_model = InceptionV3(weights='imagenet', include_top=False)

# 글로벌 공간 평균값 풀링 레이어를 더합니다
x = base_model.output
x = GlobalAveragePooling2D()(x)
# 완전 연결 레이어를 더합니다
x = Dense(1024, activation='relu')(x)
# 로지스틱 레이어를 더합니다 -- 200가지 클래스가 있다고 가정합니다
predictions = Dense(200, activation='softmax')(x)

# 다음은 학습할 모델입니다
model = Model(inputs=base_model.input, outputs=predictions)

# 첫째로: (난수로 초기값이 설정된) 가장 상위 레이어들만 학습시킵니다
# 다시 말해서 모든 InceptionV3 콘볼루션 레이어를 고정합니다
for layer in base_model.layers:
    layer.trainable = False

# 모델을 컴파일합니다 (*꼭* 레이어를 학습불가 상태로 세팅하고난 *후*에 컴파일합니다)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# 모델을 새로운 데이터에 대해 몇 세대간 학습합니다
model.fit_generator(...)

# 이 시점에서 상위 레이어들은 충분히 학습이 되었기에,
# inception V3의 콘볼루션 레이어에 대한 파인튜닝을 시작합니다 
# 가장 밑 N개의 레이어를 고정하고 나머지 상위 레이어를 학습시킵니다

# 레이어 이름과 레이어 인덱스를 시각화하여
# 얼마나 많은 레이어를 고정시켜야 하는지 확인합니다:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# 가장 상위 2개의 inception 블록을 학습하기로 고릅니다,
# 다시 말하면 첫 249개의 레이어는 고정시키고 나머지는 고정하지 않습니다:
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

# 이러한 수정사항이 효과를 내려면 모델을 다시 컴파일해야 합니다
# 낮은 학습 속도로 세팅된 SGD를 사용합니다
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# 다시 한 번 모델을 학습시킵니다
# (이번엔 상위 2개의 inception 블록을 상위의 밀집 레이어들과 함께 파인튜닝합니다)
model.fit_generator(...)
```


### 커스텀 인풋 텐서에 대한 InceptionV3 빌드 

```python
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input

# 다음의 inpu_tensor에 다른 케라스 모델이나 레이어의 아웃풋이 들어갈 수도 있습니다
input_tensor = Input(shape=(224, 224, 3))  # K.image_data_format() == 'channels_last'라고 가정합니다

model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=True)
```

-----

# 각 모델에 대한 설명서

| 모델 | 사이즈 | 상위-1 정확성 | 상위-5 정확성 | 매개변수 | 깊이 |
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

상위-1과 상위-5 정확성은 ImageNet의 검증 데이터셋에 대한 모델의 성능을 가리킵니다.

깊이란 네트워크의 토폴로지 깊이를 말합니다. 이는 활성화 레이어, 배치 정규화 레이어 등을 포함합니다.

-----


## Xception


```python
keras.applications.xception.Xception(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
```

ImageNet에 대해 가중치가 선행학습된 Xception V1 모델.

이 모델은 ImageNet에 대해서 0.790의 상위-1 검증 정확성을 가집니다.
또한 0.945의 상위-5 검증 정확성을 가집니다.

이 모델은 `'channels_last'` (height, width, channels) 데이터 포맷만 지원하는 것을 참고하십시오.

이 모델의 디폴트 인풋 사이즈는 299x299입니다.

### 인수

- include_top: 네트워크의 최상단에 완전 연결 레이어를 넣을지 여부.
- weights: `None` (임의의 초기값 설정) 혹은 `'imagenet'` (ImageNet에 대한 선행 학습) 중 하나.
- input_tensor: optional Keras tensor (i.e. output of `layers.Input()`) to use as image input for the model.
- input_shape: optional shape tuple, only to be specified
    if `include_top` is `False` (otherwise the input shape
    has to be `(299, 299, 3)`.
    It should have exactly 3 inputs channels,
    and width and height should be no smaller than 71.
    E.g. `(150, 150, 3)` would be one valid value.
- pooling: 특성추출을 위한 선택적 풀링 모드로,
    `include_top`이 `False`일 경우 유효합니다.
    - `None`은 모델의 아웃풋이
        마지박 콘볼루션 레이어의
        4D 텐서 아웃풋임을 의미합니다.
    - `'avg'`는 글로벌 평균값 풀링이
        마지막 콘볼루션 레이어의
        아웃풋에 적용되어
        모델의 아웃풋은 2D 텐서임을 의미합니다.
    - `'max'`는 글로벌 최대값 풀링이
        적용됨을 의미합니다.
- classes: 이미지를 분류하기 위한 선택적 클래스 수, 
    `include_top`이 `True`일 경우, 
    그리고 `weights` 인수가 특정되지 않은 경우만 특정합니다.

### 반환값

케라스 `Model` 인스턴스.

### 참고

- [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)

### 라이센스

이 가중치는 케라스 측에서 직접 학습시켰으며 MIT 라이센스로 배포되었습니다.


-----


## VGG16

```python
keras.applications.vgg16.VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
```

VGG16 model, with weights pre-trained on ImageNet.

This model can be built both with `'channels_first'` data format (channels, height, width) or `'channels_last'` data format (height, width, channels).

The default input size for this model is 224x224.

### Arguments

- include_top: whether to include the 3 fully-connected layers at the top of the network.
- weights: one of `None` (random initialization) or `'imagenet'` (pre-training on ImageNet).
- input_tensor: optional Keras tensor (i.e. output of `layers.Input()`) to use as image input for the model.
- input_shape: optional shape tuple, only to be specified
    if `include_top` is `False` (otherwise the input shape
    has to be `(224, 224, 3)` (with `'channels_last'` data format)
    or `(3, 224, 224)` (with `'channels_first'` data format).
    It should have exactly 3 inputs channels,
    and width and height should be no smaller than 32.
    E.g. `(200, 200, 3)` would be one valid value.
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

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556): please cite this paper if you use the VGG models in your work.

### License

These weights are ported from the ones [released by VGG at Oxford](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) under the [Creative Commons Attribution License](https://creativecommons.org/licenses/by/4.0/).

-----

## VGG19


```python
keras.applications.vgg19.VGG19(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
```


VGG19 model, with weights pre-trained on ImageNet.

This model can be built both with `'channels_first'` data format (channels, height, width) or `'channels_last'` data format (height, width, channels).

The default input size for this model is 224x224.

### Arguments

- include_top: whether to include the 3 fully-connected layers at the top of the network.
- weights: one of `None` (random initialization) or `'imagenet'` (pre-training on ImageNet).
- input_tensor: optional Keras tensor (i.e. output of `layers.Input()`) to use as image input for the model.
- input_shape: optional shape tuple, only to be specified
    if `include_top` is `False` (otherwise the input shape
    has to be `(224, 224, 3)` (with `'channels_last'` data format)
    or `(3, 224, 224)` (with `'channels_first'` data format).
    It should have exactly 3 inputs channels,
    and width and height should be no smaller than 32.
    E.g. `(200, 200, 3)` would be one valid value.
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

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

### License

These weights are ported from the ones [released by VGG at Oxford](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) under the [Creative Commons Attribution License](https://creativecommons.org/licenses/by/4.0/).

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


ResNet, ResNetV2, ResNeXt models, with weights pre-trained on ImageNet.

This model and can be built both with `'channels_first'` data format (channels, height, width) or `'channels_last'` data format (height, width, channels).

The default input size for this model is 224x224.


### Arguments

- include_top: whether to include the fully-connected layer at the top of the network.
- weights: one of `None` (random initialization) or `'imagenet'` (pre-training on ImageNet).
- input_tensor: optional Keras tensor (i.e. output of `layers.Input()`) to use as image input for the model.
- input_shape: optional shape tuple, only to be specified
    if `include_top` is `False` (otherwise the input shape
    has to be `(224, 224, 3)` (with `'channels_last'` data format)
    or `(3, 224, 224)` (with `'channels_first'` data format).
    It should have exactly 3 inputs channels,
    and width and height should be no smaller than 32.
    E.g. `(200, 200, 3)` would be one valid value.
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

- `ResNet`: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- `ResNetV2`: [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)
- `ResNeXt`: [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431)

### License

These weights are ported from the following:

- `ResNet`: [The original repository of Kaiming He](https://github.com/KaimingHe/deep-residual-networks) under the [MIT license](https://github.com/KaimingHe/deep-residual-networks/blob/master/LICENSE).
- `ResNetV2`: [Facebook](https://github.com/facebook/fb.resnet.torch) under the [BSD license](https://github.com/facebook/fb.resnet.torch/blob/master/LICENSE).
- `ResNeXt`: [Facebook AI Research](https://github.com/facebookresearch/ResNeXt) under the [BSD license](https://github.com/facebookresearch/ResNeXt/blob/master/LICENSE).

-----

## InceptionV3


```python
keras.applications.inception_v3.InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
```

Inception V3 model, with weights pre-trained on ImageNet.

This model and can be built both with `'channels_first'` data format (channels, height, width) or `'channels_last'` data format (height, width, channels).

The default input size for this model is 299x299.


### Arguments

- include_top: whether to include the fully-connected layer at the top of the network.
- weights: one of `None` (random initialization) or `'imagenet'` (pre-training on ImageNet).
- input_tensor: optional Keras tensor (i.e. output of `layers.Input()`) to use as image input for the model.
- input_shape: optional shape tuple, only to be specified
    if `include_top` is `False` (otherwise the input shape
    has to be `(299, 299, 3)` (with `'channels_last'` data format)
    or `(3, 299, 299)` (with `'channels_first'` data format).
    It should have exactly 3 inputs channels,
    and width and height should be no smaller than 75.
    E.g. `(150, 150, 3)` would be one valid value.
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

- [Rethinking the Inception Architecture for Computer Vision](http://arxiv.org/abs/1512.00567)

### License

These weights are released under [the Apache License](https://github.com/tensorflow/models/blob/master/LICENSE).

-----

## InceptionResNetV2


```python
keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
```

Inception-ResNet V2 model, with weights pre-trained on ImageNet.

This model and can be built both with `'channels_first'` data format (channels, height, width) or `'channels_last'` data format (height, width, channels).

The default input size for this model is 299x299.


### Arguments

- include_top: whether to include the fully-connected layer at the top of the network.
- weights: one of `None` (random initialization) or `'imagenet'` (pre-training on ImageNet).
- input_tensor: optional Keras tensor (i.e. output of `layers.Input()`) to use as image input for the model.
- input_shape: optional shape tuple, only to be specified
    if `include_top` is `False` (otherwise the input shape
    has to be `(299, 299, 3)` (with `'channels_last'` data format)
    or `(3, 299, 299)` (with `'channels_first'` data format).
    It should have exactly 3 inputs channels,
    and width and height should be no smaller than 75.
    E.g. `(150, 150, 3)` would be one valid value.
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

- [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261)

### License

These weights are released under [the Apache License](https://github.com/tensorflow/models/blob/master/LICENSE).

-----

## MobileNet


```python
keras.applications.mobilenet.MobileNet(input_shape=None, alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)
```

MobileNet model, with weights pre-trained on ImageNet.

Note that this model only supports the data format `'channels_last'` (height, width, channels).

The default input size for this model is 224x224.

### Arguments

- input_shape: optional shape tuple, only to be specified
    if `include_top` is `False` (otherwise the input shape
    has to be `(224, 224, 3)` (with `'channels_last'` data format)
    or `(3, 224, 224)` (with `'channels_first'` data format).
    It should have exactly 3 inputs channels,
    and width and height should be no smaller than 32.
    E.g. `(200, 200, 3)` would be one valid value.
- alpha: controls the width of the network.
    - If `alpha` < 1.0, proportionally decreases the number
        of filters in each layer.
    - If `alpha` > 1.0, proportionally increases the number
        of filters in each layer.
    - If `alpha` = 1, default number of filters from the paper
        are used at each layer.
- depth_multiplier: depth multiplier for depthwise convolution
    (also called the resolution multiplier)
- dropout: dropout rate
- include_top: whether to include the fully-connected
    layer at the top of the network.
- weights: `None` (random initialization) or
    `'imagenet'` (ImageNet weights)
- input_tensor: optional Keras tensor (i.e. output of
    `layers.Input()`)
    to use as image input for the model.
- pooling: Optional pooling mode for feature extraction
    when `include_top` is `False`.
    - `None` means that the output of the model
    will be the 4D tensor output of the
        last convolutional layer.
    - `'avg'` means that global average pooling
        will be applied to the output of the
        last convolutional layer, and thus
        the output of the model will be a
        2D tensor.
    - `'max'` means that global max pooling will
        be applied.
- classes: optional number of classes to classify images
    into, only to be specified if `include_top` is `True`, and
    if no `weights` argument is specified.

### Returns

A Keras `Model` instance.

### References

- [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf)

### License

These weights are released under [the Apache License](https://github.com/tensorflow/models/blob/master/LICENSE).

-----

## DenseNet


```python
keras.applications.densenet.DenseNet121(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
keras.applications.densenet.DenseNet169(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
keras.applications.densenet.DenseNet201(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
```

DenseNet models, with weights pre-trained on ImageNet.

This model and can be built both with `'channels_first'` data format (channels, height, width) or `'channels_last'` data format (height, width, channels).

The default input size for this model is 224x224.

### Arguments

- blocks: numbers of building blocks for the four dense layers.
- include_top: whether to include the fully-connected
    layer at the top of the network.
- weights: one of `None` (random initialization),
    'imagenet' (pre-training on ImageNet),
    or the path to the weights file to be loaded.
- input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
    to use as image input for the model.
- input_shape: optional shape tuple, only to be specified
    if `include_top` is False (otherwise the input shape
    has to be `(224, 224, 3)` (with `'channels_last'` data format)
    or `(3, 224, 224)` (with `'channels_first'` data format).
    It should have exactly 3 inputs channels,
    and width and height should be no smaller than 32.
    E.g. `(200, 200, 3)` would be one valid value.
- pooling: optional pooling mode for feature extraction
    when `include_top` is `False`.
    - `None` means that the output of the model will be
        the 4D tensor output of the
        last convolutional layer.
    - `avg` means that global average pooling
        will be applied to the output of the
        last convolutional layer, and thus
        the output of the model will be a 2D tensor.
    - `max` means that global max pooling will
        be applied.
- classes: optional number of classes to classify images
    into, only to be specified if `include_top` is True, and
    if no `weights` argument is specified.

### Returns

A Keras model instance.

### References

- [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) (CVPR 2017 Best Paper Award)

### License

These weights are released under [the BSD 3-clause License](https://github.com/liuzhuang13/DenseNet/blob/master/LICENSE).

-----

## NASNet


```python
keras.applications.nasnet.NASNetLarge(input_shape=None, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)
keras.applications.nasnet.NASNetMobile(input_shape=None, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)
```

Neural Architecture Search Network (NASNet) models, with weights pre-trained on ImageNet.

The default input size for the NASNetLarge model is 331x331 and for the
NASNetMobile model is 224x224.

### Arguments

- input_shape: optional shape tuple, only to be specified
    if `include_top` is `False` (otherwise the input shape
    has to be `(224, 224, 3)` (with `'channels_last'` data format)
    or `(3, 224, 224)` (with `'channels_first'` data format)
    for NASNetMobile or `(331, 331, 3)` (with `'channels_last'`
    data format) or `(3, 331, 331)` (with `'channels_first'`
    data format) for NASNetLarge.
    It should have exactly 3 inputs channels,
    and width and height should be no smaller than 32.
    E.g. `(200, 200, 3)` would be one valid value.
- include_top: whether to include the fully-connected
    layer at the top of the network.
- weights: `None` (random initialization) or
    `'imagenet'` (ImageNet weights)
- input_tensor: optional Keras tensor (i.e. output of
    `layers.Input()`)
    to use as image input for the model.
- pooling: Optional pooling mode for feature extraction
    when `include_top` is `False`.
    - `None` means that the output of the model
    will be the 4D tensor output of the
        last convolutional layer.
    - `'avg'` means that global average pooling
        will be applied to the output of the
        last convolutional layer, and thus
        the output of the model will be a
        2D tensor.
    - `'max'` means that global max pooling will
        be applied.
- classes: optional number of classes to classify images
    into, only to be specified if `include_top` is `True`, and
    if no `weights` argument is specified.

### Returns

A Keras `Model` instance.

### References

- [Learning Transferable Architectures for Scalable Image Recognition](https://arxiv.org/abs/1707.07012)

### License

These weights are released under [the Apache License](https://github.com/tensorflow/models/blob/master/LICENSE).

-----

## MobileNetV2


```python
keras.applications.mobilenet_v2.MobileNetV2(input_shape=None, alpha=1.0, depth_multiplier=1, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)
```

MobileNetV2 model, with weights pre-trained on ImageNet.

Note that this model only supports the data format `'channels_last'` (height, width, channels).

The default input size for this model is 224x224.

### Arguments

- input_shape: optional shape tuple, to be specified if you would
    like to use a model with an input img resolution that is not
    (224, 224, 3).
    It should have exactly 3 inputs channels (224, 224, 3).
    You can also omit this option if you would like
    to infer input_shape from an input_tensor.
    If you choose to include both input_tensor and input_shape then
    input_shape will be used if they match, if the shapes
    do not match then we will throw an error.
    E.g. `(160, 160, 3)` would be one valid value.
- alpha: controls the width of the network. This is known as the
    width multiplier in the MobileNetV2 paper.
    - If `alpha` < 1.0, proportionally decreases the number
        of filters in each layer.
    - If `alpha` > 1.0, proportionally increases the number
        of filters in each layer.
    - If `alpha` = 1, default number of filters from the paper
         are used at each layer.
- depth_multiplier: depth multiplier for depthwise convolution
      (also called the resolution multiplier)
- include_top: whether to include the fully-connected
      layer at the top of the network.
- weights: one of `None` (random initialization),
        'imagenet' (pre-training on ImageNet),
        or the path to the weights file to be loaded.
- input_tensor: optional Keras tensor (i.e. output of
      `layers.Input()`)
      to use as image input for the model.
- pooling: Optional pooling mode for feature extraction
    when `include_top` is `False`.
    - `None` means that the output of the model
    will be the 4D tensor output of the
        last convolutional layer.
    - `'avg'` means that global average pooling
        will be applied to the output of the
        last convolutional layer, and thus
        the output of the model will be a
        2D tensor.
    - `'max'` means that global max pooling will
        be applied. 
- classes: optional number of classes to classify images
      into, only to be specified if `include_top` is True, and
      if no `weights` argument is specified.

### Returns

A Keras model instance.

### Raises

ValueError: in case of invalid argument for `weights`,
    or invalid input shape or invalid depth_multiplier, alpha,
    rows when weights='imagenet'

### References

- [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)

### License

These weights are released under [the Apache License](https://github.com/tensorflow/models/blob/master/LICENSE).

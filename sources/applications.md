# 어플리케이션

케라스 어플리케이션은 선행학습된 가중치와 함께 사용할 수 있도록 한 딥러닝 모델입니다.
이 모델로 예측, 특성추출, 파인튜닝을 할 수 있습니다.

가중치는 모델을 인스턴스화 할 때 자동으로 다운로드됩니다. 이는 `~/.keras/models/`에 저장됩니다.

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

위의 아키텍쳐 전부는 모든 백엔드(TensorFlow, Theano, and CNTK)와 호환가능하고, 인스턴스화 시 `~/.keras/keras.json`의 케라스 구성에 세팅된 이미지 데이터 포멧에 따라 모델이 만들어집니다. 예를 들어, 만약 `image_data_format=channels_last`로 세팅이 되어있다면, 이 리포지토리에서 불러온 모든 모델은 "Height-Width-Depth"의 Tensorflow 데이터 포맷 형식에 따라서 만들어집니다.

참고로:
- `Keras < 2.2.0`의 경우, Xception 모델은 `SeparableConvolution` 층을 이용하기 때문에 TensorFlow로만 사용가능합니다.
- `Keras < 2.1.5`의 경우, MobileNet 모델은 `DepthwiseConvolution` 층을 이용하기 때문에 TensorFlow로만 사용가능합니다.

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

### VGG19를 사용한 임의의 중간 층으로부터의 특성추출

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

# 전역 공간 평균값 풀링층을 더합니다
x = base_model.output
x = GlobalAveragePooling2D()(x)
# 완전연결층을 더합니다
x = Dense(1024, activation='relu')(x)
# 로지스틱 층을 더합니다 -- 200가지 클래스가 있다고 가정합니다
predictions = Dense(200, activation='softmax')(x)

# 다음은 학습할 모델입니다
model = Model(inputs=base_model.input, outputs=predictions)

# 첫째로: (난수로 초기값이 설정된) 가장 상위 층들만 학습시킵니다
# 다시 말해서 모든 InceptionV3 합성곱 층을 고정합니다
for layer in base_model.layers:
    layer.trainable = False

# 모델을 컴파일합니다 (*꼭* 층을 학습불가 상태로 세팅하고난 *후*에 컴파일합니다)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# 모델을 새로운 데이터에 대해서 몇 에폭 학습합니다
model.fit_generator(...)

# 이 시점에서 상위 층들은 충분히 학습이 되었기에,
# inception V3의 합성곱 층에 대한 파인튜닝을 시작합니다 
# 가장 밑 N개의 층을 고정하고 나머지 상위 층을 학습시킵니다

# 층의 이름과 층의 인덱스를 시각화하여
# 얼마나 많은 층을 고정시켜야 하는지 확인합니다:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# 가장 상위 2개의 inception 블록을 학습하기로 고릅니다,
# 다시 말하면 첫 249개의 층은 고정시키고 나머지는 고정하지 않습니다:
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

# 이러한 수정사항이 효과를 내려면 모델을 다시 컴파일해야 합니다
# 낮은 학습 속도로 세팅된 SGD를 사용합니다
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# 다시 한 번 모델을 학습시킵니다
# (이번엔 상위 2개의 inception 블록을 상위의 완전연결층들과 함께 파인튜닝합니다)
model.fit_generator(...)
```


### 커스텀 입력 텐서에 대한 InceptionV3 빌드 

```python
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input

# 다음의 input_tensor에 다른 케라스 모델이나 층의 출력값이 들어갈 수도 있습니다
input_tensor = Input(shape=(224, 224, 3))  # K.image_data_format() == 'channels_last'라고 가정합니다

model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=True)
```

-----

# 각 모델에 대한 설명서

| 모델 | 사이즈 | 상위-1 정확도 | 상위-5 정확도 | 매개변수 | 깊이 |
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

상위-1과 상위-5 정확도은 ImageNet의 검증 데이터셋에 대한 모델의 성능을 가리킵니다.

깊이란 네트워크의 토폴로지 깊이를 말합니다. 이는 활성화 층, 배치 정규화 층 등을 포함합니다.

-----


## Xception


```python
keras.applications.xception.Xception(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
```

ImageNet에 대해 가중치가 선행학습된 Xception V1 모델.

이 모델은 ImageNet에 대해서 0.790의 상위-1 검증 정확도를 가집니다.
또한 0.945의 상위-5 검증 정확도를 가집니다.

이 모델은 `'channels_last'` (height, width, channels) 데이터 포맷만 지원하는 것을 참고하십시오.

이 모델의 기본 입력 크기는 299x299입니다.

### 인자

- include_top: 네트워크의 최상단에 완전연결층을 넣을지 여부.
- weights: `None` (임의의 초기값 설정) 혹은 `'imagenet'` (ImageNet에 대한 선행 학습) 중 하나.
- input_tensor: 모델의 이미지 입력으로 사용할 수 있는 선택적 케라스 텐서 (다시말해, `layers.Input()`의 출력값).
- input_shape: 선택적 형태 튜플로,
    `include_top`이 `False`일 경우만 특정하십시오.
    (그렇지 않다면 입력 형태가 `(229, 229 3)`이어야 합니다).
    입력 채널이 정확히 3개여야 하며
    넓이와 높이가 71 미만이어서는 안됩니다.
    예시. `(150, 150, 3)`은 유효한 값입니다.
- pooling: 특성추출을 위한 선택적 풀링 모드로,
    `include_top`이 `False`일 경우 유효합니다.
    - `None`은 모델의 출력값이
        마지막 합성곱 층의
        4D 텐서 출력임을 의미합니다.
    - `'avg'`는 전역 평균값 풀링이
        마지막 합성곱 층의
        출력에 적용되어
        모델의 출력값은 2D 텐서임을 의미합니다.
    - `'max'`는 전역 최대값 풀링이
        적용됨을 의미합니다.
- classes: 이미지를 분류하기 위한 선택적 클래스 수, 
    `include_top`이 `True`일 경우, 
    그리고 `weights` 인자가 특정되지 않은 경우만 특정합니다.

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

ImageNet에 대해 가중치가 선행학습된 VGG16 모델

이 모델에는 `'channels_first'` 데이터 포맷(채널, 높이, 넓이)과 `'channels_last'` 데이터 포맷(높이, 넓이, 채널) 둘 모두 사용할 수 있습니다.

이 모델의 기본 입력 크기는 224x224 입니다.

### 인자

- include_top: 네트워크의 최상단에 3개의 완전연결층을 넣을지 여부.
- weights: `None` (임의의 초기값 설정) 혹은 `'imagenet'` (ImageNet에 대한 선행 학습) 중 하나.
- input_tensor: 모델의 이미지 입력으로 사용할 수 있는 선택적 케라스 텐서 (다시말해, `layers.Input()`의 출력값).
- input_shape: 선택적 형태 튜플로,
    `include_top`이 `False`일 경우만 특정하십시오.
    (그렇지 않다면 인풋 형태가 `(224, 224, 3)`이고 `'channels_last'` 데이터 포맷을 취하거나
    혹은 입력 형태가 `(3, 224, 224)`이고 `'channels_first'` 데이터 포맷을 취해야 합니다).
    입력 채널이 정확히 3개여야 하며
    넓이와 높이가 32 미만이어서는 안됩니다.
    예시. `(200, 200, 3)`은 유효한 값입니다.
- pooling: 특성추출을 위한 선택적 풀링 모드로,
    `include_top`이 `False`일 경우 유효합니다.
    - `None`은 모델의 출력값이
        마지막 합성곱 층의
        4D 텐서 출력임을 의미합니다.
    - `'avg'`는 전역 평균값 풀링이
        마지막 합성곱 층의
        출력값에 적용되어
        모델의 출력값이 2D 텐서가 됨을 의미합니다.
    - `'max'`는 전역 최대값 풀링이
        적용됨을 의미합니다.
- classes: 이미지를 분류하기 위한 선택적 클래스의 수로, 
    `include_top`이 `True`일 경우, 
    그리고 `weights` 인자가 따로 정해지지 않은 경우만 특정합니다.

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

이 모델의 기본 입력 크기는 224x224입니다.

### 인자

- include_top: 네트워크의 최상단에 3개의 완전연결층을 넣을지 여부.
- weights: `None` (임의의 초기값 설정) 혹은 `'imagenet'` (ImageNet에 대한 선행 학습) 중 하나.
- input_tensor: 모델의 이미지 입력으로 사용할 수 있는 선택적 케라스 텐서 (다시말해, `layers.Input()`의 출력값).
- input_shape: 선택적 형태 튜플로,
    `include_top`이 `False`일 경우만 특정하십시오.
    (그렇지 않다면 입력 형태가 `(224, 224, 3)`이고 `'channels_last'` 데이터 포맷을 취하거나
    혹은 입력 형태가 `(3, 224, 224)`이고 `'channels_first'` 데이터 포맷을 취해야 합니다).
    입력 채널이 정확히 3개여야 하며
    넓이와 높이가 32 미만이어서는 안됩니다.
    예시. `(200, 200, 3)`은 유효한 값입니다.
- pooling: 특성추출을 위한 선택적 풀링 모드로,
    `include_top`이 `False`일 경우 유효합니다.
    - `None`은 모델의 출력값이
        마지막 합성곱 층의
        4D 텐서 출력임을 의미합니다.
    - `'avg'`는 전역 평균값 풀링이
        마지막 합성곱 층의
        출력값에 적용되어
        모델의 출력값이 2D 텐서가 됨을 의미합니다.
    - `'max'`는 전역 최대값 풀링이
        적용됨을 의미합니다.
- classes: 이미지를 분류하기 위한 선택적 클래스 수로, 
    `include_top`이 `True`일 경우, 
    그리고 `weights` 인자가 따로 정해지지 않은 경우만 특정합니다.

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

이 모델의 기본 입력 크기는 224x224입니다.


### 인자

- include_top: 네트워크의 최상단에 완전연결층을 넣을지 여부.
- weights: `None` (임의의 초기값 설정) 혹은 `'imagenet'` (ImageNet에 대한 선행 학습) 중 하나.
- input_tensor: 모델의 이미지 입력으로 사용할 수 있는 선택적 케라스 텐서 (다시말해, `layers.Input()`의 출력값).
- input_shape: 선택적 형태 튜플로,
    `include_top`이 `False`일 경우만 특정하십시오.
    (그렇지 않다면 입력 형태가 `(224, 224, 3)`이고 `'channels_last'` 데이터 포맷을 취하거나
    혹은 입력 형태가 `(3, 224, 224)`이고 `'channels_first'` 데이터 포맷을 취해야 합니다).
    입력 채널이 정확히 3개여야 하며
    넓이와 높이가 32 미만이어서는 안됩니다.
    예시. `(200, 200, 3)`은 유효한 값입니다.
- pooling: 특성추출을 위한 선택적 풀링 모드로,
    `include_top`이 `False`일 경우 유효합니다.
    - `None`은 모델의 출력값이
        마지막 합성곱 층의
        4D 텐서 출력임을 의미합니다.
    - `'avg'`는 전역 평균값 풀링이
        마지막 합성곱 층의
        출력값에 적용되어
        모델의 출력값이 2D 텐서가 됨을 의미합니다.
    - `'max'`는 전역 최대값 풀링이
        적용됨을 의미합니다.
- classes: 이미지를 분류하기 위한 선택적 클래스의 수로, 
    `include_top`이 `True`일 경우, 
    그리고 `weights` 인자가 따로 정해지지 않은 경우만 특정합니다.

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

이 모델의 기본 입력 크기는 299x299 입니다.


### 인자

- include_top: 네트워크의 최상단에 완전연결층을 넣을지 여부.
- weights: `None` (임의의 초기값 설정) 혹은 `'imagenet'` (ImageNet에 대한 선행 학습) 중 하나.
- input_tensor: 모델의 이미지 입력으로 사용할 수 있는 선택적 케라스 텐서 (다시말해, `layers.Input()`의 출력값).
- input_shape: 선택적 형태 튜플로,
    `include_top`이 `False`일 경우만 특정하십시오.
    (그렇지 않다면 입력 형태가 `(299, 299, 3)`이고 `'channels_last'` 데이터 포맷을 취하거나
    혹은 입력 형태가 `(3, 299, 299)`이고 `'channels_first'` 데이터 포맷을 취해야 합니다).
    입력 채널이 정확히 3개여야 하며
    넓이와 높이가 75 미만이어서는 안됩니다.
    예시. `(150, 150, 3)`은 유효한 값입니다.
- pooling: 특성추출을 위한 선택적 풀링 모드로,
    `include_top`이 `False`일 경우 유효합니다.
    - `None`은 모델의 출력값이
        마지막 합성곱 층의
        4D 텐서 출력임을 의미합니다.
    - `'avg'`는 전역 평균값 풀링이
        마지막 합성곱 층의
        출력값에 적용되어
        모델의 출력값이 2D 텐서가 됨을 의미합니다.
    - `'max'`는 전역 최대값 풀링이
        적용됨을 의미합니다.
- classes: 이미지를 분류하기 위한 선택적 클래스의 수로, 
    `include_top`이 `True`일 경우, 
    그리고 `weights` 인자가 따로 정해지지 않은 경우만 특정합니다.

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

이 모델의 기본 입력 크기는 299x299 입니다.


### 인자

- include_top: 네트워크의 최상단에 완전연결층을 넣을지 여부.
- weights: `None` (임의의 초기값 설정) 혹은 `'imagenet'` (ImageNet에 대한 선행 학습) 중 하나.
- input_tensor: 모델의 이미지 입력으로 사용할 수 있는 선택적 케라스 텐서 (다시말해, `layers.Input()`의 출력값).
- input_shape: 선택적 형태 튜플로,
    `include_top`이 `False`일 경우만 특정하십시오.
    (그렇지 않다면 입력 형태가 `(299, 299, 3)`이고 `'channels_last'` 데이터 포맷을 취하거나
    혹은 입력 형태가 `(3, 299, 299)`이고 `'channels_first'` 데이터 포맷을 취해야 합니다).
    입력 채널이 정확히 3개여야 하며
    넓이와 높이가 75 미만이어서는 안됩니다.
    예시. `(150, 150, 3)`은 유효한 값입니다.
- pooling: 특성추출을 위한 선택적 풀링 모드로,
    `include_top`이 `False`일 경우 유효합니다.
    - `None`은 모델의 출력값이
        마지막 합성곱 층의
        4D 텐서 출력임을 의미합니다.
    - `'avg'`는 전역 평균값 풀링이
        마지막 합성곱 층의
        출력값에 적용되어
        모델의 출력값이 2D 텐서가 됨을 의미합니다.
    - `'max'`는 전역 최대값 풀링이
        적용됨을 의미합니다.
- classes: 이미지를 분류하기 위한 선택적 클래스의 수로, 
    `include_top`이 `True`일 경우, 
    그리고 `weights` 인자가 따로 정해지지 않은 경우만 특정합니다.

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

이 모델의 기본 입력 크기는 224x224 입니다.

### 인자

- input_shape: 선택적 형태 튜플로,
    `include_top`이 `False`일 경우만 특정하십시오.
    (그렇지 않다면 입력 형태가 `(224, 224, 3)`이고 `'channels_last'` 데이터 포맷을 취하거나
    혹은 입력 형태가 `(3, 224, 224)`이고 `'channels_first'` 데이터 포맷을 취해야 합니다).
    입력 채널이 정확히 3개여야 하며
    넓이와 높이가 32 미만이어서는 안됩니다.
    예시. `(200, 200, 3)`은 유효한 값입니다.
- alpha: 네트워크의 넓이를 조정합니다.
    - `alpha` < 1.0 인 경우, 그에 비례해서 각 층의
        필터 숫자를 감소시킵니다.
    - `alpha` > 1.0 인 경우, 그에 비례해서 각 층의
        필터 숫자를 증가시킵니다.
    - `alpha` = 1 인 경우, 각 층의 필터의 수가
        참고 논문에 따른 기본값으로 정해집니다.
- depth_multiplier: 깊이별 합성곱의 깊이 승수
    (해상도 승수라고도 합니다)
- dropout: 드롭아웃 속도
- include_top: 네트워크의 최상단에
    완전연결층을 넣을지 여부.
- weights: `None` (임의의 초기값 설정) 혹은
    `'imagenet'` (ImageNet에 대한 선행 학습)
- input_tensor: 모델의 이미지 입력으로 사용할 수 있는
    선택적 케라스 텐서
    (다시말해, `layers.Input()`의 출력값).
- pooling: 특성추출을 위한 선택적 풀링 모드로,
    `include_top`이 `False`일 경우 유효합니다.
    - `None`은 모델의 출력값이
        마지막 합성곱 층의
        4D 텐서 출력임을 의미합니다.
    - `'avg'`는 전역 평균값 풀링이
        마지막 합성곱 층의
        출력값에 적용되어
        모델의 출력값이 
        2D 텐서가 됨을 의미합니다.
    - `'max'`는 전역 최대값 풀링이
        적용됨을 의미합니다.
- classes: 이미지를 분류하기 위한 선택적 클래스의 수로, 
    `include_top`이 `True`일 경우, 
    그리고 `weights` 인자가 따로 정해지지 않은 경우만 특정합니다.

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

이 모델의 기본 입력 크기는 224x224 입니다.

### 인자

- blocks: 4개의 밀집 층의 빌딩 블록의 수.
- include_top: 네트워크의 최상단에
    완전연결층을 넣을지 여부.
- weights: `None` (임의의 초기값 설정),
    `'imagenet'` (ImageNet에 대한 선행 학습), 혹은
    가중치 파일을 불러올 경로 중 하나.
- input_tensor: 모델의 이미지 입력으로 사용할 수 있는 선택적 케라스 텐서
    (다시말해, `layers.Input()`의 출력값).
- input_shape: 선택적 형태 튜플로,
    `include_top`이 `False`일 경우만 특정하십시오.
    (그렇지 않다면 입력 형태가 `(224, 224, 3)`이고 `'channels_last'` 데이터 포맷을 취하거나
    혹은 입력 형태가 `(3, 224, 224)`이고 `'channels_first'` 데이터 포맷을 취해야 합니다).
    입력 채널이 정확히 3개여야 하며
    넓이와 높이가 32 미만이어서는 안됩니다.
    예시. `(200, 200, 3)`은 유효한 값입니다.
- pooling: 특성추출을 위한 선택적 풀링 모드로,
    `include_top`이 `False`일 경우 유효합니다.
    - `None`은 모델의 출력값이
        마지막 합성곱 층의
        4D 텐서 출력임을 의미합니다.
    - `'avg'`는 전역 평균값 풀링이
        마지막 합성곱 층의
        출력값에 적용되어
        모델의 출력값이 2D 텐서가 됨을 의미합니다.
    - `'max'`는 전역 최대값 풀링이
        적용됨을 의미합니다.
- classes: 이미지를 분류하기 위한 선택적 클래스의 수로, 
    `include_top`이 `True`일 경우, 
    그리고 `weights` 인자가 따로 정해지지 않은 경우만 특정합니다.

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

NASNetLarge 모델의 기본 입력 크기는 331x331 이고
NASNetMobile 모델은 224x224 입니다.

### 인자

- input_shape: 선택적 형태 튜플로,
    `include_top`이 `False`일 경우만 특정하십시오.
    (그렇지 않다면 NASNetMobile의 경우 입력 형태가 `(224, 224, 3)`이고 
    `'channels_last'` 데이터 포맷을 취하거나 혹은 태가 `(3, 224, 224)`이고 
    `'channels_first'` 데이터 포맷을 취해야 하며, NASNetLarge의 경우  
    입력이 `(331, 331, 3)`에 `'channels_last'` 데이터 포맷, 
    혹은 입력이 `(3, 331, 331)`에 `'channels_first'` 데이터 포맷이어야 합니다).
    입력 채널이 정확히 3개여야 하며
    넓이와 높이가 32 미만이어서는 안됩니다.
    예시. `(200, 200, 3)`은 유효한 값입니다.
- include_top: 네트워크의 최상단에
    완전연결층을 넣을지 여부.
- weights: `None` (임의의 초기값 설정) 혹은
    `'imagenet'` (ImageNet에 대한 선행 학습)
- input_tensor: 모델의 이미지 입력으로 사용할 수 있는
    선택적 케라스 텐서
    (다시말해, `layers.Input()`의 출력값).
- pooling: 특성추출을 위한 선택적 풀링 모드로,
    `include_top`이 `False`일 경우 유효합니다.
    - `None`은 모델의 출력값이
        마지막 합성곱 층의
        4D 텐서 출력임을 의미합니다.
    - `'avg'`는 전역 평균값 풀링이
        마지막 합성곱 층의
        출력값에 적용되어
        모델의 출력값이 
        2D 텐서가 됨을 의미합니다.
    - `'max'`는 전역 최대값 풀링이
        적용됨을 의미합니다.
- classes: 이미지를 분류하기 위한 선택적 클래스의 수로, 
    `include_top`이 `True`일 경우, 
    그리고 `weights` 인자가 따로 정해지지 않은 경우만 특정합니다.

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

이 모델의 기본 입력 크기는 224x224 입니다.

### 인자

- input_shape: 선택적 형태 튜플로,
    (224, 224, 3)이 아닌 입력 이미지 해상도를 사용할 경우에만
    특정하십시오.
    입력 채널이 정확히 3개(224, 224, 3)이어야 합니다.
    input_tensor에서 input_shape을 추론하고 싶다면
    이 옵션을 생략해도 됩니다.
    input_tensor와 input_shape 둘 모두를 활용하는 경우
    둘의 형태가 매치된다면 input_shape이 사용되고,
    그렇지 않다면 에러를 알립니다.
    예시. `(160, 160, 3)`은 유효한 값입니다.
- alpha: 네트워크의 넓이를 조정합니다.
    이는 MobileNetV2 논문에서 넓이 승수로 기술됩니다.
    - `alpha` < 1.0 인 경우, 그에 비례해서 각 층의
        필터 숫자를 감소시킵니다.
    - `alpha` > 1.0 인 경우, 그에 비례해서 각 층의
        필터 숫자를 증가시킵니다.
    - `alpha` = 1 인 경우, 각 층의 필터의 수가
        참고 논문에 따른 기본값으로 정해집니다.
- depth_multiplier: 깊이별 합성곱의 깊이 승수
    (해상도 승수라고도 합니다)
- include_top: 네트워크의 최상단에
    완전연결층을 넣을지 여부.
- weights: `None` (임의의 초기값 설정),
    `'imagenet'` (ImageNet에 대한 선행 학습), 혹은
    가중치 파일을 불러올 경로 중 하나.
- input_tensor: 모델의 이미지 입력으로 사용할 수 있는
    선택적 케라스 텐서
    (다시말해, `layers.Input()`의 출력값).
- pooling: 특성추출을 위한 선택적 풀링 모드로,
    `include_top`이 `False`일 경우 유효합니다.
    - `None`은 모델의 출력값이
        마지막 합성곱 층의
        4D 텐서 출력임을 의미합니다.
    - `'avg'`는 전역 평균값 풀링이
        마지막 합성곱 층의
        출력값에 적용되어
        모델의 출력값이 
        2D 텐서가 됨을 의미합니다.
    - `'max'`는 전역 최대값 풀링이
        적용됨을 의미합니다.
- classes: 이미지를 분류하기 위한 선택적 클래스의 수로, 
    `include_top`이 `True`일 경우, 
    그리고 `weights` 인자가 따로 정해지지 않은 경우만 특정합니다.

### 반환값

케라스 `Model` 인스턴스.

### 오류처리

ValueError: `weights`에 유효하지 않은 인자를 넣은 경우,
    혹은 weights='imagenet'일 때 유효하지 않은 형태의 입력이나 유효하지 않은 depth_multiplier, alpha, rows를 넣은 경우

### 참고

- [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)

### 라이센스

이 가중치는 [the Apache License](https://github.com/tensorflow/models/blob/master/LICENSE)에 따라 배포되었습니다.

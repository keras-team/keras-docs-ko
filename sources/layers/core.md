<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L796)</span>

### Dense

```python
keras.layers.Dense(units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

일반적인 밀집 연결 신경망 층<sub>densely-connected NN layer</sub>.

`Dense`는 `output = activation(dot(input, kernel) + bias)` 을 실행합니다.
여기서 `activation`은 `activation` 인자로 전달되는 원소 단위<sub>element-wise</sub> 활성화 함수이고,
`kernel`은 층이 만들어낸 가중치 행렬<sub>weight matrix</sub>이며,
`bias`는 층이 만들어낸 편향<sub>bias</sub> 벡터이며 `use_bias`가 `True`인 경우만 적용 가능합니다.

참고: 층의 입력 랭크가 2보다 클 경우, `kernel`과의 내적<sub>dot product</sub>을 하기 전에 층의 차원을 1차원으로 만들어주어야 합니다.

__예시__

```python
# 시퀀스 모델의 첫 번째 층:
model = Sequential()
model.add(Dense(32, input_shape=(16,)))
# 모델은 (*, 16) 형태의 배열을 입력으로 받고
# (*, 32) 형태의 배열을 출력합니다

# 첫 번째 층 이후에는,
# 입력의 크기를 특정하지 않아도 됩니다:
model.add(Dense(32))
```

__인자__

- __units__: 양의 정수, 출력 공간의 차원.
- __activation__: 사용할 활성화 함수
  ([활성화 함수](../activations.md)를 참조하십시오).
  따로 설정하지 않으면, 활성화 함수가 적용되지 않습니다.
  (즉, "linear" 활성화 함수: `a(x) = x`).
- __use_bias__: `Bool`. 층이 편향 벡터를 사용하는지 여부.
- __kernel_initializer__: `kernel` 가중치 행렬의 초기화<sub>initalizer</sub> 함수
  ([초기값 함수](../initializers.md)를 참조하십시오).
- __bias_initializer__: 편향 벡터의 초기값 함수
  ([초기값 함수](../initializers.md)를 참조하십시오).
- __kernel_regularizer__: `kernel` 가중치 행렬에 적용되는 규제<sub>regularizer</sub> 함수
  ([규제 함수](../regularizers.md)를 참조하십시오).
- __bias_regularizer__: 편향 벡터에 적용되는 규제 함수
  ([규제 함수](../regularizers.md)를 참조하십시오).
- __activity_regularizer__: 층의 출력값(층의 “활성화” 값)에 적용되는 규제 함수
  ([규제 함수](../regularizers.md)를 참조하십시오).
- __kernel_constraint__: `kernel` 가중치 행렬에 적용되는 제약<sub>constraints</sub> 함수
  ([제약 함수](../constraints.md)을 참조하십시오).
- __bias_constraint__: 편향 벡터에 적용하는 제약 함수
  ([제약 함수](../constraints.md)을 참조하십시오).

__입력 형태__

`(batch_size, ..., input_dim)` 형태의 nD 텐서.
가장 일반적인 경우는`(batch_size, input_dim)` 형태의 2D 입력입니다.

__출력 형태__

`(batch_size, ..., units)` 형태의 nD 텐서.
예를 들어, `(batch_size, input_dim)` 형태의 2D 입력에 대해서
출력은 `(batch_size, units)`의 형태를 가집니다.
    

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L277)</span>

### Activation

```python
keras.layers.Activation(activation)
```

출력값에 활성화 함수를 적용합니다.

__인자__

- __activation__:  Theano나 Tensorflow 또는 다른 곳에서 사용하는 활성화 함수의 이름
  ([활성화 함수](../activations.md)를 참조하십시오)

__입력 형태__

임의의 형태입니다. 모델의 첫 번째 층으로 Activation층을
사용할 때 키워드 인자 `input_shape`을 함께 사용하여 형태를 지정합니다. 
`input_shape`는 정수 튜플<sub>tuple of integers</sub>로 샘플 축을 포함하지 않습니다.

__출력 형태__

입력 형태와 동일하다.
    

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L81)</span>

### Dropout

```python
keras.layers.Dropout(rate, noise_shape=None, seed=None)
```

입력에 드롭아웃을 적용합니다.

드롭아웃은 학습 과정 중 업데이트를 할 때  `rate` 의 비율에 따라 입력 유닛을 무작위로 0으로 설정합니다. 이는 과적합을 방지하는데 도움이 됩니다.

__인자__

- __rate__: 0과 1사이 `float`. 드롭시킬 입력 유닛의 비율.
- __noise_shape__: 입력과 곱하게 되는 이진 드롭아웃 마스크<sub>mask</sub>의
  형태를 나타내는 1D 정수 텐서.
  예를 들어, 입력이 `(batch_size, timesteps, features)`의
  형태를 가지는 경우, 드롭아웃 마스크를
  모든 시간 단계에 대해서 동일하게 적용하고 싶다면
  `noise_shape=(batch_size, 1, features)`를 사용하면 됩니다.
- __seed__: 난수<sub>random</sub> 시드로 사용할 파이썬 정수형.

__참조__

- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](
  http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L462)</span>

### Flatten

```python
keras.layers.Flatten(data_format=None)
```

입력을 1차원으로 펼칩니다. 배치 크기에는 영향을 미치치 않습니다.

__인자__

- __data_format__: `string`,
  입력 차원의 순서를 지정할수 있으며 `channels_last` (기본값) 혹은 `channels_first` 중 하나를 지정합니다.              
  이 인자는 모델을 하나의 데이터 형식에서 다른 형식으로
  바꿀 때 가중치 순서를 보존할 때 이용됩니다.
  `channels_last`는 `(batch, ..., channels)` 형태의 입력에 대응되고, `channels_first`는 `(batch, channels, ...)` 형태의 입력에 대응합니다.
  기본 값은 `~/.keras/keras.json`에 위치한
  케라스 구성 파일의 `image_data_format` 값으로 지정됩니다.
  따로 설정하지 않으면 `"channels_last"`으로 설정이 됩니다.

__예시__

```python
model = Sequential()
model.add(Conv2D(64, (3, 3),
                 input_shape=(3, 32, 32), padding='same',))
# 현재: model.output_shape == (None, 64, 32, 32)

model.add(Flatten())
# 현재: model.output_shape == (None, 65536)
```

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/engine/input_layer.py#L114)</span>

### Input

```python
keras.engine.input_layer.Input()
```

`Input()`으로 케라스 텐서를 인스턴스화<sub>Instantiate</sub>합니다.

케라스 텐서는 기저 백엔드(Theano, TensorFlow 혹은 CNTK)에서
비롯된 텐서 객체로, 우리는 모델의 입력과 출력을 아는 것만으로도
케라스 모델을 만들 수 있는 특정한 속성을 증강<sub>augment</sub>시킬수 있습니다.

예를 들어 a, b와 c가 케라스 텐서라고 하면
`model = Model(input=[a, b], output=c)`을 실행할 수 있습니다.

추가된 케라스 속성은 다음과 같습니다.
`_keras_shape`: 케라스의 형태 유추를 통해 전파되는 정수 튜플.
`_keras_history`: 텐서에 적용된 마지막 층.
해당 층에서 전체 층 그래프를 여러번 검색할 수 있습니다.

__인자__

- __shape__: 배치 크기를 제외한 정수 튜플 형태.
  예를 들어 `shape=(32,)`는 예상되는 입력이
  32차원 벡터의 배치라는 것을 나타냅니다.
- __batch_shape__: 배치 크기를 포함한 정수 튜플 형태.
  예를 들어 `batch_shape=(10, 32)`는 예상되는 입력이
  10개의 32차원 벡터로 이루어진 배치라는 것을 나타냅니다.
  `batch_shape=(None, 32)`는 임의의 수의 32차원 벡터로 이루어진
  배치를 나타냅니다.
- __name__: 지정할 수 있는 층의 문자열 이름.
  모델 내에서 이름은 고유해야 하며 동일한 이름을 다시 사용하면 안됩니다.
  따로 지정하지 않을 경우, 자동으로 생성됩니다.
- __dtype__: `string`, 입력으로 들어가야 할 데이터 자료형
  (`float32`, `float64`, `int32`...)
- __sparse__: `Bool`, 만들어 낼 플레이스홀더<sub>placeholder</sub>가 희소<sub>sparse</sub>한지
  여부를 확인합니다.
- __tensor__: `Input` 층으로 래핑<sub>wrap</sub>할 수 있는 기존 텐서.
  따로 설정하는 경우, 층은 플레이스홀더 텐서를 만들어 내지 않습니다.

__반환값__

  텐서.

__예시__

```python
# 다음은 케라스의 로지스틱 회귀입니다
x = Input(shape=(32,))
y = Dense(16, activation='softmax')(x)
model = Model(x, y)
```

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L311)</span>

### Reshape

```python
keras.layers.Reshape(target_shape)

```

출력을 특정 형태로 변형시킵니다.



__인자__

- __target_shape__: 목푯값 형태. 정수 튜플.
  배치 축은 포함하지 않습니다. 

__입력 형태__

임의의 형태입니다. 모델의 첫 번째 층으로 Reshape층을
사용할 때 키워드 인자 `input_shape`을 함께 사용하여 형태를 지정합니다. 
`input_shape`는 정수 튜플로 샘플 축을 포함하지 않습니다.

__출력 형태__

`(batch_size,) + target_shape`

__예시__

```python
# 시퀀스 모델의 첫 번째 층입니다
model = Sequential()
model.add(Reshape((3, 4), input_shape=(12,)))
# 현재: model.output_shape == (None, 3, 4)
# 참고: `None`은 배치 차원입니다

# 시퀀스 모델의 중간 층입니다
model.add(Reshape((6, 2)))
# 현재: model.output_shape == (None, 6, 2)

# `-1`을 차원으로 사용해서 형태 유추를 지원합니다
model.add(Reshape((-1, 2, 2)))
# 현재: model.output_shape == (None, 3, 2, 2)

```

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L408)</span>

### Permute

```python
keras.layers.Permute(dims)

```

주어진 패턴에 따라서 입력의 차원을 치환합니다.

예를 들어 순환 신경망<sub>Recurrnent Neural Network</sub>과 convnet을 함께 연결하는 경우에 유용합니다.

__예시__

```python
model = Sequential()
model.add(Permute((2, 1), input_shape=(10, 64)))
# 현재: model.output_shape == (None, 64, 10)
# 참고: `None`은 배치 차원입니다

```

__인자__

- __dims__: 정수 튜플. 치환 패턴, 샘플 차원을 포함하지 않습니다.
  인덱스은 1에서 시작합니다.
  예를 들어, `(2, 1)`은 입력의 첫 번째와 두 번째 차원을
  치환합니다.

__입력 형태__

임의의 형태입니다. 모델의 첫 번째 층으로 Permute층을
사용할 때 키워드 인자 `input_shape`을 함께 사용하여 형태를 지정합니다. 
`input_shape`는 정수 튜플로 샘플 축을 포함하지 않습니다.

__출력 형태__

입력 형태와 동일하나, 특정된 패턴에 따라 차원의 순서가 재조정됩니다.
    

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L524)</span>

### RepeatVector

```python
keras.layers.RepeatVector(n)

```

입력을 n회 반복합니다.

__예시__

```python
model = Sequential()
model.add(Dense(32, input_dim=32))
# 현재: model.output_shape == (None, 32)
# 참고: `None`은 배치 차원입니다

model.add(RepeatVector(3))
# 현재: model.output_shape == (None, 3, 32)

```

__인자__

- __n__: 정수, 반복 인자.

__입력 형태__

`(num_samples, features)` 형태의 2D 텐서.

__출력 형태__

`(num_samples, n, features)` 형태의 3D 텐서.
    

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L566)</span>

### Lambda

```python
keras.layers.Lambda(function, output_shape=None, mask=None, arguments=None)

```

사용자가 생성한 임의의 표현을 `Layer` 객체로 래핑<sub>wrap</sub>합니다.

__예시__

```python
# x -> x^2 층를 더합니다
model.add(Lambda(lambda x: x ** 2))

```

```python
# 입력의 음성 부분과 양성 부분의
# 연결을 반환하는
# 층을 더합니다.

def antirectifier(x):
    x -= K.mean(x, axis=1, keepdims=True)
    x = K.l2_normalize(x, axis=1)
    pos = K.relu(x)
    neg = K.relu(-x)
    return K.concatenate([pos, neg], axis=1)

def antirectifier_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 2  # 2D 탠서만 유효합니다.
    shape[-1] *= 2
    return tuple(shape)

model.add(Lambda(antirectifier,
                 output_shape=antirectifier_output_shape))

```

```python
# 두 입력 텐서의 아다마르 곱과
# 합을 반환하는 층을 더합니다.

def hadamard_product_sum(tensors):
    out1 = tensors[0] * tensors[1]
    out2 = K.sum(out1, axis=-1)
    return [out1, out2]

def hadamard_product_sum_output_shape(input_shapes):
    shape1 = list(input_shapes[0])
    shape2 = list(input_shapes[1])
    assert shape1 == shape2  # 그렇지 않으면 아다마르 곱이 성립하지 않습니다
    return [tuple(shape1), tuple(shape2[:-1])]

x1 = Dense(32)(input_1)
x2 = Dense(32)(input_2)
layer = Lambda(hadamard_product_sum, hadamard_product_sum_output_shape)
x_hadamard, x_sum = layer([x1, x2])

```

__인자__

- __function__: 평가할 함수.
  첫 번째 인자로 텐서 혹은 텐서 리스트를 입력을 받습니다.
- __output_shape__: 함수를 통해 예측되는 출력 형태.
  Theano를 사용하는 경우에만 유효합니다. 튜플 혹은 함수가 될 수 있습니다.
  튜플인 경우, 첫 번째 차원만 차원을 지정합니다.
  샘플 차원이 입력 차원과 동일하다고 가정하거나(`output_shape = (input_shape[0], ) + output_shape`)
   혹은, 입력이 `None`이고 샘플 차원 또한 `None`이라고 가정합니다. ( `output_shape = (None, ) + output_shape`)
  함수인 경우, 전체 형태를 입력 함수의 형태로 지정 합니다. (`output_shape = f(input_shape)`
- __mask__: 없음(마스크를 하지 않는다) 또는 임베딩용 입력 마스크 텐서. 
- __arguments__: 함수에 전달하는 키워드 인자의 딕셔너리<sub>Dictionary</sub>

__입력 형태__

임의의 형태입니다. 모델의 첫 번째 층으로 Lambda층을
사용할 때 키워드 인자 `input_shape`을 함께 사용하여 형태를 지정합니다. 
`input_shape`는 정수 튜플로 샘플 축을 포함하지 않습니다.

__출력 형태__

`output_shape`  인자의 형태를 따릅니다.
혹은 TensorFlow나 CNTK를 사용하는 경우 자동으로 형태가 지정됩니다.



------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L940)</span>

### ActivityRegularization

```python
keras.layers.ActivityRegularization(l1=0.0, l2=0.0)

```

입력 엑티비티에 대한 비용 함수<sub>cost function</sub>를 업데이트를 시행할 때 사용하는 층입니다.

__인자__

- __l1__: L1 규제화 인수 (양의 부동소수점).
- __l2__: L2 규제화 인수 (양의 부동소수점).

__입력 형태__

임의의 형태입니다. 모델의 첫 번째 층으로 ActivityRegularization층을
사용할 때 키워드 인자 `input_shape`을 함께 사용하여 형태를 지정합니다. 
`input_shape`는 정수 튜플로 샘플 축을 포함하지 않습니다.

__출력 형태__

입력 형태와 동일.
    

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L28)</span>

### Masking

```python
keras.layers.Masking(mask_value=0.0)

```

시간 단계를 건너 뛰기 위해 마스크 값을 이용해 시퀀스<sub>sequence</sub>를 마스킹합니다.

주어진 샘플 시간 단계<sub>time step</sub>의 모든 특성<sub>feature</sub>이 `mask_value`와 동일하고 마스킹을 지원한다면 모든 하위 층에서 해당되는 샘플 시간 단계를 마스킹합니다.

아직 하위 층이 마스킹을 지원하지 않는데 입력 마스킹을
받아들이는 경우 예외가 발생합니다.

__예시__

LSTM 층에 전달할 `(samples, timesteps, features)`의
형태를 가진 NumPy 데이터 배열 `x`를 생각해 봅시다.
샘플 시간 단계에 대한 특성이 없는 경우, 시간 단계 #3의 샘플 #0과 시간 단계 #5의 샘플 #2를 마스킹하고 싶다면, 다음을 실행하면 됩니다

- `x[0, 3, :] = 0.`, 그리고 `x[2, 5, :] = 0.`으로 설정합니다
- LSTM층 전에 `mask_value=0.`의 `Masking` 층을 삽입합니다

```python
model = Sequential()
model.add(Masking(mask_value=0., input_shape=(timesteps, features)))
model.add(LSTM(32))

```

__인자__

​    __mask_value__: None 혹은 건너뛸 마스크 값

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L141)</span>

### SpatialDropout1D

```python
keras.layers.SpatialDropout1D(rate)

```

드롭아웃의 공간적 1D 버전.

이 버전은 드롭아웃과 같은 함수를 수행하지만, 개별적 원소 대신
1D 특성 맵 전체를 드롭시킵니다.  초기 합성곱<sub>convolution</sub>층에서는                                  특성 맵 내 인접한 프레임<sub>frame</sub>이 강한 상관관계를 보이는 경우가 많습니다.                  보통의 드롭아웃은 활성화 함수들을 정규화 시키지 못하고 그저 학습 속도를                 감소시킵니다.
이러한 경우, SpatialDropout1D은 일반적인 드롭아웃 대신 특성 맵 사이의 독립성을 유지하는데 도움을 줍니다.

__인자__

​    __rate__: 0과 1사이 부동소수점. 드롭시킬 입력 유닛의 비율.

__입력 형태__

​    `(samples, timesteps, channels)`형태의 3D 텐서.

__출력 형태__

​    입력 형태와 동일.

__참조__

- [Efficient Object Localization Using Convolutional Networks](
  https://arxiv.org/abs/1411.4280)

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L178)</span>

### SpatialDropout2D

```python
keras.layers.SpatialDropout2D(rate, data_format=None)

```

드롭아웃의 공간적 2D 버전.

이 버전은 드롭아웃과 같은 함수를 수행하지만, 개별적 원소 대신
1D 특성 맵 전체를 드롭시킵니다.  초기 합성곱층에서는                                  특성 맵 내 인접한 픽셀<sub>Pixels</sub>이 강한 상관관계를 보이는 경우가 많습니다.                       보통의 드롭아웃은 활성화 함수들을 정규화 시키지 못하고 그저 학습 속도를                 감소시킵니다.
이러한 경우, SpatialDropout2D은 일반적인 드롭아웃 대신 특성 맵 사이의 독립성을 유지하는데 도움을 줍니다.

__인수__

- __rate__: 0과 1사이 부동소수점. 드롭시킬 입력 유닛의 비율.
- __data_format__: 입력 차원의 순서를 지정할수 있으며 `channels_last` (기본값)     혹은 `channels_first` 중 하나를 지정합니다.                                              `channels_first` 모드에서는, 채널 차원(깊이)의 인덱스 1이며,
  `channels_last` 모드에서는, 채널 차원(깊이)의 인덱스 3입니다.                                                            기본값은 `~/.keras/keras.json`에 위치한
  케라스 구성 파일의 `image_data_format` 값으로 지정됩니다.
  따로 설정하지 않으면, `"channels_last"`으로 설정이 됩니다.

__입력 형태__

data_format=`channels_first`인 경우
`(samples, channels, rows, cols)` 형태의 4D 텐서.
data_format=`channels_last`인 경우
`(samples, rows, cols, channels)` 형태의 4D 텐서.

__출력 형태__

입력 형태와 동일.

__참조__

- [Efficient Object Localization Using Convolutional Networks](
  https://arxiv.org/abs/1411.4280)

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L228)</span>

### SpatialDropout3D

```python
keras.layers.SpatialDropout3D(rate, data_format=None)

```

드롭아웃의 공간적 3D 버전.

이 버전은 드롭아웃과 같은 함수를 수행하지만, 개별적 원소 대신
1D 특성 맵 전체를 드롭시킵니다.  초기 합성곱층에서는                                  특성 맵 내 인접한 복셀<sub>Voxels</sub>이 강한 상관관계를 보이는 경우가 많습니다.                       보통의 드롭아웃은 활성화 함수들을 정규화 시키지 못하고 그저 학습 속도를                 감소시킵니다.
이러한 경우, SpatialDropout3D은 일반적인 드롭아웃 대신 특성 맵 사이의 독립성을 유지하는데 도움을 줍니다.

__인자__

- __rate__: 0과 1사이 부동소수점. 드롭시킬 입력 유닛의 비율.
- __data_format__:입력 차원의 순서를 지정할수 있으며 `channels_last` (기본값)       혹은 `channels_first` 중 하나를 지정합니다. 
  `channels_first` 모드에서는, 채널 차원(깊이)의 인덱스 1이며,       `channels_last` 모드에서는, 채널 차원(깊이)의인덱스 4이다.
  기본값은 `~/.keras/keras.json`에 위치한
  케라스 구성 파일의 `image_data_format` 값으로 지정됩니다.
  따로 설정하지 않으면, `"channels_last"`으로 설정이 됩니다.

__입력 형태__

data_format=`channels_first`인 경우
`(samples, channels, dim1, dim2, dim3)` 형태의 5D 텐서.
data_format=`channels_last`인 경우
`(samples, dim1, dim2, dim3, channels)` 형태의 5D 텐서.

__출력 형태__

입력 형태와 동일.

__참조__

- [Efficient Object Localization Using Convolutional Networks](
  https://arxiv.org/abs/1411.4280)

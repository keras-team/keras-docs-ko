<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L241)</span>
### Conv1D

```python
keras.layers.Conv1D(filters, kernel_size, strides=1, padding='valid', data_format='channels_last', dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

1D 컨볼루션 레이어(예. 시간적 컨볼루션).

이 레이어는 단일 공간적 (혹은 시간적) 차원에서
인풋 레이어와 컨볼루션되는 컨볼루션 커넬을 생성하여
하나의 아웃풋 텐서를 만들어냅니다.
`use_bias`가 참인 경우, 편향 벡터<sub>bias vector</sub>를 생성해 아웃풋에 더합니다.
끝으로, `activation`이 `None`이 아닌 경우
이는 아웃풋에도 적용됩니다.

이 레이어를 모델의 첫 레이어로 사용하는 경우
`input_shape` 인수를 제공해야합니다 (정수 튜플 혹은 `None`,
배치 축은 포함하지 않습니다). 예. `data_format="channels_last"`이고
10 시간 단계에 단계 당 128 특성을 지닌 시계열 시퀀스는
`input_shape=(10, 128)`로 표현되고, 혹은 길이가 정해지지 않은 시간 단계에 단계 당 128 특성을 지닌
시퀀스는 `(None, 128)`로 표현됩니다.

__인수__

- __filters__: 정수, 아웃풋 공간의 차원
    (다시 말해, 컨볼루션의 아웃풋 필터의 수).
- __kernel_size__: 정수 혹은 단일 정수의 튜플/리스트.
    1D 컨볼루션 윈도우의 길이를 특정합니다.
- __strides__: 정수 혹은 단일 정수의 튜플/리스트.
    컨볼루션의 보폭 길이를 특정합니다.
    `stride` 값 != 1이면, `dilation_rate` 값 != 1의 어떤 경우와도
    호환이 불가능합니다.
- __padding__: `"valid"`, `"causal"` 혹은 `"same"` (대소문자 무시).
    `"valid"`는 "패딩 없음"을 의미합니다.
    `"same"`은 아웃풋이 원래 인풋과 동일한
    길이를 갖도록 인풋을 패딩합니다.
    `"causal"`은 인과적 (팽창된) 컨볼루션을 실행합니다.
    예. `output[t]`가 `input[t + 1:]`에 종속되지 않습니다.
    제로 패딩을 사용해 아웃풋이
    원래의 인풋과 같은 길이를 갖도록 합니다.
    모델이 시간적 순서를 위반하면 안되는
    시간적 데이터를 모델링 할 때 유용합니다.
    [WaveNet: A Generative Model for Raw Audio, section 2.1](
    https://arxiv.org/abs/1609.03499).
- __data_format__: 문자열,
    `"channels_last"` (디폴트 값) 혹은 "channels_first"` 중 하나.
    인풋 내 차원의 순서를 나타냅니다.
    `"channels_last"`는 `(batch, steps, channels)` 형태의
    인풋에 호응하고
    (시간적 데이터의 케라스 디폴트 형식)
    `"channels_first"`는 `(batch, channels, steps)` 형태의
    인풋에 호응합니다.
- __dilation_rate__: 정수 혹은 단일 정수의 튜플/리스트.
    팽창 컨볼루션의 팽창 속도를 특정합니다.
    현재는 `dilation_rate` 값 != 1이면
    `strides` 값 != 1의 어떤 경우와도 호환이 불가합니다.
- __activation__: 사용할 활성화 함수
    ([활성화](../activations.md)를 참조하십시오).
    따로 정하지 않으면, 활성화가 적용되지 않습니다
    (다시 말해, "선형적" 활성화: `a(x) = x`).
- __use_bias__: 불리언, 레이어가 편향 벡터를 사용하는지 여부.
- __kernel_initializer__: `kernel` 가중치 행렬의 초기값 설정기
    ([초기값 설정기](../initializers.md)를 참조하십시오).
- __bias_initializer__: 편향 벡터의 초기값 설정기
    ([초기값 설정기](../initializers.md)를 참조하십시오).
- __kernel_regularizer__: `kernel` 가중치 행렬에
    적용되는 정규화 함수
    ([정규화](../regularizers.md)를 참조하십시오).
- __bias_regularizer__: 편향 벡터에 적용되는 정규화 함수
    ([정규화](../regularizers.md)를 참조하십시오).
- __activity_regularizer__: 레이어의 아웃풋(레이어의 “활성화”)에
    적용되는 정규화 함수
    ([정규화](../regularizers.md)를 참조하십시오).
- __kernel_constraint__: 커널 행렬에 적용되는 제약 함수
    ([제약](../constraints.md)을 참조하십시오).
- __bias_constraint__: 편향 벡터에 적용하는 제약 함수
    ([제약](../constraints.md)을 참조하십시오).

__인풋 형태__

`(batch, steps, channels)` 형태의 3D 텐서.

__아웃풋 형태__

`(batch, new_steps, filters)` 형태의 3D 텐서.
패딩이나 보폭으로 인해 `steps` 값이 변했을 수 있습니다.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L368)</span>
### Conv2D

```python
keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

2D 컨볼루션 레이어 (예. 이미지에 대한 공간적 컨볼루션).

이 레이어는 레이어의 인풋과 컨볼루션되어
아웃풋 텐서를 만들어내는 컨볼루션 커널을
생성합니다. `use_bias`가 참일 경우,
편향 벡터를 만들어 아웃풋에 더합니다. 끝으로,
`activation`이 `None`이 아닐 경우, 이는 아웃풋에도 적용됩니다.

이 레이어를 모델의 첫 레이어로 사용하는 경우
`input_shape` 키워드 인수를 제공하십시오
(정수 튜플, 배치 축은 포함하지 않습니다).
예. `data_format="channels_last"`인 128x128 RGB
사진의 경우 `input_shape=(128, 128, 3)`이 됩니다.

__인수__

- __filters__: 정수, 아웃풋 공간의 차원
    (다시 말해, 컨볼루션의 아웃풋 필터의 수).
- __kernel_size__: 정수 혹은 2개 정수의 튜플/리스트. 2D 컨볼루션
    윈도우의 높이와 넓이를 특정합니다.
    정수 하나로 모든 공간적 차원에 대해서 동일한
    값을 특정할 수도 있습니다.
- __strides__: 정수 혹은 2개 정수의 튜플/리스트.
    높이와 넓이에 따라 컨볼루션의
    보폭 길이를 특정합니다.
    정수 하나로 모든 공간적 차원에 대해서 동일한
    값을 특정할 수도 있습니다.
    `stride` 값 != 1이면, `dilation_rate` 값 != 1의 어떤 경우와도
    호환이 불가능합니다.
- __padding__: `"valid"` 혹은 `"same"` (대소문자 무시).
    `"same"`은 다음의 설명과 같이 백엔드에 따라 약간의 기복이
    있다는 것을 참고하십시오:
    (https://github.com/keras-team/keras/pull/9473#issuecomment-372166860)
- __data_format__: 문자열,
    `"channels_last"` 혹은 "channels_first"` 중 하나.
    인풋 내 차원의 순서를 나타냅니다.
    `"channels_last"`는 `(batch, height, width, channels)`
    형태의 인풋에 호응하고
    `"channels_first"`는 `(batch, channels, height, width)` 형태의
    인풋에 호응합니다.
    디폴트 값은 `~/.keras/keras.json`에 위치한
    케라스 구성 파일의 `image_data_format` 값으로 지정됩니다.
    따로 설정하지 않으면, 이는 `"channels_last"`가 됩니다.
- __dilation_rate__: 정수 혹은 2개 정수의 튜플/리스트.
    팽창 컨볼루션의 팽창 속도를 특정합니다.
    정수 하나로 모든 공간적 차원에 대해서 동일한
    값을 특정할 수도 있습니다.
    현재는 `dilation_rate` 값 != 1이면
    `strides` 값 != 1의 어떤 경우와도 호환이 불가합니다.
- __activation__: 사용할 활성화 함수
    ([활성화](../activations.md)를 참조하십시오).
    따로 정하지 않으면, 활성화가 적용되지 않습니다
    (다시 말해, "선형적" 활성화: `a(x) = x`).
- __use_bias__: 불리언, 레이어가 편향 벡터를 사용하는지 여부.
- __kernel_initializer__: `kernel` 가중치 행렬의 초기값 설정기
    ([초기값 설정기](../initializers.md)를 참조하십시오).
- __bias_initializer__: 편향 벡터의 초기값 설정기
    ([초기값 설정기](../initializers.md)를 참조하십시오).
- __kernel_regularizer__: `kernel` 가중치 행렬에
    적용되는 정규화 함수
    ([정규화](../regularizers.md)를 참조하십시오).
- __bias_regularizer__: 편향 벡터에 적용되는 정규화 함수
    ([정규화](../regularizers.md)를 참조하십시오).
- __activity_regularizer__: 레이어의 아웃풋(레이어의 “활성화”)에
    적용되는 정규화 함수
    ([정규화](../regularizers.md)를 참조하십시오).
- __kernel_constraint__: 커널 행렬에 적용되는 제약 함수
    ([제약](../constraints.md)을 참조하십시오).
- __bias_constraint__: 편향 벡터에 적용하는 제약 함수
    ([제약](../constraints.md)을 참조하십시오).

__인풋 형태__

`data_format`이 `"channels_first"`이면
`(batch, channels, rows, cols)`
형태의 4D 텐서.
`data_format`이 `"channels_last"`이면
`(batch, rows, cols, channels)`
형태의 4D 텐서.

__아웃풋 형태__

`data_format`이 `"channels_first"`이면
`(batch, filters, new_rows, new_cols)`
형태의 4D 텐서.
`data_format`이 `"channels_last"`이면
`(batch, new_rows, new_cols, filters)`
형태의 4D 텐서.
패딩으로 인해 `rows`와 `cols` 값이 변했을 수 있습니다.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L1421)</span>
### SeparableConv1D

```python
keras.layers.SeparableConv1D(filters, kernel_size, strides=1, padding='valid', data_format='channels_last', dilation_rate=1, depth_multiplier=1, activation=None, use_bias=True, depthwise_initializer='glorot_uniform', pointwise_initializer='glorot_uniform', bias_initializer='zeros', depthwise_regularizer=None, pointwise_regularizer=None, bias_regularizer=None, activity_regularizer=None, depthwise_constraint=None, pointwise_constraint=None, bias_constraint=None)
```

깊이별 분리형 1D 컨볼루션.

분리형 컨볼루션은 우선
깊이별 공간 컨볼루션을 수행하고
(이는 각 인풋 채널에 따로 적용됩니다),
이어서 그 결과로 발생한 아웃풋 채널을 포인트별 컨볼루션을
사용해 섞어줍니다. `depth_multiplier` 인수는 깊이별 단계에서
인풋 채널 당 몇 개의 아웃풋 채널을 생성할지 조정합니다.

직관적으로, 분리형 컨볼루션은
컨볼루션 커널을 두 개의 작은 커널로 분해하는 기법으로 보거나,
Inception 블록의 극단적 버전으로 이해할 수 있습니다.

__인수__

- __filters__: 정수, 아웃풋 공간의 차원
    (다시 말해, 컨볼루션의 아웃풋 필터의 수).
- __kernel_size__: 정수 혹은 단일 정수의 튜플/리스트.
    1D 컨볼루션 윈도우의 길이를 특정합니다.
- __strides__: 정수 혹은 단일 정수의 튜플/리스트.
    컨볼루션의 보폭 길이를 특정합니다.
    `stride` 값 != 1이면, `dilation_rate` 값 != 1의 어떤 경우와도
    호환이 불가능합니다.
- __padding__: `"valid"` 혹은 `"same"` (대소문자 무시).
- __data_format__: 문자열,
    `"channels_last"` 혹은 "channels_first"` 중 하나.
    인풋 내 차원의 순서를 나타냅니다.
    `"channels_last"`는 `(batch, steps, channels)`
    형태의 인풋에 호응하고
    `"channels_first"`는 `(batch, channels, steps)`
    형태의 인풋에 호응합니다.
- __dilation_rate__: 정수 혹은 단일 정수의 튜플/리스트.
    팽창 컨볼루션의 팽창 속도를 특정합니다.
    현재는 `dilation_rate` 값 != 1이면
    `strides` 값 != 1의 어떤 경우와도 호환이 불가합니다.
- __depth_multiplier__: 각 인풋 채널에 대한 깊이별 컨볼루션
    아웃풋 채널의 개수.
    깊이별 컨볼루션 아웃풋 채널의 총 개수는
    `filters_in * depth_multiplier`가 됩니다.
- __activation__: 사용할 활성화 함수
    ([활성화](../activations.md)를 참조하십시오).
    따로 정하지 않으면, 활성화가 적용되지 않습니다
    (다시 말해, "선형적" 활성화: `a(x) = x`).
- __use_bias__: 불리언, 레이어가 편향 벡터를 사용하는지 여부.
- __depthwise_initializer__: 깊이별 커널 행렬의 초기값 설정기.
    ([초기값 설정기](../initializers.md)를 참고하십시오).
- __pointwise_initializer__: 포인트별 커널 행렬의 초기값 설정기.
    ([초기값 설정기](../initializers.md)를 참고하십시오).
- __bias_initializer__: 편향 벡터의 초기값 설정기
    ([초기값 설정기](../initializers.md)를 참조하십시오).
- __depthwise_regularizer__: 깊이별 커널 행렬에 적용되는
    정규화 함수
    ([정규화](../regularizers.md)를 참조하십시오).
- __pointwise_regularizer__: 포인트별 커널 행렬에 적용되는
    정규화 함수
    ([정규화](../regularizers.md)를 참조하십시오).
- __bias_regularizer__: 편향 벡터에 적용되는 정규화 함수
    ([정규화](../regularizers.md)를 참조하십시오).
- __activity_regularizer__: 레이어의 아웃풋(레이어의 “활성화”)에
    적용되는 정규화 함수
    ([정규화](../regularizers.md)를 참조하십시오).
- __depthwise_constraint__: 깊이별 커널 행렬에 적용되는
    제약 함수
    ([제약](../constraints.md)를 참조하십시오).
- __pointwise_constraint__: 포인트별 커널 행렬에 적용되는
    제약 함수
    ([제약](../constraints.md)를 참조하십시오).
- __bias_constraint__: 편향 벡터에 적용하는 제약 함수
    ([제약](../constraints.md)을 참조하십시오).

__인풋 형태__

`data_format`이 `"channels_first"`이면
`(batch, channels, steps)`
형태의 3D 텐서.
`data_format`이 `"channels_last"`이면
`(batch, steps, channels)`
형태의 3D 텐서.

__아웃풋 형태__

`data_format`이 `"channels_first"`이면
`(batch, filters, new_steps)`
형태의 3D 텐서.
`data_format`이 `"channels_last"`이면
`(batch, new_steps, filters)`
형태의 3D 텐서.
패딩이나 보폭으로 인해 `new_steps` 값이 변했을 수 있습니다.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L1553)</span>
### SeparableConv2D

```python
keras.layers.SeparableConv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), depth_multiplier=1, activation=None, use_bias=True, depthwise_initializer='glorot_uniform', pointwise_initializer='glorot_uniform', bias_initializer='zeros', depthwise_regularizer=None, pointwise_regularizer=None, bias_regularizer=None, activity_regularizer=None, depthwise_constraint=None, pointwise_constraint=None, bias_constraint=None)
```

깊이별 분리형 2D 컨볼루션.

분리형 컨볼루션은 우선
깊이별 공간 컨볼루션을 수행하고
(이는 각 인풋 채널에 따로 적용됩니다),
이어서 그 결과로 발생한 아웃풋 채널을 포인트별 컨볼루션을
사용해 섞어줍니다. `depth_multiplier` 인수는 깊이별 단계에서
인풋 채널 당 몇 개의 아웃풋 채널을 생성할지 조정합니다.

직관적으로, 분리형 컨볼루션은
컨볼루션 커널을 두 개의 작은 커널로 분해하는 기법으로 보거나,
Inception 블록의 극단적 버전으로 이해할 수 있습니다.

__인수__

- __filters__: 정수, 아웃풋 공간의 차원
    (다시 말해, 컨볼루션의 아웃풋 필터의 수).
- __kernel_size__: 정수 혹은 2개 정수의 튜플/리스트. 2D 컨볼루션
    윈도우의 높이와 넓이를 특정합니다.
    정수 하나로 모든 공간적 차원에 대해서 동일한
    값을 특정할 수도 있습니다.
- __strides__: 정수 혹은 2개 정수의 튜플/리스트.
    높이와 넓이에 따라 컨볼루션의
    보폭 길이를 특정합니다.
    정수 하나로 모든 공간적 차원에 대해서 동일한
    값을 특정할 수도 있습니다.
    `stride` 값 != 1이면, `dilation_rate` 값 != 1의 어떤 경우와도
    호환이 불가능합니다.
- __padding__: `"valid"` 혹은 `"same"` (대소문자 무시).
- __data_format__: 문자열,
    `"channels_last"` 혹은 "channels_first"` 중 하나.
    인풋 내 차원의 순서를 나타냅니다.
    `"channels_last"`는 `(batch, height, width, channels)`
    형태의 인풋에 호응하고
    `"channels_first"`는 `(batch, channels, height, width)` 형태의
    인풋에 호응합니다.
    디폴트 값은 `~/.keras/keras.json`에 위치한
    케라스 구성 파일의 `image_data_format` 값으로 지정됩니다.
    따로 설정하지 않으면, 이는 `"channels_last"`가 됩니다.
- __dilation_rate__: 정수 혹은 2개 정수의 튜플/리스트.
    팽창 컨볼루션의 팽창 속도를 특정합니다.
    현재는 `dilation_rate` 값 != 1이면
    `strides` 값 != 1의 어떤 경우와도 호환이 불가합니다.
- __depth_multiplier__: 각 인풋 채널에 대한 깊이별 컨볼루션
    아웃풋 채널의 개수.
    깊이별 컨볼루션 아웃풋 채널의 총 개수는
    `filters_in * depth_multiplier`가 됩니다.
- __activation__: 사용할 활성화 함수
    ([활성화](../activations.md)를 참조하십시오).
    따로 정하지 않으면, 활성화가 적용되지 않습니다
    (다시 말해, "선형적" 활성화: `a(x) = x`).
- __use_bias__: 불리언, 레이어가 편향 벡터를 사용하는지 여부.
- __depthwise_initializer__: 깊이별 커널 행렬의 초기값 설정기.
    ([초기값 설정기](../initializers.md)를 참고하십시오).
- __pointwise_initializer__: 포인트별 커널 행렬의 초기값 설정기.
    ([초기값 설정기](../initializers.md)를 참고하십시오).
- __bias_initializer__: 편향 벡터의 초기값 설정기
    ([초기값 설정기](../initializers.md)를 참조하십시오).
- __depthwise_regularizer__: 깊이별 커널 행렬에 적용되는
    정규화 함수
    ([정규화](../regularizers.md)를 참조하십시오).
- __pointwise_regularizer__: 포인트별 커널 행렬에 적용되는
    정규화 함수
    ([정규화](../regularizers.md)를 참조하십시오).
- __bias_regularizer__: 편향 벡터에 적용되는 정규화 함수
    ([정규화](../regularizers.md)를 참조하십시오).
- __activity_regularizer__: 레이어의 아웃풋(레이어의 “활성화”)에
    적용되는 정규화 함수
    ([정규화](../regularizers.md)를 참조하십시오).
- __depthwise_constraint__: 깊이별 커널 행렬에 적용되는
    제약 함수
    ([제약](../constraints.md)를 참조하십시오).
- __pointwise_constraint__: Constraint function applied to
    the pointwise kernel matrix
    (see [constraints](../constraints.md)).
- __bias_constraint__: 편향 벡터에 적용하는 제약 함수
    ([제약](../constraints.md)을 참조하십시오).

__인풋 형태__

`data_format`이 `"channels_first"`이면
`(batch, channels, rows, cols)`
형태의 4D 텐서.
`data_format`이 `"channels_last"`이면
`(batch, rows, cols, channels)`
형태의 4D 텐서.

__아웃풋 형태__

`data_format`이 `"channels_first"`이면
`(batch, filters, new_rows, new_cols)`
형태의 4D 텐서.
`data_format`이 `"channels_last"`이면
`(batch, new_rows, new_cols, filters)`
형태의 4D 텐서.
패딩으로 인해 `rows`와 `cols` 값이 변했을 수 있습니다.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L1694)</span>
### DepthwiseConv2D

```python
keras.layers.DepthwiseConv2D(kernel_size, strides=(1, 1), padding='valid', depth_multiplier=1, data_format=None, activation=None, use_bias=True, depthwise_initializer='glorot_uniform', bias_initializer='zeros', depthwise_regularizer=None, bias_regularizer=None, activity_regularizer=None, depthwise_constraint=None, bias_constraint=None)
```

깊이별 분리형 2D 컨볼루션.

깊이별 분리형 컨볼루션은 깊이별 공간 컨볼루션을 수행하는
첫 단계만으로 구성됩니다 (이는 각 인풋 채널에 따로 적용됩니다).
이어서 그 결과로 발생한 아웃풋 채널을 포인트별 컨볼루션을
사용해 섞어줍니다. `depth_multiplier` 인수는 깊이별 단계에서
인풋 채널 당 몇 개의 아웃풋 채널을 생성할지 조정합니다.

__인수__

- __kernel_size__: 정수 혹은 2개 정수의 튜플/리스트. 2D 컨볼루션
    윈도우의 높이와 넓이를 특정합니다.
    정수 하나로 모든 공간적 차원에 대해서 동일한
    값을 특정할 수도 있습니다.
- __strides__: 정수 혹은 2개 정수의 튜플/리스트.
    높이와 넓이에 따라 컨볼루션의
    보폭 길이를 특정합니다.
    정수 하나로 모든 공간적 차원에 대해서 동일한
    값을 특정할 수도 있습니다.
    `stride` 값 != 1이면, `dilation_rate` 값 != 1의 어떤 경우와도
    호환이 불가능합니다.
- __padding__: `"valid"` 혹은 `"same"` (대소문자 무시).
- __depth_multiplier__: 각 인풋 채널에 대한 깊이별 컨볼루션
    아웃풋 채널의 개수.
    깊이별 컨볼루션 아웃풋 채널의 총 개수는
    `filters_in * depth_multiplier`가 됩니다.
- __data_format__: 문자열,
    `"channels_last"` 혹은 "channels_first"` 중 하나.
    인풋 내 차원의 순서를 나타냅니다.
    `"channels_last"`는 `(batch, height, width, channels)`
    형태의 인풋에 호응하고
    `"channels_first"`는 `(batch, channels, height, width)` 형태의
    인풋에 호응합니다.
    디폴트 값은 `~/.keras/keras.json`에 위치한
    케라스 구성 파일의 `image_data_format` 값으로 지정됩니다.
    따로 설정하지 않으면, 이는 `"channels_last"`가 됩니다.
- __activation__: 사용할 활성화 함수
    ([활성화](../activations.md)를 참조하십시오).
    따로 정하지 않으면, 활성화가 적용되지 않습니다
    (다시 말해, "선형적" 활성화: `a(x) = x`).
- __use_bias__: 불리언, 레이어가 편향 벡터를 사용하는지 여부.
- __depthwise_initializer__: 깊이별 커널 행렬의 초기값 설정기.
    ([초기값 설정기](../initializers.md)를 참고하십시오).
- __bias_initializer__: 편향 벡터의 초기값 설정기
    ([초기값 설정기](../initializers.md)를 참조하십시오).
- __depthwise_regularizer__: 깊이별 커널 행렬에 적용되는
    정규화 함수
    ([정규화](../regularizers.md)를 참조하십시오).
- __bias_regularizer__: 편향 벡터에 적용되는 정규화 함수
    ([정규화](../regularizers.md)를 참조하십시오).
- __activity_regularizer__: 레이어의 아웃풋(레이어의 “활성화”)에
    적용되는 정규화 함수
    ([정규화](../regularizers.md)를 참조하십시오).
- __depthwise_constraint__: 깊이별 커널 행렬에 적용되는
    제약 함수
    ([제약](../constraints.md)를 참조하십시오).
- __bias_constraint__: 편향 벡터에 적용하는 제약 함수
    ([제약](../constraints.md)을 참조하십시오).

__인풋 형태__

`data_format`이 `"channels_first"`이면
`(batch, channels, rows, cols)`
형태의 4D 텐서.
`data_format`이 `"channels_last"`이면
`(batch, rows, cols, channels)`
형태의 4D 텐서.

__아웃풋 형태__

`data_format`이 `"channels_first"`이면
`(batch, filters, new_rows, new_cols)`
형태의 4D 텐서.
`data_format`이 `"channels_last"`이면
`(batch, new_rows, new_cols, filters)`
형태의 4D 텐서.
패딩으로 인해 `rows`와 `cols` 값이 변했을 수 있습니다.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L628)</span>
### Conv2DTranspose

```python
keras.layers.Conv2DTranspose(filters, kernel_size, strides=(1, 1), padding='valid', output_padding=None, data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

전치 컨볼루션 레이어 (디컨볼루션으로 불리기도 합니다).

전치 컨볼루션은 일반적으로 다음의 상황에서 필요합니다:
보통의 컨볼루션과 반대방향으로 가는 변형을 사용하고
싶은 경우, 다시 말해 해당 컨볼루션과 호환되는 연결 패턴을
유지하면서 아웃풋의 형태를 가진 어떤 것을
인풋의 형태로 바꾸고 싶은 경우
유용합니다.

이 레이어를 모델의 첫 레이어로 사용하는 경우
`input_shape` 키워드 인수를 제공하십시오
(정수 튜플, 배치 축은 포함하지 않습니다).
예. `data_format="channels_last"`인 128x128 RGB
사진의 경우 `input_shape=(128, 128, 3)`이 됩니다.

__인수__

- __filters__: 정수, 아웃풋 공간의 차원
    (다시 말해, 컨볼루션의 아웃풋 필터의 수).
- __kernel_size__: 정수 혹은 2개 정수의 튜플/리스트. 2D 컨볼루션
    윈도우의 높이와 넓이를 특정합니다.
    정수 하나로 모든 공간적 차원에 대해서 동일한
    값을 특정할 수도 있습니다.
- __strides__: 정수 혹은 2개 정수의 튜플/리스트.
    높이와 넓이에 따라 컨볼루션의
    보폭 길이를 특정합니다.
    정수 하나로 모든 공간적 차원에 대해서 동일한
    값을 특정할 수도 있습니다.
    `stride` 값 != 1이면, `dilation_rate` 값 != 1의 어떤 경우와도
    호환이 불가능합니다.
- __padding__: `"valid"` 혹은 `"same"` (대소문자 무시).
- __output_padding__: 정수 혹은 2개 정수의 튜플/리스트.
    아웃풋 텐서의 높이와 넓이에 따라 패딩의
    양을 특정합니다.
    정수 하나로 모든 공간적 차원에 대해서 동일한
    값을 특정할 수도 있습니다.
    주어진 차원에 대한 아웃풋 패딩의 양은 같은 차원에
    대한 보폭 값보다 낮아야 합니다.
    `None` (디폴트값)으로 설정된 경우, 아웃풋 형태가 유추됩니다.
- __data_format__: 문자열,
    `"channels_last"` 혹은 "channels_first"` 중 하나.
    인풋 내 차원의 순서를 나타냅니다.
    `"channels_last"`는 `(batch, height, width, channels)`
    형태의 인풋에 호응하고
    `"channels_first"`는 `(batch, channels, height, width)` 형태의
    인풋에 호응합니다.
    디폴트 값은 `~/.keras/keras.json`에 위치한
    케라스 구성 파일의 `image_data_format` 값으로 지정됩니다.
    따로 설정하지 않으면, 이는 `"channels_last"`가 됩니다.
- __dilation_rate__: 정수 혹은 2개 정수의 튜플/리스트.
    팽창 컨볼루션의 팽창 속도를 특정합니다.
    정수 하나로 모든 공간적 차원에 대해서 동일한
    값을 특정할 수도 있습니다.
    현재는 `dilation_rate` 값 != 1이면
    `strides` 값 != 1의 어떤 경우와도 호환이 불가합니다.
- __activation__: 사용할 활성화 함수
    ([활성화](../activations.md)를 참조하십시오).
    따로 정하지 않으면, 활성화가 적용되지 않습니다
    (다시 말해, "선형적" 활성화: `a(x) = x`).
- __use_bias__: 불리언, 레이어가 편향 벡터를 사용하는지 여부.
- __kernel_initializer__: `kernel` 가중치 행렬의 초기값 설정기
    ([초기값 설정기](../initializers.md)를 참조하십시오).
- __bias_initializer__: 편향 벡터의 초기값 설정기
    ([초기값 설정기](../initializers.md)를 참조하십시오).
- __kernel_regularizer__: `kernel` 가중치 행렬에
    적용되는 정규화 함수
    ([정규화](../regularizers.md)를 참조하십시오).
- __bias_regularizer__: 편향 벡터에 적용되는 정규화 함수
    ([정규화](../regularizers.md)를 참조하십시오).
- __activity_regularizer__: 레이어의 아웃풋(레이어의 “활성화”)에
    적용되는 정규화 함수
    ([정규화](../regularizers.md)를 참조하십시오).
- __kernel_constraint__: 커널 행렬에 적용되는 제약 함수
    ([제약](../constraints.md)을 참조하십시오).
- __bias_constraint__: 편향 벡터에 적용하는 제약 함수
    ([제약](../constraints.md)을 참조하십시오).

__인풋 형태__

`data_format`이 `"channels_first"`이면
`(batch, channels, rows, cols)`
형태의 4D 텐서.
`data_format`이 `"channels_last"`이면
`(batch, rows, cols, channels)`
형태의 4D 텐서.

__아웃풋 형태__

`data_format`이 `"channels_first"`이면
`(batch, filters, new_rows, new_cols)`
형태의 4D 텐서.
`data_format`이 `"channels_last"`이면
`(batch, new_rows, new_cols, filters)`
형태의 4D 텐서.
패딩에 의해 `rows`와 `cols` 값이 변했을 수도 있습니다.
`output_padding`이 특정된 경우:

```
new_rows = ((rows - 1) * strides[0] + kernel_size[0]
            - 2 * padding[0] + output_padding[0])
new_cols = ((cols - 1) * strides[1] + kernel_size[1]
            - 2 * padding[1] + output_padding[1])
```

__참조__

- [A guide to convolution arithmetic for deep learning](
   https://arxiv.org/abs/1603.07285v1)
- [Deconvolutional Networks](
   https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf)
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L499)</span>
### Conv3D

```python
keras.layers.Conv3D(filters, kernel_size, strides=(1, 1, 1), padding='valid', data_format=None, dilation_rate=(1, 1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

3D 컨볼루션 레이어 (예. 부피에 대한 공간적 컨볼루션).

이 레이어는 레이어의 인풋과 컨볼루션되어
아웃풋 텐서를 만들어내는 컨볼루션 커널을
생성합니다. `use_bias`가 참일 경우,
편향 벡터를 만들어 아웃풋에 더합니다. 끝으로,
`activation`이 `None`이 아닐 경우, 이는 아웃풋에도 적용됩니다.

이 레이어를 모델의 첫 레이어로 사용하는 경우
`input_shape` 키워드 인수를 제공하십시오
(정수 튜플, 배치 축은 포함하지 않습니다).
예. `data_format="channels_last"`인,
단일 채널의 128x128x128 부피의 경우
`input_shape=(128, 128, 128, 1)`이 됩니다.

__인수__

- __filters__: 정수, 아웃풋 공간의 차원
    (다시 말해, 컨볼루션의 아웃풋 필터의 수).
- __kernel_size__: 정수 혹은 3개 정수의 튜플/리스트. 3D 컨볼루션
    윈도우의 깊이, 높이와 넓이를 특정합니다.
    정수 하나로 모든 공간적 차원에 대해서 동일한
    값을 특정할 수도 있습니다.
- __strides__: 정수 혹은 3개 정수의 튜플/리스트.
    각 공간적 차원에 따라 컨볼루션의 보폭 길이를 특정합니다.
    정수 하나로 모든 공간적 차원에 대해서 동일한
    값을 특정할 수도 있습니다.
    `stride` 값 != 1이면, `dilation_rate` 값 != 1의 어떤 경우와도
    호환이 불가능합니다.
- __padding__: `"valid"` 혹은 `"same"` (대소문자 무시).
- __data_format__: 문자열,
    `"channels_last"` 혹은 "channels_first"` 중 하나.
    인풋 내 차원의 순서를 나타냅니다.
    `"channels_last"`는 `(batch, height, width, channels)`
    형태의 인풋에 호응하고
    `"channels_first"`는 `(batch, channels, height, width)` 형태의
    인풋에 호응합니다.
    디폴트 값은 `~/.keras/keras.json`에 위치한
    케라스 구성 파일의 `image_data_format` 값으로 지정됩니다.
    따로 설정하지 않으면, 이는 `"channels_last"`가 됩니다.
- __dilation_rate__: 정수 혹은 3개의 정수의 튜플/리스트.
    팽창 컨볼루션의 팽창 속도를 특정합니다.
    정수 하나로 모든 공간적 차원에 대해서 동일한
    값을 특정할 수도 있습니다.
    현재는 `dilation_rate` 값 != 1이면
    `strides` 값 != 1의 어떤 경우와도 호환이 불가합니다.
- __activation__: 사용할 활성화 함수
    ([활성화](../activations.md)를 참조하십시오).
    따로 정하지 않으면, 활성화가 적용되지 않습니다
    (다시 말해, "선형적" 활성화: `a(x) = x`).
- __use_bias__: 불리언, 레이어가 편향 벡터를 사용하는지 여부.
- __kernel_initializer__: `kernel` 가중치 행렬의 초기값 설정기
    ([초기값 설정기](../initializers.md)를 참조하십시오).
- __bias_initializer__: 편향 벡터의 초기값 설정기
    ([초기값 설정기](../initializers.md)를 참조하십시오).
- __kernel_regularizer__: `kernel` 가중치 행렬에
    적용되는 정규화 함수
    ([정규화](../regularizers.md)를 참조하십시오).
- __bias_regularizer__: 편향 벡터에 적용되는 정규화 함수
    ([정규화](../regularizers.md)를 참조하십시오).
- __activity_regularizer__: 레이어의 아웃풋(레이어의 “활성화”)에
    적용되는 정규화 함수
    ([정규화](../regularizers.md)를 참조하십시오).
- __kernel_constraint__: 커널 행렬에 적용되는 제약 함수
    ([제약](../constraints.md)을 참조하십시오).
- __bias_constraint__: 편향 벡터에 적용하는 제약 함수
    ([제약](../constraints.md)을 참조하십시오).

__인풋 형태__

`data_format`이 `"channels_first"`이면
`(batch, channels, conv_dim1, conv_dim2, conv_dim3)`
형태의 5D 텐서.
`data_format`이 `"channels_last"`이면
`(batch, conv_dim1, conv_dim2, conv_dim3, channels)`
형태의 5D 텐서..

__아웃풋 형태__

`data_format`이 `"channels_first"`이면
`(batch, filters, new_conv_dim1, new_conv_dim2, new_conv_dim3)`
형태의 5D 텐서.
`data_format`이 `"channels_last"`이면
`(batch, new_conv_dim1, new_conv_dim2, new_conv_dim3, filters)`
형태의 5D 텐서.
패딩으로 인해 `new_conv_dim1`, `new_conv_dim2`와 `new_conv_dim3`
값이 변했을 수도 있습니다.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L901)</span>
### Conv3DTranspose

```python
keras.layers.Conv3DTranspose(filters, kernel_size, strides=(1, 1, 1), padding='valid', output_padding=None, data_format=None, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

전치 컨볼루션 레이어 (디컨볼루션으로 불리기도 합니다).

전치 컨볼루션은 일반적으로 다음의 상황에서 필요합니다:
보통의 컨볼루션과 반대방향으로 가는 변형을 사용하고
싶은 경우, 다시 말해 해당 컨볼루션과 호환되는 연결 패턴을
유지하면서 아웃풋의 형태를 가진 어떤 것을
인풋의 형태로 바꾸고 싶은 경우
유용합니다.

이 레이어를 모델의 첫 레이어로 사용하는 경우
`input_shape` 키워드 인수를 제공하십시오
(정수 튜플, 배치 축은 포함하지 않습니다).
예. `data_format="channels_last"`인 3개 채널의 128x128x128 부피의 경우
`input_shape=(128, 128, 128, 1)`이 됩니다.

__인수__

- __filters__: 정수, 아웃풋 공간의 차원
    (다시 말해, 컨볼루션의 아웃풋 필터의 수).
- __kernel_size__: 정수 혹은 3개 정수의 튜플/리스트. 3D 컨볼루션
    윈도우의 깊이, 높이와 넓이를 특정합니다.
    정수 하나로 모든 공간적 차원에 대해서 동일한
    값을 특정할 수도 있습니다.
- __strides__: 정수 혹은 3개 정수의 튜플/리스트.
    각 공간적 차원에 따라 컨볼루션의 보폭 길이를 특정합니다.
    정수 하나로 모든 공간적 차원에 대해서 동일한
    값을 특정할 수도 있습니다.
    `stride` 값 != 1이면, `dilation_rate` 값 != 1의 어떤 경우와도
    호환이 불가능합니다.
- __padding__: `"valid"` 혹은 `"same"` (대소문자 무시).
- __output_padding__: 정수 혹은 3개 정수의 튜플/리스트.
    아웃풋 텐서의 깊이, 높이, 넓이에 따라 패딩의 양을
    특정합니다.
    정수 하나로 모든 공간적 차원에 대해서 동일한
    값을 특정할 수도 있습니다.
    주어진 차원에 대한 아웃풋 패딩의 양은 같은 차원에
    대한 보폭 값보다 낮아야 합니다.
    `None` (디폴트값)으로 설정된 경우, 아웃풋 형태가 유추됩니다.
- __data_format__: 문자열,
    `"channels_last"` 혹은 "channels_first"` 중 하나.
    인풋 내 차원의 순서를 나타냅니다.
    `"channels_last"`는 `(batch, height, width, channels)`
    형태의 인풋에 호응하고
    `"channels_first"`는 `(batch, channels, height, width)` 형태의
    인풋에 호응합니다.
    디폴트 값은 `~/.keras/keras.json`에 위치한
    케라스 구성 파일의 `image_data_format` 값으로 지정됩니다.
    따로 설정하지 않으면, 이는 `"channels_last"`가 됩니다.
- __dilation_rate__: 정수 혹은 3개의 정수의 튜플/리스트.
    팽창 컨볼루션의 팽창 속도를 특정합니다.
    정수 하나로 모든 공간적 차원에 대해서 동일한
    값을 특정할 수도 있습니다.
    현재는 `dilation_rate` 값 != 1이면
    `strides` 값 != 1의 어떤 경우와도 호환이 불가합니다.
- __activation__: 사용할 활성화 함수
    ([활성화](../activations.md)를 참조하십시오).
    따로 정하지 않으면, 활성화가 적용되지 않습니다
    (다시 말해, "선형적" 활성화: `a(x) = x`).
- __use_bias__: 불리언, 레이어가 편향 벡터를 사용하는지 여부.
- __kernel_initializer__: `kernel` 가중치 행렬의 초기값 설정기
    ([초기값 설정기](../initializers.md)를 참조하십시오).
- __bias_initializer__: 편향 벡터의 초기값 설정기
    ([초기값 설정기](../initializers.md)를 참조하십시오).
- __kernel_regularizer__: `kernel` 가중치 행렬에
    적용되는 정규화 함수
    ([정규화](../regularizers.md)를 참조하십시오).
- __bias_regularizer__: 편향 벡터에 적용되는 정규화 함수
    ([정규화](../regularizers.md)를 참조하십시오).
- __activity_regularizer__: 레이어의 아웃풋(레이어의 “활성화”)에
    적용되는 정규화 함수
    ([정규화](../regularizers.md)를 참조하십시오).
- __kernel_constraint__: 커널 행렬에 적용되는 제약 함수
    ([제약](../constraints.md)을 참조하십시오).
- __bias_constraint__: 편향 벡터에 적용하는 제약 함수
    ([제약](../constraints.md)을 참조하십시오).

__인풋 형태__

`data_format`이 `"channels_first"`이면
`(batch, channels, depth, rows, cols)`
형태의 5D 텐서.
`data_format`이 `"channels_last"`이면
`(batch, depth, rows, cols, channels)`
형태의 5D 텐서.

__아웃풋 형태__

`data_format`이 `"channels_first"`이면
`(batch, filters, new_depth, new_rows, new_cols)`
형태의 5D 텐서.
`data_format`이 `"channels_last"`이면
`(batch, new_depth, new_rows, new_cols, filters)`
형태의 5D 텐서.
패딩으로 인해, `depth`, `rows`와 `cols` 값이 달라졌을 수도 있습니다.
`output_padding`이 특정된 경우:

```
new_depth = ((depth - 1) * strides[0] + kernel_size[0]
             - 2 * padding[0] + output_padding[0])
new_rows = ((rows - 1) * strides[1] + kernel_size[1]
            - 2 * padding[1] + output_padding[1])
new_cols = ((cols - 1) * strides[2] + kernel_size[2]
            - 2 * padding[2] + output_padding[2])
```

__참조__

- [A guide to convolution arithmetic for deep learning](
   https://arxiv.org/abs/1603.07285v1)
- [Deconvolutional Networks](
   https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf)
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L2376)</span>
### Cropping1D

```python
keras.layers.Cropping1D(cropping=(1, 1))
```

1D 인풋(예. 시간적 시퀀스)용 크롭핑 레이어.

시간 차원(축 1)을 따라서 잘라냅니다.

__인수__

- __cropping__: 정수, 혹은 (길이 2의) 정수 튜플.
    크롭핑 차원(축1)의 시작과 끝에서
    잘라낼 유닛의 개수.
    단일 정수가 제공된 경우
    시작과 끝 모두 같은 값이 적용됩니다.

__인풋 형태__

`(batch, axis_to_crop, features)` 형태의 3D 텐서.

__아웃풋 형태__

`(batch, cropped_axis, features)` 형태의 3D 텐서.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L2408)</span>
### Cropping2D

```python
keras.layers.Cropping2D(cropping=((0, 0), (0, 0)), data_format=None)
```

2D 인풋(예. 사진)용 크롭핑 레이어 .

공간 차원에 따라, 다시 말해 높이와 넓이에 따라 절단합니다.

__인수__

- __cropping__: 정수, 2개 정수의 튜플, 혹은 2개 정수의 튜플 2개로 이루어진 튜플.
    - 정수인 경우: 높이와 넓이에
        동일한 대칭 크롭핑이 적용됩니다.
    - 2개 정수의 튜플인 경우:
        높이와 넓이에 대한
        두 가지 대칭 크롭핑 값을 나타냅니다
        `(symmetric_height_crop, symmetric_width_crop)`.
    - 2개 정수의 튜플 2개로 이루어진 튜플인 경우:
        `((top_crop, bottom_crop), (left_crop, right_crop))`으로
        보면 됩니다.
- __data_format__: 문자열,
    `"channels_last"` 혹은 "channels_first"` 중 하나.
    인풋 내 차원의 순서를 나타냅니다.
    `"channels_last"`는 `(batch, height, width, channels)`
    형태의 인풋에 호응하고
    `"channels_first"`는 `(batch, channels, height, width)` 형태의
    인풋에 호응합니다.
    디폴트 값은 `~/.keras/keras.json`에 위치한
    케라스 구성 파일의 `image_data_format` 값으로 지정됩니다.
    따로 설정하지 않으면, 이는 `"channels_last"`가 됩니다.

__인풋 형태__

다음 형태의 4D 텐서:
- `data_format`이 `"channels_last"`이면
    `(batch, rows, cols, channels)`
- `data_format`이 `"channels_first"`이면
    `(batch, channels, rows, cols)`

__아웃풋 형태__

다음 형태의 4D 텐서:
- `data_format`이 `"channels_last"`이면
    `(batch, cropped_rows, cropped_cols, channels)`
- `data_format`이 `"channels_first"`이면
    `(batch, channels, cropped_rows, cropped_cols)`

__예시__


```python
# 2D 인풋 이미지 혹은 특성 맵을 잘라냅니다
model = Sequential()
model.add(Cropping2D(cropping=((2, 2), (4, 4)),
                     input_shape=(28, 28, 3)))
# 현재 model.output_shape == (None, 24, 20, 3)
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Cropping2D(cropping=((2, 2), (2, 2))))
# 현재 model.output_shape == (None, 20, 16, 64)
```
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L2491)</span>
### Cropping3D

```python
keras.layers.Cropping3D(cropping=((1, 1), (1, 1), (1, 1)), data_format=None)
```

3D 데이터(예. 공간적 혹은 시공간적)용 크롭핑 레이어.

__인수__

- __cropping__: 정수, 3개 정수의 튜플, 혹은 2개 정수의 튜플 3개로 이루어진 튜플.
    - 정수인 경우: 깊이, 높이, 넓이에
        동일한 대칭 크롭핑이 적용됩니다.
    - 3개 정수의 튜플인 경우:
        깊이, 높이, 넓이에 대한
        두 가지 대칭 크롭핑 값을 나타냅니다:
        `(symmetric_dim1_crop, symmetric_dim2_crop, symmetric_dim3_crop)`.
    - 2개 정수의 튜플 3개로 이루어진 튜플인 경우:
        `((left_dim1_crop, right_dim1_crop),
          (left_dim2_crop, right_dim2_crop),
          (left_dim3_crop, right_dim3_crop))`으로
        보면 됩니다.        
- __data_format__: 문자열,
    `"channels_last"` 혹은 "channels_first"` 중 하나.
    인풋 내 차원의 순서를 나타냅니다.
    `"channels_last"`는 `(batch, height, width, channels)`
    형태의 인풋에 호응하고
    `"channels_first"`는 `(batch, channels, height, width)` 형태의
    인풋에 호응합니다.
    디폴트 값은 `~/.keras/keras.json`에 위치한
    케라스 구성 파일의 `image_data_format` 값으로 지정됩니다.
    따로 설정하지 않으면, 이는 `"channels_last"`가 됩니다.

__인풋 형태__

다음 형태의 5D 텐서:
- `data_format`이 `"channels_last"`이면
    `(batch, first_axis_to_crop, second_axis_to_crop, third_axis_to_crop,
      depth)`
- `data_format`이 `"channels_first"`이면
    `(batch, depth,
      first_axis_to_crop, second_axis_to_crop, third_axis_to_crop)`

__아웃풋 형태__

다음 형태의 5D 텐서:
- `data_format`이 `"channels_last"`이면
    `(batch, first_cropped_axis, second_cropped_axis, third_cropped_axis,
      depth)`
- `data_format`이 `"channels_first"`이면
    `(batch, depth,
      first_cropped_axis, second_cropped_axis, third_cropped_axis)`
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L1944)</span>
### UpSampling1D

```python
keras.layers.UpSampling1D(size=2)
```

1D 인풋용 업샘플링 레이어.

시간 축에 따라 각 시간 단계를 `size` 회 반복합니다.

__인수__

- __size__: 정수. 업샘플링 인수.

__인풋 형태__

`(batch, steps, features)` 형태의 3D 텐서.

__아웃풋 형태__

3D tensor with shape: `(batch, upsampled_steps, features)`.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L1974)</span>
### UpSampling2D

```python
keras.layers.UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest')
```

2D 인풋용 업샘플링 레이어.

데이터의 행과 열을
각각 size[0]과 size[1]회씩 반복합니다.

__인수__

- __size__: 정수, 혹은 2개 정수의 튜플.
    행과 열에 대한 업샘플링 인수.
- __data_format__: 문자열,
    `"channels_last"` 혹은 "channels_first"` 중 하나.
    인풋 내 차원의 순서를 나타냅니다.
    `"channels_last"`는 `(batch, height, width, channels)`
    형태의 인풋에 호응하고
    `"channels_first"`는 `(batch, channels, height, width)` 형태의
    인풋에 호응합니다.
    디폴트 값은 `~/.keras/keras.json`에 위치한
    케라스 구성 파일의 `image_data_format` 값으로 지정됩니다.
    따로 설정하지 않으면, 이는 `"channels_last"`가 됩니다.
- __interpolation__: A string, one of `nearest` or `bilinear`.
    Note that CNTK does not support yet the `bilinear` upscaling
    and that with Theano, only `size=(2, 2)` is possible.

__인풋 형태__

다음 형태의 4D 텐서:
- `data_format`이 `"channels_last"`이면
    `(batch, rows, cols, channels)`
- `data_format`이 `"channels_first"`이면
    `(batch, channels, rows, cols)`

__아웃풋 형태__

다음 형태의 4D 텐서:
- `data_format`이 `"channels_last"`이면
    `(batch, upsampled_rows, upsampled_cols, channels)`
- `data_format`이 `"channels_first"`이면
    `(batch, channels, upsampled_rows, upsampled_cols)`
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L2032)</span>
### UpSampling3D

```python
keras.layers.UpSampling3D(size=(2, 2, 2), data_format=None)
```

3D 인풋용 업샘플링 레이어.

데이터의 첫 번째, 두 번째, 세 번째 차원을
각각 size[0], size[1], size[2]회씩 반복합니다.

__인수__

- __size__: int, or tuple of 3 integers.
    The upsampling factors for dim1, dim2 and dim3.
- __data_format__: 문자열,
    `"channels_last"` 혹은 "channels_first"` 중 하나.
    인풋 내 차원의 순서를 나타냅니다.
    `"channels_last"`는 `(batch, height, width, channels)`
    형태의 인풋에 호응하고
    `"channels_first"`는 `(batch, channels, height, width)` 형태의
    인풋에 호응합니다.
    디폴트 값은 `~/.keras/keras.json`에 위치한
    케라스 구성 파일의 `image_data_format` 값으로 지정됩니다.
    따로 설정하지 않으면, 이는 `"channels_last"`가 됩니다.

__인풋 형태__

다음 형태의 5D 텐서:
- `data_format`이 `"channels_last"`이면
    `(batch, dim1, dim2, dim3, channels)`
- `data_format`이 `"channels_first"`이면
    `(batch, channels, dim1, dim2, dim3)`

__아웃풋 형태__

다음 형태의 5D 텐서:
- `data_format`이 `"channels_last"`이면
    `(batch, upsampled_dim1, upsampled_dim2, upsampled_dim3, channels)`
- `data_format`이 `"channels_first"`이면
    `(batch, channels, upsampled_dim1, upsampled_dim2, upsampled_dim3)`
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L2124)</span>
### ZeroPadding1D

```python
keras.layers.ZeroPadding1D(padding=1)
```

1D 인풋(예. 시간적 시퀀스)용 제로-패딩 레이어.

__인수__

- __padding__: 정수, (길이 2의) 정수 튜플, 혹은 딕셔너리.
    - 정수의 경우:

    패딩 차원(축 1)의 시작과 끝에 
    채울 0 값의 개수.

    - (길이 2의) 정수 튜플의 경우:

    패딩 차원(`(left_pad, right_pad)`)의
    시작과 끝에 채울 0 값의 개수.

__인풋 형태__

`(batch, axis_to_pad, features)` 형태의 3D 텐서.

__아웃풋 형태__

`(batch, padded_axis, features)` 형태의 3D 텐서.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L2159)</span>
### ZeroPadding2D

```python
keras.layers.ZeroPadding2D(padding=(1, 1), data_format=None)
```

2D 인풋(예. 사진)의 제로 패딩 레이어.

이 레이어는 이미지 텐서의 위, 아래, 좌, 우에
0으로 이루어진 행과 열을 더할 수 있습니다.

__인수__

- __padding__: 정수, 2개 정수의 튜플, 혹은 2개 정수의 튜플 2개로 이루어진 튜플.
    - 정수인 경우: 높이와 넓이에
        동일한 대칭 패딩이 적용됩니다.
    - 2개 정수의 튜플인 경우:
        높이와 넓이에 대한
        두 가지 대칭 패팅 값을 나타냅니다:
        `(symmetric_height_pad, symmetric_width_pad)`.
    - 2개 정수의 튜플 2개로 이루어진 튜플인 경우:
        `((top_pad, bottom_pad), (left_pad, right_pad))`로
        보면 됩니다.
- __data_format__: 문자열,
    `"channels_last"` 혹은 "channels_first"` 중 하나.
    인풋 내 차원의 순서를 나타냅니다.
    `"channels_last"`는 `(batch, height, width, channels)`
    형태의 인풋에 호응하고
    `"channels_first"`는 `(batch, channels, height, width)` 형태의
    인풋에 호응합니다.
    디폴트 값은 `~/.keras/keras.json`에 위치한
    케라스 구성 파일의 `image_data_format` 값으로 지정됩니다.
    따로 설정하지 않으면, 이는 `"channels_last"`가 됩니다.

__인풋 형태__

다음 형태의 4D 텐서:
- `data_format`이 `"channels_last"`이면
    `(batch, rows, cols, channels)`
- `data_format`이 `"channels_last"`이면
    `(batch, channels, rows, cols)`

__아웃풋 형태__

다음 형태의 4D 텐서:
- `data_format`이 `"channels_last"`이면
    `(batch, padded_rows, padded_cols, channels)`
- `data_format`이 `"channels_last"`이면
    `(batch, channels, padded_rows, padded_cols)`
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L2235)</span>
### ZeroPadding3D

```python
keras.layers.ZeroPadding3D(padding=(1, 1, 1), data_format=None)
```

3D 데이터(예. 공간적, 혹은 시공간적)용 제로-패딩 레이어.

__인수__

- __padding__: 정수, 3개 정수의 튜플, 혹은 2개 정수의 튜플 3개로 이루어진 튜플.
    - 정수인 경우: 깊이, 높이, 넓이에
        동일한 대칭 패딩이 적용됩니다.
    - 3개 정수의 튜플인 경우:
        깊이, 높이, 넓이에 대한
        두 가지 대칭 패딩 값을 나타냅니다:
        `(symmetric_dim1_pad, symmetric_dim2_pad, symmetric_dim3_pad)`.
    - 2개 정수의 튜플 3개로 이루어진 튜플인 경우:
        `((left_dim1_pad, right_dim1_pad),
          (left_dim2_pad, right_dim2_pad),
          (left_dim3_pad, right_dim3_pad))`로
        보면 됩니다.
- __data_format__: 문자열,
    `"channels_last"` 혹은 "channels_first"` 중 하나.
    인풋 내 차원의 순서를 나타냅니다.
    `"channels_last"`는 `(batch, height, width, channels)`
    형태의 인풋에 호응하고
    `"channels_first"`는 `(batch, channels, height, width)` 형태의
    인풋에 호응합니다.
    디폴트 값은 `~/.keras/keras.json`에 위치한
    케라스 구성 파일의 `image_data_format` 값으로 지정됩니다.
    따로 설정하지 않으면, 이는 `"channels_last"`가 됩니다.

__인풋 형태__

다음 형태의 5D 텐서:
- `data_format`이 `"channels_last"`이면
    `(batch, first_axis_to_pad, second_axis_to_pad, third_axis_to_pad,
      depth)`
- `data_format`이 `"channels_first"`이면
    `(batch, depth,
      first_axis_to_pad, second_axis_to_pad, third_axis_to_pad)`

__아웃풋 형태__

다음 형태의 5D 텐서:
- `data_format`이 `"channels_last"`이면
    `(batch, first_padded_axis, second_padded_axis, third_axis_to_pad,
      depth)`
- `data_format`이 `"channels_first"`이면
    `(batch, depth,
      first_padded_axis, second_padded_axis, third_axis_to_pad)`
    

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L241)</span>
### Conv1D

```python
keras.layers.Conv1D(filters, kernel_size, strides=1, padding='valid', data_format='channels_last', dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

1D 합성곱 층<sub>convolution layer</sub> (예: 시계열<sub>temporal</sub> 합성곱).

이 층은 하나의 공간적 (혹은 시간적) 차원에서 입력 텐서와 합성곱되어 출력 텐서를 만드는 합성곱 커널을 생성합니다. `use_bias`가 참인 경우, 편향 <sub>bias</sub>벡터를 생성해 출력 텐서에 더합니다. `activation`이 `None`이 아닌 경우 이 또한 출력 텐서에 적용됩니다.  
  
모델의 가장 처음에 올 경우 `input_shape`인자를 통해 입력값의 형태를 함께 지정해야 합니다. `input_shape`는 `None`또는 정수로 된 튜플로 배치 축<sub>axis</sub>은 포함시키지 않습니다. 예를 들어, 128개의 요인<sub>feature</sub>과 10개의 시간 단계로 이루어진 시계열 데이터에 `Conv1D`를 적용하고자 하는 경우 `input_shape=(10, 128)`, `data_format="channels_last"`로 지정합니다. `data_format` 인자를 `"channels_last"`로 정하는 까닭은 일반적으로 합성곱 신경망에서 다루는 채널의 위치를 가장 마지막에 두도록 함으로써 층이 입력값의 마지막에 있는 요인 차원을 일종의 채널처럼 취급하게끔 하고 합성곱의 필터가 시계열 차원을 따라서 적용되게끔 하기 위함입니다. 만약 시계열의 길이가 표본 또는 배치별로 다를 경우 `input_shape=(None, 128)`로 지정합니다.

__인자__

- __filters__: `int`. 출력할 결과값의 차원으로 합성곱 필터의 갯수를 나타냅니다. 
- __kernel_size__: `int` 또는 `int`로 이루어진 튜플/리스트. 1D 합성곱 필터의 크기를 지정합니다.
- __strides__: `int` 또는 `int`로 이루어진 튜플/리스트. 합성곱 필터의 스트라이드를 지정합니다. 기본값은 `(1, 1)`입니다. 만약 팽창 합성곱<sub>dilated convolution</sub>을 사용하고자 할 때 스트라이드의 크기를 `1`보다 크게 지정했다면 `dilation_rate`인자는 반드시 `1`로 맞춰야 합니다.
- __padding__: `str`. 입력값의 패딩처리 여부를 `"valid"`, `"causal"` 또는 `"same"` 가운데 하나로 지정합니다(대소문자 무관). `"valid"`는 패딩이 없는 경우, `"same"`은 출력의 형태를 입력과 같게 맞추고자 하는 경우에 사용합니다. `"causal"`은 인과적<sub>causal</sub> 혹은 팽창 인과적<sub>dilated causal</sub> 합성곱을 수행하게끔 합니다. 인과적 합성곱은 `t`시점의 결과값이 오직 `t`보다 이전 시점의 입력값에만 영향을 받도록 하는 합성곱 방식입니다. 이 경우 입출력의 길이를 맞추기 위해 `0`값으로 패딩을 처리하게 되며, 시간 순서를 지켜야 하는 데이터를 모델링해야 하는 경우에 유용합니다. 참고: [WaveNet: A Generative Model for Raw Audio, section 2.1](https://arxiv.org/abs/1609.03499)
- __data_format__: `str`. 입력 데이터의 차원 순서를 정의하는 인자로 `"channels_last"`(기본값) 또는 `"channels_first"` 가운데 하나를 지정합니다. 입력 형태가 `(batch, steps, channels)`로 채널 정보가 마지막에 올 경우 `"channels_last"`를, `(batch, channels, steps)`로 채널 정보가 먼저 올 경우 `"channels_first"`를 선택합니다. 
- __dilation_rate__: `int` 또는 `int`로 이루어진 튜플/리스트. 팽창 합성곱 필터의 팽창비율을 결정합니다. 팽창 합성곱은 원래 조밀한 형태 그대로 입력에 적용되는 합성곱 필터를 각 방향으로 원소 사이의 간격을 띄우는 방식으로 팽창시켜 성긴 대신 보다 넓은 영역에 적용될 수 있도록 변형한 합성곱입니다. 자세한 내용은 [Multi-Scale Context Aggregation by Dilated Convolutions](https://arxiv.org/abs/1511.07122v3)을 참고하십시오. 기본값은 `(1, 1)`이며, 현재 버전에서는 `dilation_rate`가 `1`보다 큰 경우 `1`보다 큰 `strides`를 지정할 수 없습니다. 
- __activation__: 사용할 활성화 함수입니다. 기본값은 `None`으로, 별도로 지정하지 않으면 전달할 경우 활성화 함수가 적용되지 않습니다 (`a(x) = x`). 참고: [활성화 함수](../activations.md)
- __use_bias__: `bool`, 층의 연산에 편향을 적용할지 여부를 결정합니다.
- __kernel_initializer__: `kernel` 가중치 행렬의 초기화 함수를 결정합니다. 참고: [초기화 함수](../initializers.md)
- __bias_initializer__: 편향 벡터의 초기화 함수를 결정합니다. 참고: [초기화 함수](../initializers.md)
- __kernel_regularizer__: `kernel` 가중치 행렬에 적용할 규제 함수를 결정합니다. 참고: [규제 함수](../regularizers.md)
- __bias_regularizer__: 편향 벡터에 적용할 규제 함수를 결정합니다. 참고: [규제 함수](../regularizers.md)
- __activity_regularizer__: 층의 출력값에 적용할 규제 함수를 결정합니다. 참고: [규제 함수](../regularizers.md)
- __kernel_constraint__: `kernel` 가중치 행렬에 적용할 제약을 결정합니다. 참고: [제약](../constraints.md)
- __bias_constraint__: 편향 벡터에 적용할 제약을 결정합니다. 참고: [제약](../constraints.md)

__입력 형태__

`(batch, steps, channels)` 형태의 3D 텐서.

__출력 형태__

`(batch, new_steps, filters)` 형태의 3D 텐서. 패딩이나 스트라이드로 인해 `steps`값이 바뀔 수 있습니다.


----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L368)</span>
### Conv2D

```python
keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

2D 합성곱 층 (예: 이미지 데이터에서 이루어지는 공간 차원의 합성곱).

이 층은 입력 텐서와 합성곱되어 출력 텐서를 만드는 합성곱 커널을 생성합니다. `use_bias`가 참인 경우, 편향 <sub>bias</sub>벡터를 생성해 출력 텐서에 더합니다. `activation`이 `None`이 아닌 경우 이 또한 출력 텐서에 적용됩니다.  

모델의 가장 처음에 올 경우 `input_shape`인자를 통해 입력값의 형태를 함께 지정해야 합니다. `input_shape`는 `None`또는 정수로 된 튜플로 배치 축<sub>axis</sub>은 포함시키지 않습니다. 예를 들어 `data_format="channels_last"`인 128x128 RGB 이미지의 경우 `input_shape=(128, 128, 3)`이 됩니다.

__인자__

- __filters__: `int`. 출력할 결과값의 차원으로 합성곱 필터의 갯수를 나타냅니다. 
- __kernel_size__: `int` 또는 2개의 `int`로 이루어진 튜플/리스트. 2D 합성곱 필터의 행방향과 열방향 크기를 지정합니다. `int`하나를 입력할 경우 모든 방향의 크기를 동일하게 지정합니다.
- __strides__: `int` 또는 2개의 `int`로 이루어진 튜플/리스트. 합성곱 필터의 스트라이드를 지정합니다. `int`하나를 입력할 경우 열방향, 행방향의 스트라이드를 동일하게 지정합니다. 기본값은 `(1, 1)`입니다. 만약 팽창 합성곱을 사용하고자 할 때 스트라이드의 크기를 `1`보다 크게 지정했다면 `dilation_rate`인자는 반드시 `1`로 맞춰야 합니다.
- __padding__: `str`. 입력값의 패딩처리 여부를 `"valid"`, `"causal"` 또는 `"same"` 가운데 하나로 지정합니다(대소문자 무관). `"valid"`는 패딩이 없는 경우, `"same"`은 출력의 형태를 입력과 같게 맞추고자 하는 경우에 사용합니다. `"same"`의 경우 `strides`가 `1`이 아닐 때, 사용하는 백엔드에 따라 값이 조금씩 달라질 수 있습니다 [참고](https://github.com/keras-team/keras/pull/9473#issuecomment-372166860).
- __data_format__: `str`. 입력 데이터의 차원 순서를 정의하는 인자로 `"channels_last"`(기본값) 또는 `"channels_first"` 가운데 하나를 지정합니다. 입력 형태가 `(batch, height, width, channels)`로 채널 정보가 마지막에 올 경우 `"channels_last"`를, `(batch, channels, height, width)`로 채널 정보가 먼저 올 경우 `"channels_first"`를 선택합니다. 케라스 설정 `~/.keras/keras.json`파일에 있는 `image_data_format`값을 기본값으로 사용하며, 해당 값이 없는 경우 자동으로 `"channels_last"`를 기본값으로 적용합니다. 
- __dilation_rate__: `int` 또는 2개의 `int`로 이루어진 튜플/리스트. 팽창 합성곱 필터의 팽창비율을 결정합니다. 팽창 합성곱은 원래 조밀한 형태 그대로 입력에 적용되는 합성곱 필터를 각 방향으로 원소 사이의 간격을 띄우는 방식으로 팽창시켜 성긴 대신 보다 넓은 영역에 적용될 수 있도록 변형한 합성곱입니다. 자세한 내용은 [Multi-Scale Context Aggregation by Dilated Convolutions](https://arxiv.org/abs/1511.07122v3)을 참고하십시오. 기본값은 `(1, 1)`이며, 현재 버전에서는 `dilation_rate`가 `1`보다 큰 경우 `1`보다 큰 `strides`를 지정할 수 없습니다. 
- __activation__: 사용할 활성화 함수입니다. 기본값은 `None`으로, 별도로 지정하지 않으면 전달할 경우 활성화 함수가 적용되지 않습니다 (`a(x) = x`). 참고: [활성화 함수](../activations.md)
- __use_bias__: `bool`, 층의 연산에 편향을 적용할지 여부를 결정합니다.
- __kernel_initializer__: `kernel` 가중치 행렬의 초기화 함수를 결정합니다. 참고: [초기화 함수](../initializers.md)
- __bias_initializer__: 편향 벡터의 초기화 함수를 결정합니다. 참고: [초기화 함수](../initializers.md)
- __kernel_regularizer__: `kernel` 가중치 행렬에 적용할 규제 함수를 결정합니다. 참고: [규제 함수](../regularizers.md)
- __bias_regularizer__: 편향 벡터에 적용할 규제 함수를 결정합니다. 참고: [규제 함수](../regularizers.md)
- __activity_regularizer__: 층의 출력값에 적용할 규제 함수를 결정합니다. 참고: [규제 함수](../regularizers.md)
- __kernel_constraint__: `kernel` 가중치 행렬에 적용할 제약을 결정합니다. 참고: [제약](../constraints.md)
- __bias_constraint__: 편향 벡터에 적용할 제약을 결정합니다. 참고: [제약](../constraints.md)

__입력 형태__

`data_format`이 `"channels_first"`이면 `(batch, channels, rows, cols)` 형태의 4D 텐서.  
`data_format`이 `"channels_last"`이면 `(batch, rows, cols, channels)` 형태의 4D 텐서.  

__출력 형태__

`data_format`이 `"channels_first"`이면 `(batch, filters, new_rows, new_cols)` 형태의 4D 텐서.  
`data_format`이 `"channels_last"`이면 `(batch, new_rows, new_cols, filters)` 형태의 4D 텐서.  
패딩으로 인해 `rows`와 `cols` 값이 바뀔  수 있습니다.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L1421)</span>
### SeparableConv1D

```python
keras.layers.SeparableConv1D(filters, kernel_size, strides=1, padding='valid', data_format='channels_last', dilation_rate=1, depth_multiplier=1, activation=None, use_bias=True, depthwise_initializer='glorot_uniform', pointwise_initializer='glorot_uniform', bias_initializer='zeros', depthwise_regularizer=None, pointwise_regularizer=None, bias_regularizer=None, activity_regularizer=None, depthwise_constraint=None, pointwise_constraint=None, bias_constraint=None)
```
깊이별 분리 1D 합성곱<sub>Depthwise Separable 1D Convolution</sub>.

먼저 입력값의 각 채널(깊이)별로 따로 합성곱<sub>depthwise convolution</sub>을 한 뒤에, 다시 합쳐진 출력값의 각 위치<sub>point</sub>에서 채널 차원에 대한 합성곱<sub>pointwise convolution</sub>을 하여 앞의 결과를 하나로 묶습니다. 이때 `depth_multiplier`인자는 채널별 합성곱 단계에서 각 입력 채널당 몇 개의 출력 채널을 생성할 것인지 결정합니다. 직관적으로 볼 때 분리 합성곱은 합성곱 필터를 두 개의 작은 필터로 분해하혀 수행하는 합성곱 또는 극단적 형태의 인셉션 블록으로 이해할 수 있습니다.

__인자__

- __filters__: `int`. 출력할 결과값의 차원으로 합성곱 필터의 갯수를 나타냅니다. 
- __kernel_size__: `int` 또는 `int`로 이루어진 튜플/리스트. 1D 합성곱 필터의 크기를 지정합니다.
- __strides__: `int` 또는 `int`로 이루어진 튜플/리스트. 합성곱 필터의 스트라이드를 지정합니다. 기본값은 `(1, 1)`입니다. 만약 팽창 합성곱을 사용하고자 할 때 스트라이드의 크기를 `1`보다 크게 지정했다면 `dilation_rate`인자는 반드시 `1`로 맞춰야 합니다.
- __padding__: `str`. 입력값의 패딩처리 여부를 `"valid"` 또는 `"same"` 가운데 하나로 지정합니다(대소문자 무관). `"valid"`는 패딩이 없는 경우, `"same"`은 출력의 형태를 입력과 같게 맞추고자 하는 경우에 사용합니다. 
- __data_format__: `str`. 입력 데이터의 차원 순서를 정의하는 인자로 `"channels_last"`(기본값) 또는 `"channels_first"` 가운데 하나를 지정합니다. 입력 형태가 `(batch, steps, channels)`로 채널 정보가 마지막에 올 경우 `"channels_last"`를, `(batch, channels, steps)`로 채널 정보가 먼저 올 경우 `"channels_first"`를 선택합니다. 
- __dilation_rate__: `int` 또는 `int`로 이루어진 튜플/리스트. 팽창 합성곱 필터의 팽창비율을 결정합니다. 팽창 합성곱은 원래 조밀한 형태 그대로 입력에 적용되는 합성곱 필터를 각 방향으로 원소 사이의 간격을 띄우는 방식으로 팽창시켜 성긴 대신 보다 넓은 영역에 적용될 수 있도록 변형한 합성곱입니다. 자세한 내용은 [Multi-Scale Context Aggregation by Dilated Convolutions](https://arxiv.org/abs/1511.07122v3)을 참고하십시오. 기본값은 `(1, 1)`이며, 현재 버전에서는 `dilation_rate`가 `1`보다 큰 경우 `1`보다 큰 `strides`를 지정할 수 없습니다. 
- __depth_multiplier__: 각 입력 채널당 몇 개의 출력 채널을 생성할 것인지 결정합니다. 출력 채널의 총 개수는 `filters_in * depth_multiplier`가 됩니다.
- __activation__: 사용할 활성화 함수입니다. 기본값은 `None`으로, 별도로 지정하지 않으면 전달할 경우 활성화 함수가 적용되지 않습니다 (`a(x) = x`). 참고: [활성화 함수](../activations.md)
- __use_bias__: `bool`, 층의 연산에 편향을 적용할지 여부를 결정합니다.
- __depthwise_initializer__: 깊이별 필터 행렬의 초기화 함수를 결정합니다. 참고: [초기화 함수](../initializers.md)
- __pointwise_initializer__: 위치별 필터 행렬의 초기화 함수를 결정합니다. 참고: [초기화 함수](../initializers.md)
- __bias_initializer__: 편향 벡터의 초기화 함수를 결정합니다. 참고: [초기화 함수](../initializers.md)
- __depthwise_regularizer__: 깊이별 필터 행렬에 적용할 규제 함수를 결정합니다. 참고: [규제 함수](../regularizers.md)
- __pointwise_regularizer__: 위치별 필터 행렬에 적용할 규제 함수를 결정합니다. 참고: [규제 함수](../regularizers.md)
- __bias_regularizer__: 편향 벡터에 적용할 규제 함수를 결정합니다. 참고: [규제 함수](../regularizers.md)
- __activity_regularizer__: 층의 출력값에 적용할 규제 함수를 결정합니다. 참고: [규제 함수](../regularizers.md)
- __depthwise_constraint__: 깊이별 필터 행렬에 적용할 제약을 결정합니다. 참고: [제약](../constraints.md)
- __pointwise_constraint__: 위치별 필터 행렬에 적용할 제약을 결정합니다. 참고: [제약](../constraints.md)
- __bias_constraint__: 편향 벡터에 적용할 제약을 결정합니다. 참고: [제약](../constraints.md)

__입력 형태__

`data_format`이 `"channels_first"`이면 `(batch, channels, steps)` 형태의 3D 텐서.  
`data_format`이 `"channels_last"`이면 `(batch, steps, channels)` 형태의 3D 텐서.  

__출력 형태__

`data_format`이 `"channels_first"`이면 `(batch, filters, new_steps)` 형태의 3D 텐서.  
`data_format`이 `"channels_last"`이면 `(batch, new_steps, filters)` 형태의 3D 텐서.  

패딩이나 스트라이드로 인해 `new_steps` 값이 바뀔 수 있습니다.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L1553)</span>
### SeparableConv2D

```python
keras.layers.SeparableConv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), depth_multiplier=1, activation=None, use_bias=True, depthwise_initializer='glorot_uniform', pointwise_initializer='glorot_uniform', bias_initializer='zeros', depthwise_regularizer=None, pointwise_regularizer=None, bias_regularizer=None, activity_regularizer=None, depthwise_constraint=None, pointwise_constraint=None, bias_constraint=None)
```

깊이별 분리 2D 합성곱<sub>Depthwise Separable 2D Convolution</sub>.

먼저 입력값의 각 채널(깊이)별로 따로 공간 차원의 합성곱을 한 뒤에, 다시 합쳐진 출력값의 각 위치에서 채널 차원에 대한 합성곱을 하여 앞의 결과를 하나로 묶습니다. 이때 `depth_multiplier`인자는 채널별 합성곱 단계에서 각 입력 채널당 몇 개의 출력 채널을 생성할 것인지 결정합니다. 직관적으로 볼 때 분리 합성곱은 합성곱 필터를 두 개의 작은 필터로 분해하혀 수행하는 합성곱 또는 극단적 형태의 인셉션 블록으로 이해할 수 있습니다.

__인자__

- __filters__: `int`. 출력할 결과값의 차원으로 합성곱 필터의 갯수를 나타냅니다. 
- __kernel_size__: `int` 또는 2개의 `int`로 이루어진 튜플/리스트. 2D 합성곱 필터의 행방향과 열방향 크기를 지정합니다. `int`하나를 입력할 경우 모든 방향의 크기를 동일하게 지정합니다.
- __strides__: `int` 또는 2개의 `int`로 이루어진 튜플/리스트. 합성곱 필터의 스트라이드를 지정합니다. `int`하나를 입력할 경우 열방향, 행방향의 스트라이드를 동일하게 지정합니다. 기본값은 `(1, 1)`입니다. 만약 팽창 합성곱을 사용하고자 할 때 스트라이드의 크기를 `1`보다 크게 지정했다면 `dilation_rate`인자는 반드시 `1`로 맞춰야 합니다.
- __padding__: `str`. 입력값의 패딩처리 여부를 `"valid"`, `"causal"` 또는 `"same"` 가운데 하나로 지정합니다(대소문자 무관). `"valid"`는 패딩이 없는 경우, `"same"`은 출력의 형태를 입력과 같게 맞추고자 하는 경우에 사용합니다. `"same"`의 경우 `strides`가 `1`이 아닐 때, 사용하는 백엔드에 따라 값이 조금씩 달라질 수 있습니다 [참고](https://github.com/keras-team/keras/pull/9473#issuecomment-372166860).
- __data_format__: `str`. 입력 데이터의 차원 순서를 정의하는 인자로 `"channels_last"`(기본값) 또는 `"channels_first"` 가운데 하나를 지정합니다. 입력 형태가 `(batch, height, width, channels)`로 채널 정보가 마지막에 올 경우 `"channels_last"`를, `(batch, channels, height, width)`로 채널 정보가 먼저 올 경우 `"channels_first"`를 선택합니다. 케라스 설정 `~/.keras/keras.json`파일에 있는 `image_data_format`값을 기본값으로 사용하며, 해당 값이 없는 경우 자동으로 `"channels_last"`를 기본값으로 적용합니다. 
- __dilation_rate__: `int` 또는 2개의 `int`로 이루어진 튜플/리스트. 팽창 합성곱 필터의 팽창비율을 결정합니다. 팽창 합성곱은 원래 조밀한 형태 그대로 입력에 적용되는 합성곱 필터를 각 방향으로 원소 사이의 간격을 띄우는 방식으로 팽창시켜 성긴 대신 보다 넓은 영역에 적용될 수 있도록 변형한 합성곱입니다. 자세한 내용은 [Multi-Scale Context Aggregation by Dilated Convolutions](https://arxiv.org/abs/1511.07122v3)을 참고하십시오. 기본값은 `(1, 1)`이며, 현재 버전에서는 `dilation_rate`가 `1`보다 큰 경우 `1`보다 큰 `strides`를 지정할 수 없습니다. 
- __depth_multiplier__: 각 입력 채널당 몇 개의 출력 채널을 생성할 것인지 결정합니다. 출력 채널의 총 개수는 `filters_in * depth_multiplier`가 됩니다.
- __activation__: 사용할 활성화 함수입니다. 기본값은 `None`으로, 별도로 지정하지 않으면 전달할 경우 활성화 함수가 적용되지 않습니다 (`a(x) = x`). 참고: [활성화 함수](../activations.md)
- __use_bias__: `bool`, 층의 연산에 편향을 적용할지 여부를 결정합니다.
- __depthwise_initializer__: 깊이별 필터 행렬의 초기화 함수를 결정합니다. 참고: [초기화 함수](../initializers.md)
- __pointwise_initializer__: 위치별 필터 행렬의 초기화 함수를 결정합니다. 참고: [초기화 함수](../initializers.md)
- __bias_initializer__: 편향 벡터의 초기화 함수를 결정합니다. 참고: [초기화 함수](../initializers.md)
- __depthwise_regularizer__: 깊이별 필터 행렬에 적용할 규제 함수를 결정합니다. 참고: [규제 함수](../regularizers.md)
- __pointwise_regularizer__: 위치별 필터 행렬에 적용할 규제 함수를 결정합니다. 참고: [규제 함수](../regularizers.md)
- __bias_regularizer__: 편향 벡터에 적용할 규제 함수를 결정합니다. 참고: [규제 함수](../regularizers.md)
- __activity_regularizer__: 층의 출력값에 적용할 규제 함수를 결정합니다. 참고: [규제 함수](../regularizers.md)
- __depthwise_constraint__: 깊이별 필터 행렬에 적용할 제약을 결정합니다. 참고: [제약](../constraints.md)
- __pointwise_constraint__: 위치별 필터 행렬에 적용할 제약을 결정합니다. 참고: [제약](../constraints.md)
- __bias_constraint__: 편향 벡터에 적용할 제약을 결정합니다. 참고: [제약](../constraints.md)

__입력 형태__

`data_format`이 `"channels_first"`이면 `(batch, channels, rows, cols)` 형태의 4D 텐서.  
`data_format`이 `"channels_last"`이면 `(batch, rows, cols, channels)` 형태의 4D 텐서.  

__출력 형태__

`data_format`이 `"channels_first"`이면 `(batch, filters, new_rows, new_cols)` 형태의 4D 텐서.  
`data_format`이 `"channels_last"`이면 `(batch, new_rows, new_cols, filters)` 형태의 4D 텐서.  
패딩으로 인해 `rows`와 `cols` 값이 바뀔 수 있습니다.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L1694)</span>
### DepthwiseConv2D

```python
keras.layers.DepthwiseConv2D(kernel_size, strides=(1, 1), padding='valid', depth_multiplier=1, data_format=None, activation=None, use_bias=True, depthwise_initializer='glorot_uniform', bias_initializer='zeros', depthwise_regularizer=None, bias_regularizer=None, activity_regularizer=None, depthwise_constraint=None, bias_constraint=None)
```

깊이별 2D 합성곱.

깊이별 분리 합성곱의 첫 단계만을 수행하는 층입니다. 입력값의 각 채널(깊이)별로 따로 공간 차원의 합성곱을 수행합니다. 이때 `depth_multiplier`인자는 채널별 합성곱 단계에서 각 입력 채널당 몇 개의 출력 채널을 생성할 것인지 결정합니다.

__인자__

- __kernel_size__: `int` 또는 2개의 `int`로 이루어진 튜플/리스트. 2D 합성곱 필터의 행방향과 열방향 크기를 지정합니다. `int`하나를 입력할 경우 모든 방향의 크기를 동일하게 지정합니다.
- __strides__: `int` 또는 2개의 `int`로 이루어진 튜플/리스트. 합성곱 필터의 스트라이드를 지정합니다. `int`하나를 입력할 경우 열방향, 행방향의 스트라이드를 동일하게 지정합니다. 기본값은 `(1, 1)`입니다. 만약 팽창 합성곱을 사용하고자 할 때 스트라이드의 크기를 `1`보다 크게 지정했다면 `dilation_rate`인자는 반드시 `1`로 맞춰야 합니다.
- __padding__: `str`. 입력값의 패딩처리 여부를 `"valid"`, `"causal"` 또는 `"same"` 가운데 하나로 지정합니다(대소문자 무관). `"valid"`는 패딩이 없는 경우, `"same"`은 출력의 형태를 입력과 같게 맞추고자 하는 경우에 사용합니다. `"same"`의 경우 `strides`가 `1`이 아닐 때, 사용하는 백엔드에 따라 값이 조금씩 달라질 수 있습니다 [참고](https://github.com/keras-team/keras/pull/9473#issuecomment-372166860).
- __depth_multiplier__: 각 입력 채널당 몇 개의 출력 채널을 생성할 것인지 결정합니다. 출력 채널의 총 개수는 `filters_in * depth_multiplier`가 됩니다.
- __data_format__: `str`. 입력 데이터의 차원 순서를 정의하는 인자로 `"channels_last"`(기본값) 또는 `"channels_first"` 가운데 하나를 지정합니다. 입력 형태가 `(batch, height, width, channels)`로 채널 정보가 마지막에 올 경우 `"channels_last"`를, `(batch, channels, height, width)`로 채널 정보가 먼저 올 경우 `"channels_first"`를 선택합니다. 케라스 설정 `~/.keras/keras.json`파일에 있는 `image_data_format`값을 기본값으로 사용하며, 해당 값이 없는 경우 자동으로 `"channels_last"`를 기본값으로 적용합니다. 
- __activation__: 사용할 활성화 함수입니다. 기본값은 `None`으로, 별도로 지정하지 않으면 전달할 경우 활성화 함수가 적용되지 않습니다 (`a(x) = x`). 참고: [활성화 함수](../activations.md)
- __use_bias__: `bool`, 층의 연산에 편향을 적용할지 여부를 결정합니다.
- __depthwise_initializer__: 깊이별 필터 행렬의 초기화 함수를 결정합니다. 참고: [초기화 함수](../initializers.md)
- __bias_initializer__: 편향 벡터의 초기화 함수를 결정합니다. 참고: [초기화 함수](../initializers.md)
- __depthwise_regularizer__: 깊이별 필터 행렬에 적용할 규제 함수를 결정합니다. 참고: [규제 함수](../regularizers.md)
- __bias_regularizer__: 편향 벡터에 적용할 규제 함수를 결정합니다. 참고: [규제 함수](../regularizers.md)
- __activity_regularizer__: 층의 출력값에 적용할 규제 함수를 결정합니다. 참고: [규제 함수](../regularizers.md)
- __depthwise_constraint__: 깊이별 필터 행렬에 적용할 제약을 결정합니다. 참고: [제약](../constraints.md)
- __bias_constraint__: 편향 벡터에 적용할 제약을 결정합니다. 참고: [제약](../constraints.md)

__입력 형태__

`data_format`이 `"channels_first"`이면 `(batch, channels, rows, cols)` 형태의 4D 텐서.  
`data_format`이 `"channels_last"`이면 `(batch, rows, cols, channels)` 형태의 4D 텐서.  
  
__출력 형태__

`data_format`이 `"channels_first"`이면 `(batch, filters, new_rows, new_cols)` 형태의 4D 텐서.  
`data_format`이 `"channels_last"`이면 `(batch, new_rows, new_cols, filters)` 형태의 4D 텐서.  
패딩으로 인해 `rows`와 `cols` 값이 바뀔  수 있습니다.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L628)</span>
### Conv2DTranspose

```python
keras.layers.Conv2DTranspose(filters, kernel_size, strides=(1, 1), padding='valid', output_padding=None, data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

전치된<sub>transposed</sub> 합성곱 층(디컨볼루션<sub>deconvolution</sub>으로 불리기도 합니다).

일반적인 컨볼루션의 역방향으로 변환을 하고자 할 때 전치 합성곱을 사용합니다. 다시 말해, 어떤 특정한 합성곱 연산이 있을 때 그 출력 형태로부터 입력 형태를 향해 합성곱의 연결구조를 유지하면서 거슬러 올라가고자 할 때 쓸 수 있습니다.  
  
모델의 가장 처음에 올 경우 `input_shape`인자를 통해 입력값의 형태를 함께 지정해야 합니다. `input_shape`는 정수로 된 튜플로 배치 축<sub>axis</sub>은 포함시키지 않습니다. 예를 들어 `data_format="channels_last"`인 128x128 RGB 이미지의 경우 `input_shape=(128, 128, 3)`이 됩니다.

__인자__

- __filters__: `int`. 출력할 결과값의 차원으로 합성곱 필터의 갯수를 나타냅니다. 
- __kernel_size__: `int` 또는 2개의 `int`로 이루어진 튜플/리스트. 2D 합성곱 필터의 행방향과 열방향 크기를 지정합니다. `int`하나를 입력할 경우 모든 방향의 크기를 동일하게 지정합니다.
- __strides__: `int` 또는 2개의 `int`로 이루어진 튜플/리스트. 합성곱 필터의 스트라이드를 지정합니다. `int`하나를 입력할 경우 행방향과 열방향의 스트라이드를 동일하게 지정합니다. 기본값은 `(1, 1)`입니다. 만약 팽창 합성곱을 사용하고자 할 때 스트라이드의 크기를 `1`보다 크게 지정했다면 `dilation_rate`인자는 반드시 `1`로 맞춰야 합니다.
- __padding__: `str`. 입력값의 패딩처리 여부를 `"valid"` 또는 `"same"` 가운데 하나로 지정합니다(대소문자 무관). `"valid"`는 패딩이 없는 경우, `"same"`은 출력의 형태를 입력과 같게 맞추고자 하는 경우에 사용합니다. `"same"`의 경우 `strides`가 `1`이 아닐 때, 사용하는 백엔드에 따라 값이 조금씩 달라질 수 있습니다 [참고](https://github.com/keras-team/keras/pull/9473#issuecomment-372166860).
- __output_padding__: `int` 또는 2개의 `int`로 이루어진 튜플/리스트. 출력 텐서의 행방향과 열방향 패딩을 지정합니다. `int`하나를 입력할 경우 모든 방향의 크기를 동일하게 지정합니다. 각 차원에 주어진 출력 패딩 값의 크기는 같은 차원에서의 스트라이드 값보다 작아야 합니다. 기본값인 `None`으로 설정할 경우 출력 크기는 자동으로 유추됩니다. 
- __data_format__: `str`. 입력 데이터의 차원 순서를 정의하는 인자로 `"channels_last"`(기본값) 또는 `"channels_first"` 가운데 하나를 지정합니다. 입력 형태가 `(batch, height, width, channels)`로 채널 정보가 마지막에 올 경우 `"channels_last"`를, `(batch, channels, height, width)`로 채널 정보가 먼저 올 경우 `"channels_first"`를 선택합니다. 케라스 설정 `~/.keras/keras.json`파일에 있는 `image_data_format`값을 기본값으로 사용하며, 해당 값이 없는 경우 자동으로 `"channels_last"`를 기본값으로 적용합니다. 
- __dilation_rate__: `int` 또는 2개의 `int`로 이루어진 튜플/리스트. 팽창 합성곱 필터의 팽창비율을 결정합니다. 팽창 합성곱은 원래 조밀한 형태 그대로 입력에 적용되는 합성곱 필터를 각 방향으로 원소 사이의 간격을 띄우는 방식으로 팽창시켜 성긴 대신 보다 넓은 영역에 적용될 수 있도록 변형한 합성곱입니다. 자세한 내용은 [Multi-Scale Context Aggregation by Dilated Convolutions](https://arxiv.org/abs/1511.07122v3)을 참고하십시오. 기본값은 `(1, 1)`이며, 현재 버전에서는 `dilation_rate`가 `1`보다 큰 경우 `1`보다 큰 `strides`를 지정할 수 없습니다. 
- __activation__: 사용할 활성화 함수입니다. 기본값은 `None`으로, 별도로 지정하지 않으면 전달할 경우 활성화 함수가 적용되지 않습니다 (`a(x) = x`). 참고: [활성화 함수](../activations.md)
- __use_bias__: `bool`, 층의 연산에 편향을 적용할지 여부를 결정합니다.
- __kernel_initializer__: `kernel` 가중치 행렬의 초기화 함수를 결정합니다. 참고: [초기화 함수](../initializers.md)
- __bias_initializer__: 편향 벡터의 초기화 함수를 결정합니다. 참고: [초기화 함수](../initializers.md)
- __kernel_regularizer__: `kernel` 가중치 행렬에 적용할 규제 함수를 결정합니다. 참고: [규제 함수](../regularizers.md)
- __bias_regularizer__: 편향 벡터에 적용할 규제 함수를 결정합니다. 참고: [규제 함수](../regularizers.md)
- __activity_regularizer__: 층의 출력값에 적용할 규제 함수를 결정합니다. 참고: [규제 함수](../regularizers.md)
- __kernel_constraint__: `kernel` 가중치 행렬에 적용할 제약을 결정합니다. 참고: [제약](../constraints.md)
- __bias_constraint__: 편향 벡터에 적용할 제약을 결정합니다. 참고: [제약](../constraints.md)

__입력 형태__

`data_format`이 `"channels_first"`이면 `(batch, channels, rows, cols)` 형태의 4D 텐서.  
`data_format`이 `"channels_last"`이면 `(batch, rows, cols, channels)` 형태의 4D 텐서.  

__출력 형태__

`data_format`이 `"channels_first"`이면 `(batch, filters, new_rows, new_cols)` 형태의 4D 텐서.  
`data_format`이 `"channels_last"`이면 `(batch, new_rows, new_cols, filters)` 형태의 4D 텐서.  
패딩으로 인해 `rows`와 `cols` 값이 바뀔  수 있습니다.

`output_padding`을 지정한 경우는 다음 식을 따릅니다.
```
new_rows = ((rows - 1) * strides[0] + kernel_size[0]
            - 2 * padding[0] + output_padding[0])
new_cols = ((cols - 1) * strides[1] + kernel_size[1]
            - 2 * padding[1] + output_padding[1])
```

__참고__

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
3D 합성곱 층 (예: 부피에 대한 공간적 합성곱)

이 층은 입력 텐서와 합성곱되어 출력 텐서를 만드는 합성곱 커널을 생성합니다. `use_bias`가 참인 경우, 편향 <sub>bias</sub>벡터를 생성해 출력 텐서에 더합니다. `activation`이 `None`이 아닌 경우 이 또한 출력 텐서에 적용됩니다.  

모델의 가장 처음에 올 경우 `input_shape`인자를 통해 입력값의 형태를 함께 지정해야 합니다. `input_shape`는 정수로 된 튜플로 배치 축<sub>axis</sub>은 포함시키지 않습니다. 예를 들어 `data_format="channels_last"`이며 채널이 한 개인 128x128x128 입체의 경우 `input_shape=(128, 128, 128, 1)`이 됩니다.

__인자__

- __filters__: `int`. 출력할 결과값의 차원으로 합성곱 필터의 갯수를 나타냅니다. 
- __kernel_size__: `int` 또는 3개의 `int`로 이루어진 튜플/리스트. 3D 합성곱 필터의 깊이 및 행방향과 열방향 크기를 지정합니다. `int`하나를 입력할 경우 모든 방향의 크기를 동일하게 지정합니다.
- __strides__: `int` 또는 3개의 `int`로 이루어진 튜플/리스트. 합성곱 필터의 스트라이드를 지정합니다. `int`하나를 입력할 경우 모든 방향의 스트라이드를 동일하게 지정합니다. 기본값은 `(1, 1)`입니다. 만약 팽창 합성곱을 사용하고자 할 때 스트라이드의 크기를 `1`보다 크게 지정했다면 `dilation_rate`인자는 반드시 `1`로 맞춰야 합니다.
- __padding__: `str`. 입력값의 패딩처리 여부를 `"valid"` 또는 `"same"` 가운데 하나로 지정합니다(대소문자 무관). `"valid"`는 패딩이 없는 경우, `"same"`은 출력의 형태를 입력과 같게 맞추고자 하는 경우에 사용합니다. `"same"`의 경우 `strides`가 `1`이 아닐 때, 사용하는 백엔드에 따라 값이 조금씩 달라질 수 있습니다 [참고](https://github.com/keras-team/keras/pull/9473#issuecomment-372166860).
- __data_format__: `str`. 입력 데이터의 차원 순서를 정의하는 인자로 `"channels_last"`(기본값) 또는 `"channels_first"` 가운데 하나를 지정합니다. 입력 형태가 `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`로 채널 정보가 마지막에 올 경우 `"channels_last"`를, `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`로 채널 정보가 먼저 올 경우 `"channels_first"`를 선택합니다. 케라스 설정 `~/.keras/keras.json`파일에 있는 `image_data_format`값을 기본값으로 사용하며, 해당 값이 없는 경우 자동으로 `"channels_last"`를 기본값으로 적용합니다. 
- __dilation_rate__: `int` 또는 3개의 `int`로 이루어진 튜플/리스트. 팽창 합성곱 필터의 팽창비율을 결정합니다. 팽창 합성곱은 원래 조밀한 형태 그대로 입력에 적용되는 합성곱 필터를 각 원소 사이를 각 방향으로 원소 사이의 간격을 띄우는 방식으로 팽창시켜 성긴 대신 보다 넓은 영역에 적용될 수 있도록 변형한 합성곱입니다. 자세한 내용은 [Multi-Scale Context Aggregation by Dilated Convolutions](https://arxiv.org/abs/1511.07122v3)을 참고하십시오. 기본값은 `(1, 1)`이며, 현재 버전에서는 `dilation_rate`가 `1`보다 큰 경우 `1`보다 큰 `strides`를 지정할 수 없습니다. 
- __activation__: 사용할 활성화 함수입니다. 기본값은 `None`으로, 별도로 지정하지 않으면 전달할 경우 활성화 함수가 적용되지 않습니다 (`a(x) = x`). 참고: [활성화 함수](../activations.md)
- __use_bias__: `bool`, 층의 연산에 편향을 적용할지 여부를 결정합니다.
- __kernel_initializer__: `kernel` 가중치 행렬의 초기화 함수를 결정합니다. 참고: [초기화 함수](../initializers.md)
- __bias_initializer__: 편향 벡터의 초기화 함수를 결정합니다. 참고: [초기화 함수](../initializers.md)
- __kernel_regularizer__: `kernel` 가중치 행렬에 적용할 규제 함수를 결정합니다. 참고: [규제 함수](../regularizers.md)
- __bias_regularizer__: 편향 벡터에 적용할 규제 함수를 결정합니다. 참고: [규제 함수](../regularizers.md)
- __activity_regularizer__: 층의 출력값에 적용할 규제 함수를 결정합니다. 참고: [규제 함수](../regularizers.md)
- __kernel_constraint__: `kernel` 가중치 행렬에 적용할 제약을 결정합니다. 참고: [제약](../constraints.md)
- __bias_constraint__: 편향 벡터에 적용할 제약을 결정합니다. 참고: [제약](../constraints.md)


__입력 형태__

`data_format`이 `"channels_first"`이면 `(batch, channels, conv_dim1, conv_dim2, conv_dim3)` 형태의 5D 텐서.  
`data_format`이 `"channels_last"`이면 `(batch, conv_dim1, conv_dim2, conv_dim3, channels)` 형태의 5D 텐서.  

__아웃풋 형태__

`data_format`이 `"channels_first"`이면 `(batch, filters, new_conv_dim1, new_conv_dim2, new_conv_dim3)` 형태의 5D 텐서.  
`data_format`이 `"channels_last"`이면 `(batch, new_conv_dim1, new_conv_dim2, new_conv_dim3, filters)` 형태의 5D 텐서.  
패딩으로 인해 `new_conv_dim1`, `new_conv_dim2`와 `new_conv_dim3`값이 바뀔 수 있습니다.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L901)</span>
### Conv3DTranspose

```python
keras.layers.Conv3DTranspose(filters, kernel_size, strides=(1, 1, 1), padding='valid', output_padding=None, data_format=None, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```
전치된 합성곱 층(디컨볼루션으로 불리기도 합니다).

일반적인 컨볼루션의 역방향으로 변환을 하고자 할 때 전치 합성곱을 사용합니다. 다시 말해, 어떤 특정한 합성곱 연산이 있을 때 그 출력 형태로부터 입력 형태를 향해 합성곱의 연결구조를 유지하면서 거슬러 올라가고자 할 때 쓸 수 있습니다.  
  
모델의 가장 처음에 올 경우 `input_shape`인자를 통해 입력값의 형태를 함께 지정해야 합니다. `input_shape`는 정수로 된 튜플로 배치 축<sub>axis</sub>은 포함시키지 않습니다. 예를 들어 `data_format="channels_last"`이며 채널이 3개인 128x128x128 입체의 경우 `input_shape=(128, 128, 128, 3)`이 됩니다.

__인자__

- __filters__: `int`. 출력할 결과값의 차원으로 합성곱 필터의 갯수를 나타냅니다. 
- __kernel_size__: `int` 또는 3개의 `int`로 이루어진 튜플/리스트. 3D 합성곱 필터의 깊이 및 행방향과 열방향 크기를 지정합니다. `int`하나를 입력할 경우 모든 방향의 크기를 동일하게 지정합니다.
- __strides__: `int` 또는 3개의 `int`로 이루어진 튜플/리스트. 합성곱 필터의 스트라이드를 지정합니다. `int`하나를 입력할 경우 모든 방향의 스트라이드를 동일하게 지정합니다. 기본값은 `(1, 1)`입니다. 만약 팽창 합성곱을 사용하고자 할 때 스트라이드의 크기를 `1`보다 크게 지정했다면 `dilation_rate`인자는 반드시 `1`로 맞춰야 합니다.
- __padding__: `str`. 입력값의 패딩처리 여부를 `"valid"` 또는 `"same"` 가운데 하나로 지정합니다(대소문자 무관). `"valid"`는 패딩이 없는 경우, `"same"`은 출력의 형태를 입력과 같게 맞추고자 하는 경우에 사용합니다. `"same"`의 경우 `strides`가 `1`이 아닐 때, 사용하는 백엔드에 따라 값이 조금씩 달라질 수 있습니다 [참고](https://github.com/keras-team/keras/pull/9473#issuecomment-372166860).
- __output_padding__: `int` 또는 3개의 `int`로 이루어진 튜플/리스트. 출력 텐서의 깊이 및 행방향과 열방향 패딩을 지정합니다. `int`하나를 입력할 경우 모든 방향의 크기를 동일하게 지정합니다. 각 차원에 주어진 출력 패딩 값의 크기는 같은 차원에서의 스트라이드 값보다 작아야 합니다. 기본값인 `None`으로 설정할 경우 출력 크기는 자동으로 유추됩니다. 
- __data_format__: `str`. 입력 데이터의 차원 순서를 정의하는 인자로 `"channels_last"`(기본값) 또는 `"channels_first"` 가운데 하나를 지정합니다. 입력 형태가  `(batch, depth, height, width, channels)`로 채널 정보가 마지막에 올 경우 `"channels_last"`를, `(batch, channels, depth, height, width)`로 채널 정보가 먼저 올 경우 `"channels_first"`를 선택합니다. 케라스 설정 `~/.keras/keras.json`파일에 있는 `image_data_format`값을 기본값으로 사용하며, 해당 값이 없는 경우 자동으로 `"channels_last"`를 기본값으로 적용합니다. 
- __dilation_rate__: `int` 또는 3개의 `int`로 이루어진 튜플/리스트. 팽창 합성곱 필터의 팽창비율을 결정합니다. 팽창 합성곱은 원래 조밀한 형태 그대로 입력에 적용되는 합성곱 필터를 각 원소 사이를 각 방향으로 원소 사이의 간격을 띄우는 방식으로 팽창시켜 성긴 대신 보다 넓은 영역에 적용될 수 있도록 변형한 합성곱입니다. 자세한 내용은 [Multi-Scale Context Aggregation by Dilated Convolutions](https://arxiv.org/abs/1511.07122v3)을 참고하십시오. 기본값은 `(1, 1)`이며, 현재 버전에서는 `dilation_rate`가 `1`보다 큰 경우 `1`보다 큰 `strides`를 지정할 수 없습니다. 
- __activation__: 사용할 활성화 함수입니다. 기본값은 `None`으로, 별도로 지정하지 않으면 전달할 경우 활성화 함수가 적용되지 않습니다 (`a(x) = x`). 참고: [활성화 함수](../activations.md)
- __use_bias__: `bool`, 층의 연산에 편향을 적용할지 여부를 결정합니다.
- __kernel_initializer__: `kernel` 가중치 행렬의 초기화 함수를 결정합니다. 참고: [초기화 함수](../initializers.md)
- __bias_initializer__: 편향 벡터의 초기화 함수를 결정합니다. 참고: [초기화 함수](../initializers.md)
- __kernel_regularizer__: `kernel` 가중치 행렬에 적용할 규제 함수를 결정합니다. 참고: [규제 함수](../regularizers.md)
- __bias_regularizer__: 편향 벡터에 적용할 규제 함수를 결정합니다. 참고: [규제 함수](../regularizers.md)
- __activity_regularizer__: 층의 출력값에 적용할 규제 함수를 결정합니다. 참고: [규제 함수](../regularizers.md)
- __kernel_constraint__: `kernel` 가중치 행렬에 적용할 제약을 결정합니다. 참고: [제약](../constraints.md)
- __bias_constraint__: 편향 벡터에 적용할 제약을 결정합니다. 참고: [제약](../constraints.md)

__입력 형태__

`data_format`이 `"channels_first"`이면 `(batch, channels, depth, rows, cols)` 형태의 5D 텐서.  
`data_format`이 `"channels_last"`이면 `(batch, depth, rows, cols, channels)` 형태의 5D 텐서.  

__출력 형태__

`data_format`이 `"channels_first"`이면 `(batch, filters, new_depth, new_rows, new_cols)` 형태의 5D 텐서.  
`data_format`이 `"channels_last"`이면 `(batch, new_depth, new_rows, new_cols, filters)` 형태의 5D 텐서.  
패딩으로 인해, `depth`, `rows`와 `cols` 값이 달라졌을 수도 있습니다.  

`output_padding`을 지정한 경우는 다음 식을 따릅니다.
```
new_depth = ((depth - 1) * strides[0] + kernel_size[0]
             - 2 * padding[0] + output_padding[0])
new_rows = ((rows - 1) * strides[1] + kernel_size[1]
            - 2 * padding[1] + output_padding[1])
new_cols = ((cols - 1) * strides[2] + kernel_size[2]
            - 2 * padding[2] + output_padding[2])
```

__참고__

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

1D 입력(예: 시계열 데이터)용 크롭핑 층.

시간 차원(축 1)을 따라서 잘라냅니다.

__인자__

- __cropping__: `int` 혹은 2개의 `int`로 이루어진 튜플. 각각 크롭핑할 차원(첫번째 축)의 시작과 끝에서 얼마만큼을 잘라낼 것인지 정합니다. `int` 하나만 입력할 경우 시작과 끝 모두 같은 크기가 적용됩니다. 기본값은 `(1, 1)`입니다.

__입력 형태__

`(batch, axis_to_crop, features)` 형태의 3D 텐서.

__출력 형태__

`(batch, cropped_axis, features)` 형태의 3D 텐서.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L2408)</span>
### Cropping2D

```python
keras.layers.Cropping2D(cropping=((0, 0), (0, 0)), data_format=None)
```

2D 입력(예: 이미지)용 크롭핑 층.

공간 차원(높이와 넓이)을 따라서 잘라냅니다.

__인자__

- __cropping__: `int`, 2개의 `int`로 이루어진 튜플, 혹은 2개의 `int`로 이루어진 튜플 2개로 이루어진 튜플.
    - 하나의 `int`인 경우 행방향과 열방향에 동일한 크기의 대칭 크롭핑이 적용됩니다.
    - 2개 `int`의 튜플인 경우 각각 상하, 좌우에 똑같이 잘라낼 행과 열 값을 나타냅니다 `(symmetric_height_crop, symmetric_width_crop)`.
    - 2개 `int`의 튜플 2개로 이루어진 튜플인 경우 `((top_crop, bottom_crop), (left_crop, right_crop))`을 나타냅니다.
- __data_format__: `str`. 입력 데이터의 차원 순서를 정의하는 인자로 `"channels_last"`(기본값) 또는 `"channels_first"` 가운데 하나를 지정합니다. 입력 형태가 `(batch, height, width, channels)`로 채널 정보가 마지막에 올 경우 `"channels_last"`를, `(batch, channels, height, width)`로 채널 정보가 먼저 올 경우 `"channels_first"`를 선택합니다. 케라스 설정 `~/.keras/keras.json`파일에 있는 `image_data_format`값을 기본값으로 사용하며, 해당 값이 없는 경우 자동으로 `"channels_last"`를 기본값으로 적용합니다. 

__입력 형태__

다음과 같은 형태의 4D 텐서를 입력합니다.
- `data_format`이 `"channels_last"`이면 `(batch, rows, cols, channels)`.
- `data_format`이 `"channels_first"`이면 `(batch, channels, rows, cols)`.

__출력 형태__

다음 형태의 4D 텐서를 출력합니다.
- `data_format`이 `"channels_last"`이면 `(batch, cropped_rows, cropped_cols, channels)`.
- `data_format`이 `"channels_first"`이면 `(batch, channels, cropped_rows, cropped_cols)`.

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

3D 입력(예: 공간적 혹은 시공간적)용 크롭핑 층.

__인자__

- __cropping__: `int`, 3개의 `int`로  튜플, 혹은 2개 정수의 튜플 3개로 이루어진 튜플.
    - 하나의 `int`인 경우 깊이, 행방향, 열방향에 동일한 크기의 크롭핑이 적용됩니다.
    - 3개 `int`의 튜플인 경우 각각 깊이, 행방향, 에 적용될 값을 나타냅니다. `(symmetric_dim1_crop, symmetric_dim2_crop, symmetric_dim3_crop)`.
    - 2개 `int`의 튜플 3개로 이루어진 튜플인 경우 `((left_dim1_crop, right_dim1_crop), (left_dim2_crop, right_dim2_crop), (left_dim3_crop, right_dim3_crop))`을 나타냅니다.        
- __data_format__: `str`. 입력 데이터의 차원 순서를 정의하는 인자로 `"channels_last"`(기본값) 또는 `"channels_first"` 가운데 하나를 지정합니다. 입력 형태가 `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`로 채널 정보가 마지막에 올 경우 `"channels_last"`를, `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`로 채널 정보가 먼저 올 경우 `"channels_first"`를 선택합니다. 케라스 설정 `~/.keras/keras.json`파일에 있는 `image_data_format`값을 기본값으로 사용하며, 해당 값이 없는 경우 자동으로 `"channels_last"`를 기본값으로 적용합니다. 

__입력 형태__

다음 형태의 5D 텐서를 입력합니다.
- `data_format`이 `"channels_last"`이면 `(batch, first_axis_to_crop, second_axis_to_crop, third_axis_to_crop, depth)`.
- `data_format`이 `"channels_first"`이면 `(batch, depth, first_axis_to_crop, second_axis_to_crop, third_axis_to_crop)`.

__출력 형태__

다음 형태의 5D 텐서를 출력합니다.
- `data_format`이 `"channels_last"`이면 `(batch, first_cropped_axis, second_cropped_axis, third_cropped_axis, depth)`.
- `data_format`이 `"channels_first"`이면 `(batch, depth, first_cropped_axis, second_cropped_axis, third_cropped_axis)`.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L1944)</span>
### UpSampling1D

```python
keras.layers.UpSampling1D(size=2)
```

1D 입력용 업샘플링 층.

시간 축을 따라 각 시간 단계를 `size`로 지정한 만큼 반복합니다.

__인자__

- __size__: `int`. 반복할 횟수입니다.

__입력 형태__

`(batch, steps, features)` 형태의 3D 텐서를 입력합니다.

__출력 형태__

`(batch, upsampled_steps, features)` 형태의 3D 텐서를 출력합니다.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L1974)</span>
### UpSampling2D

```python
keras.layers.UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest')
```

2D 입력용 업샘플링 층.

데이터의 행과 열을 각각 `size[0]`과 `size[1]`회씩 반복합니다.

__인자__

- __size__: `int`, 혹은 2개의 `int`로 이루어진 튜플. 행과 열에서 반복할 횟수입니다.
- __data_format__: `str`. 입력 데이터의 차원 순서를 정의하는 인자로 `"channels_last"`(기본값) 또는 `"channels_first"` 가운데 하나를 지정합니다. 입력 형태가 `(batch, height, width, channels)`로 채널 정보가 마지막에 올 경우 `"channels_last"`를, `(batch, channels, height, width)`로 채널 정보가 먼저 올 경우 `"channels_first"`를 선택합니다. 케라스 설정 `~/.keras/keras.json`파일에 있는 `image_data_format`값을 기본값으로 사용하며, 해당 값이 없는 경우 자동으로 `"channels_last"`를 기본값으로 적용합니다. 
- __interpolation__: `str`. `nearest` 또는 `bilinear`를 지정합니다. `bilinear` 업스케일링의 경우 CNTK는 아직 지원하지 않으며, `Theano`의 경우 `size=(2, 2)`에 한해서 지원합니다.

__입력 형태__

다음 형태의 4D 텐서를 입력합니다.
- `data_format`이 `"channels_last"`이면 `(batch, rows, cols, channels)`.
- `data_format`이 `"channels_first"`이면 `(batch, channels, rows, cols)`.

__출력 형태__

다음 형태의 4D 텐서를 출력합니다.
- `data_format`이 `"channels_last"`이면 `(batch, upsampled_rows, upsampled_cols, channels)`.
- `data_format`이 `"channels_first"`이면 `(batch, channels, upsampled_rows, upsampled_cols)`.


----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L2032)</span>
### UpSampling3D

```python
keras.layers.UpSampling3D(size=(2, 2, 2), data_format=None)
```

3D 입력용 업샘플링 층.

데이터의 첫 번째, 두 번째, 세 번째 차원을 각각 `size[0]`, `size[1]`, `size[2]`회씩 반복합니다.

__인자__
- __size__: `int`, 혹은 3개의 `int`로 이루어진 튜플. 각각 첫 번째, 두 번째, 세 번째 차원에서 반복할 횟수입니다.
- __data_format__: `str`. 입력 데이터의 차원 순서를 정의하는 인자로 `"channels_last"`(기본값) 또는 `"channels_first"` 가운데 하나를 지정합니다. 입력 형태가 `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`로 채널 정보가 마지막에 올 경우 `"channels_last"`를, `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`로 채널 정보가 먼저 올 경우 `"channels_first"`를 선택합니다. 케라스 설정 `~/.keras/keras.json`파일에 있는 `image_data_format`값을 기본값으로 사용하며, 해당 값이 없는 경우 자동으로 `"channels_last"`를 기본값으로 적용합니다. 

__입력 형태__

다음 형태의 5D 텐서를 입력합니다.
- `data_format`이 `"channels_last"`이면 `(batch, dim1, dim2, dim3, channels)`.
- `data_format`이 `"channels_first"`이면 `(batch, channels, dim1, dim2, dim3)`.

__출력 형태__

다음 형태의 5D 텐서를 출력합니다.
- `data_format`이 `"channels_last"`이면 `(batch, upsampled_dim1, upsampled_dim2, upsampled_dim3, channels)`.
- `data_format`이 `"channels_first"`이면 `(batch, channels, upsampled_dim1, upsampled_dim2, upsampled_dim3)`.
    
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L2124)</span>
### ZeroPadding1D

```python
keras.layers.ZeroPadding1D(padding=1)
```

1D 입력(예: 시계열 데이터)용 제로 패딩 층.

__인자__

- __padding__: `int`, 혹은 2개의 `int`로 이루어진 튜플/딕셔너리.
    - 하나의 `int`인 경우 적용할 차원(축 1)의 시작과 끝에 채울 `0` 값의 개수.
    - 2개의 `int` 튜플인 경우  각각 시작과 끝에(`(left_pad, right_pad)`) 채울 `0` 값의 개수.

__입력 형태__

`(batch, axis_to_pad, features)` 형태의 3D 텐서.

__출력 형태__

`(batch, padded_axis, features)` 형태의 3D 텐서.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L2159)</span>
### ZeroPadding2D

```python
keras.layers.ZeroPadding2D(padding=(1, 1), data_format=None)
```

2D 입력(예: 이미지 데이터)용 제로-패딩 층.

이 레이어는 이미지 텐서의 상하좌우에 `0`으로 이루어진 행과 열을 더할 수 있습니다.

__인자__

- __padding__: `int`, 2개의 `int`로 이루어진 튜플, 혹은 2개의 `int`로 이루어진 튜플 2개로 이루어진 튜플.
    - 하나의 `int`인 경우 행방향과 열방향에 같은 크기의 패딩이 적용됩니다.
    - 2개의 `int` 튜플인 경우 각각 행방향, 열방향에 적용될 값을 나타냅니다. `(symmetric_height_pad, symmetric_width_pad)`.
    - 2개 정수의 튜플 2개로 이루어진 튜플인 경우 `((top_pad, bottom_pad), (left_pad, right_pad))`에 적용될 값을 나타냅니다.
- __data_format__: `str`. 입력 데이터의 차원 순서를 정의하는 인자로 `"channels_last"`(기본값) 또는 `"channels_first"` 가운데 하나를 지정합니다. 입력 형태가 `(batch, height, width, channels)`로 채널 정보가 마지막에 올 경우 `"channels_last"`를, `(batch, channels, height, width)`로 채널 정보가 먼저 올 경우 `"channels_first"`를 선택합니다. 케라스 설정 `~/.keras/keras.json`파일에 있는 `image_data_format`값을 기본값으로 사용하며, 해당 값이 없는 경우 자동으로 `"channels_last"`를 기본값으로 적용합니다. 

__입력 형태__

다음 형태의 4D 텐서를 입력합니다.
- `data_format`이 `"channels_last"`이면 `(batch, rows, cols, channels)`.
- `data_format`이 `"channels_last"`이면 `(batch, channels, rows, cols)`.

__출력 형태__

다음 형태의 4D 텐서를 입력합니다.
- `data_format`이 `"channels_last"`이면 `(batch, padded_rows, padded_cols, channels)`.
- `data_format`이 `"channels_last"`이면 `(batch, channels, padded_rows, padded_cols)`.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L2235)</span>
### ZeroPadding3D

```python
keras.layers.ZeroPadding3D(padding=(1, 1, 1), data_format=None)
```

3D 입력(예: 공간적, 시공간적 데이터)용 제로 패딩 층.

__인자__

- __padding__: `int`, 3개의 `int`로  튜플, 혹은 2개 정수의 튜플 3개로 이루어진 튜플.
    - 하나의 `int`인 경우 깊이, 행방향, 열방향에 동일한 크기의 패딩이 적용됩니다.
    - 3개 `int`의 튜플인 경우 각각 깊이, 행방향, 열방향에 적용될 값을 나타냅니다. `(symmetric_dim1_pad, symmetric_dim2_pad, symmetric_dim3_pad)`.    
    - 2개 `int`의 튜플 3개로 이루어진 튜플인 경우 `((left_dim1_pad, right_dim1_pad), (left_dim2_pad, right_dim2_pad), (left_dim3_pad, right_dim3_pad))`를 나타냅니다.
- __data_format__: `str`. 입력 데이터의 차원 순서를 정의하는 인자로 `"channels_last"`(기본값) 또는 `"channels_first"` 가운데 하나를 지정합니다. 입력 형태가 `(batch, height, width, channels)`로 채널 정보가 마지막에 올 경우 `"channels_last"`를, `(batch, channels, height, width)`로 채널 정보가 먼저 올 경우 `"channels_first"`를 선택합니다. 케라스 설정 `~/.keras/keras.json`파일에 있는 `image_data_format`값을 기본값으로 사용하며, 해당 값이 없는 경우 자동으로 `"channels_last"`를 기본값으로 적용합니다. 

__입력 형태__

다음 형태의 5D 텐서를 입력합니다.
- `data_format`이 `"channels_last"`이면 `(batch, first_axis_to_pad, second_axis_to_pad, third_axis_to_pad, depth)`.
- `data_format`이 `"channels_first"`이면 `(batch, depth, first_axis_to_pad, second_axis_to_pad, third_axis_to_pad)`.

__출력 형태__

다음 형태의 5D 텐서를 출력합니다.
- `data_format`이 `"channels_last"`이면 `(batch, first_padded_axis, second_padded_axis, third_axis_to_pad, depth)`.
- `data_format`이 `"channels_first"`이면 `(batch, depth, first_padded_axis, second_padded_axis, third_axis_to_pad)`.  

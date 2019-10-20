<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L237)</span>
### RNN

```python
keras.layers.RNN(cell, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)
```

순환 신경망<sub>Recurrent Neural Network</sub> 층<sub>layer</sub>의 기본 클래스.

__인자__
- __cell__: 순환 신경망 내부의 셀 인스턴스입니다. 순환 신경망은 특정한 연산을 입력된 시계열 길이만큼 반복하는 형태의 신경망입니다. 셀은 이 반복되는 연산 부분을 담당하는 영역으로 생성한 순환 신경망 인스턴스의 `cell`속성에 할당됩니다. 셀은 다음과 같은 하위 요소들을 가진 클래스입니다.
    - `t`시점의 입력<sub>input</sub>과 상태<sub>state</sub>를 불러오는 `call`메소드(`call(input_at_t, states_at_t)`). `t`시점의 출력<sub>output</sub>과 `t+1`시점의 상태`(output_at_t, states_at_t_plus_1)`를 반환합니다. 셀 인스턴스의 `call`메소드는 필요한 경우 `constants`인자를 이용하여 별도의 원하는 상수값을 입력받을 수 있습니다. 자세한 내용은 아래 "별도의 상수 전달 시의 유의점"을 참고하십시오.
    - `state_size` 속성. 신경망 안에서 단계마다 전달되는 상태의 크기를 나타냅니다(셀의 출력과 크기가 같아야 합니다). 셀이 가지게 될 상태가 하나인 경우 하나의 정수를, 여럿인 경우 정수로 이루어진 리스트/튜플을 입력받습니다.   
    - `output_size` 속성. 출력값의 크기를 나태냅니다. 정수 혹은 `TensorShape`를 입력받습니다. 만약 해당 속성이 없는 경우, `state_size`의 첫번째 값으로부터 유추한 결과로 대신합니다.
    
        여러 셀의 인스턴스를 층층이 쌓고자 하는 경우 셀 인스턴스의 리스트를 `cell`로 지정할 수 있습니다. 적층형 순환 신경망<sub>stacked RNN</sub>을 적용할 때 유용합니다. 
- __return_sequences__: `bool`. 시계열<sub>sequence</sub> 가운데 모든 시점의 출력값을 반환할지 마지막 시점의 출력값만을 반환할지 결정합니다. 기본값은 `False`입니다. 
- __return_state__: `bool`. 출력과 함께 마지막 시점의 상태값도 반환할지의 여부를 결정합니다. 기본값은 `False`입니다.
- __go_backwards__: `bool`. 기본값은 `False`입니다. `True`인 경우 입력의 순서를 뒤집어 거꾸로된 순서의 처리 결과를 반환합니다. 
- __stateful__: `bool`. 기본값은 `False`입니다. `True`인 경우 현재 입력된 배치의 각 인덱스 `i`에 해당하는 입력값의 마지막 상태가 다음 배치의 각 인덱스 `i`에 해당하는 입력값의 초기 상태로 사용됩니다. 매우 긴 시계열을 연속된 배치로 나누어 처리할 때 유용합니다.  
- __unroll__: `bool`. 기본값은 `False`입니다. `True`인 경우 순환구조로 처리되는 신경망을 펼쳐서 연산의 일부를 동시에 처리합니다. 이 경우 전체를 순환구조로 처리하는 것보다 빠른 연산이 가능하지만 그만큼 많은 정보를 동시에 저장해야 하기 때문에 메모리 소모가 커집니다. 시계열 길이가 짧은 경우 적합합니다. 
- __input_dim__: `int`. 입력값 가운데 요인<sub>feature</sub>들로 이루어진 차원의 크기를 지정합니다. 이 인자(또는 `input_shape`인자)는 해당 층이 모델의 입력을 직접 다루는 첫 번째 층으로 사용되는 경우에만 요구됩니다.
- __input_length__: 입력값의 시계열(순서형) 길이. 모든 배치가 단일한 길이를 가지는 경우 하나의 상수값을 배정합니다. 이후 `Flatten`에 이어 `Dense`층을 연결하려면 이 인자가 필요합니다(이 인자가 주어지지 않은 경우, `Dense`층의 출력을 계산할 수 없습니다). 해당 층이 모델의 첫 번째 층이 아닌 경우에는 첫 번째 층의 인자에서(예: `input_shape`) 길이를 지정해야 합니다.

__입력 형태__
`(batch_size, timesteps, input_dim)` 형태의 3D 텐서.

__출력 형태__
- `return_state=True`인 경우: 텐서의 리스트. 첫 번째 텐서는 출력, 나머지 텐서는 마지막 시점의 상태값으로 각각 `(batch_size, units)`의 형태를 갖습니다. RNN이나 GRU의 경우 1개, LSTM의 경우 2개의 상태값을 반환합니다. 
- `return_sequences=True`인 경우: `(batch_size, timesteps, units)`형태의 3D 텐서.
- 그 외의 경우: `(batch_size, units)` 형태의 2D 텐서.

__마스킹__  
RNN 층은 시계열의 길이가 서로 다른 입력 데이터를 패딩해서 길이를 통일한 경우에 대한 마스킹을 지원합니다. 마스킹을 적용하려면 ‘[Embedding](embeddings.md)층에서 `mask_zero=True`로 설정합니다.

__순환 신경망의 상태 저장 모드 사용에 대한 유의점__

순환 신경망 층을 상태 저장<sub>stateful</sub> 모드로 설정할 경우 한 배치 내 표본들의 마지막 상태를 계산하여 이를 다음 배치 내 표본들의 초기상태로 사용하게 됩니다. 이전 배치의 인덱스가 다음 배치의 인덱스와 1:1로 이어져야 하므로, 상태 저장을 사용하려면 모델의 배치 크기를 통일해야 합니다.  

상태 저장 순환 신경망을 사용하려면 다음과 같이 설정합니다.
- 층을 생성할 때 `stateful=True`로 지정합니다.
- 모델의 배치 크기를 고정합니다. Sequential 모델의 경우 첫 층에서 `batch_input_shape=()`을, 함수형 모델의 경우 데이터 입력을 받는 모든 첫 번째 층에서 `batch_shape=()`을 사용하여 입력할 배치의 형태<sub>shape</sub>를 지정합니다. 이 경우 지정할 값은 (배치 크기, 시계열 길이, 입력값의 차원)으로 이루어진 정수 튜플입니다(예: `(32, 10, 100)`).

저장된 상태를 초기화하고자 한다면 특정 층에 대해서는 `layer.reset_states()`, 모델 전체에 대해서는 `model.reset_states()`를 사용합니다.

__순환 신경망 초기 상태 특정 시의 유의점__

`initial_state`인자를 사용하면 심볼릭 텐서를 이용하여 순환 신경망 층의 초기 상태를 원하는 값으로 설정할 수 있습니다. 이때 설정에 사용할 값은 해당 순환 신경망 층이 요구하는 초기 상태와 같은 형태의 텐서나 텐서의 리스트여야 합니다.  

`reset_states` 호출시에 `states`인자를 사용하면 NumPy 배열<sub>array</sub>을 이용하여 순환 신경망 층의 초기 상태를 원하는 값으로 설정할 수 있습니다. 이때 설정에 사용할 값은 해당 순환 신경망 층이 요구하는 초기 상태와 같은 형태의 NumPy 배열이나 배열의 리스트여야 합니다.  

__순환 신경망에 별도의 상수 전달 시의 유의점__

`RNN.__call__`(혹은 `RNN.call`) 메소드에 `constants` 인자를 지정하면 층의 입력값 외에도 별도의 상수를 셀에 입력할 수 있습니다. 이를 사용하기 위해서는 층 내부에서 작동하는 `cell.call` 메소드도 동일한 `constants`입력을 받도록 정의되어 있어야 합니다. `constants`인자를 이용하면 어텐션 메커니즘에서 필요한 것과 같이 셀 수준에서 외부로부터 독립적인 상수를 받아서 셀 내의 계산에 적용시키는 유형의 연산을 할 수 있습니다. 


__예시__

```python
# 우선 순환 신경망 셀을 레이어 하위 클래스로 정의해 봅시다.

class MinimalRNNCell(keras.layers.Layer):

    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(MinimalRNNCell, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        h = K.dot(inputs, self.kernel)
        output = h + K.dot(prev_output, self.recurrent_kernel)
        return output, [output]

# 이 셀을 순환 신경망 레이어에 적용해 봅시다.

cell = MinimalRNNCell(32)
x = keras.Input((None, 5))
layer = RNN(cell)
y = layer(x)

# 다음은 상태 저장 순환 신경망을 구성하기 위해 셀을 어떻게 사용하는지 보여줍니다.

cells = [MinimalRNNCell(32), MinimalRNNCell(64)]
x = keras.Input((None, 5))
layer = RNN(cells)
y = layer(x)
```

----
<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L945)</span>
### SimpleRNN

```python
keras.layers.SimpleRNN(units, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)
```

이전 시점의 출력을 현 시점의 입력으로 받는 완전 연결<sub>fully-connected</sub> 순환 신경망.

__인자__
- __units__: 양의 정수. 출력값의 차원 크기를 결정합니다.
- __activation__: 사용할 활성화 함수입니다. 기본값은 하이퍼볼릭탄젠트(`tanh`)이며 `None`을 전달할 경우 활성화 함수가 적용되지 않습니다(`a(x) = x`). 참고: [활성화 함수](../activations.md)
- __use_bias__: `bool`. 층의 연산에 편향<sub>bias</bias>을 적용할지 여부를 결정합니다.
- __kernel_initializer__: `kernel` 가중치<sub>weights</sub> 행렬의 초기화 함수를 결정합니다. 이 가중치는 입력값에 곱해져서 선형변환하는 연산에 사용됩니다. 참고: [초기화 함수](../initializers.md)
- __recurrent_initializer__: `recurrent_kernel` 가중치 행렬의 초기화 함수를 결정합니다. 이 가중치는 이전 시점으로부터 전달받은 상태값에 곱해져서 선형변환하는 연산에 사용됩니다. 참고: [초기화 함수](../initializers.md)
- __bias_initializer__: 편향 벡터의 초기화 함수를 결정합니다. 참고: [초기화 함수](../initializers.md)
- __kernel_regularizer__: `kernel` 가중치 행렬에 적용할 규제 함수<sub>regularizer</sub>를 결정합니다. 참고: [규제 함수](../regularizers.md)
- __recurrent_regularizer__: `recurrent_kernel` 가중치 행렬에 적용할 규제 함수를 결정합니다. 참고: [규제 함수](../regularizers.md)
- __bias_regularizer__: 편향 벡터에 적용할 규제 함수를 결정합니다. 참고: [규제 함수](../regularizers.md)
- __activity_regularizer__: 층의 출력값에 적용할 규제 함수를 결정합니다. 참고: [규제 함수](../regularizers.md)
- __kernel_constraint__: `kernel` 가중치 행렬에 적용할 제약<sub>constraints</sub>을 결정합니다. 참고: [제약](../constraints.md))
- __recurrent_constraint__: `recurrent_kernel` 가중치 행렬에 적용할 제약을 결정합니다. 참고: [제약](../constraints.md))
- __bias_constraint__: 편향 벡터에 적용할 제약을 결정합니다. 참고: [제약](../constraints.md))
- __dropout__: `0`과 `1`사이의 `float`. 입력값의 선형 변환에 사용되는 `kernel` 가중치의 값 가운데 지정한 만큼의 비율을 무작위로 `0`으로 바꾸어 탈락시킵니다.
- __recurrent_dropout__: `0`과 `1`사이의 `float`. 상태값의 선형 변환에 사용되는 `recurrent_kernel` 가중치의 값 가운데 지정한 만큼의 비율을 무작위로 `0`으로 바꾸어 탈락시킵니다.
- __return_sequences__: `bool`. 시계열 가운데 모든 시점의 출력값을 반환할지 마지막 시점의 출력값만을 반환할지 결정합니다. 기본값은 `False`입니다. 
- __return_state__: `bool`. 출력과 함께 마지막 시점의 상태값도 반환할지의 여부를 결정합니다. 기본값은 `False`입니다.
- __go_backwards__: `bool`. 기본값은 `False`입니다. `True`인 경우 입력의 순서를 뒤집어 거꾸로된 순서의 처리 결과를 반환합니다. 
- __stateful__: `bool`. 기본값은 `False`입니다. `True`인 경우 현재 입력된 배치의 각 인덱스 `i`에 해당하는 입력값의 마지막 상태가 다음 배치의 각 인덱스 `i`에 해당하는 입력값의 초기 상태로 사용됩니다. 매우 긴 시계열을 연속된 배치로 나누어 처리할 때 유용합니다.  
- __unroll__: `bool`. 기본값은 `False`입니다. `True`인 경우 순환구조로 처리되는 신경망을 펼쳐서 연산의 일부를 동시에 처리합니다. 이 경우 전체를 순환구조로 처리하는 것보다 빠른 연산이 가능하지만 그만큼 많은 정보를 동시에 저장해야 하기 때문에 메모리 소모가 커집니다. 시계열 길이가 짧은 경우 적합합니다.  


----
<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L1491)</span>
### GRU

```python
keras.layers.GRU(units, activation='tanh', recurrent_activation='sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=2, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False, reset_after=False)
```

Gated Recurrent Unit - Cho et al. 2014.

케라스가 지원하는 GRU 버전은 두 가지입니다. 기본값은 후기 버전인 1406.1078v3을 바탕으로 만든 것으로 이전 시점에서 전달받은 은닉 상태<sub>hidden state</sub>에 가중치(`recurrent_kernel`)를 곱하기 전에 리셋 게이트가 상태에 먼저 적용되는 식을 따릅니다. 나머지는 초기 버전인 1406.1078v1을 바탕으로 만든 것으로 먼저 상태에 가중치를 곱한 다음에 리셋 게이트를 적용합니다.  

이 가운데 `CuDNNGRU`(오직 GPU만을 사용하는 연산)와 CPU연산을 모두 지원하는 것은 초기 버전입니다. 이 버전에서는 `kernel`과 `recurrent_kernel` 계산이 별도로 이루어지기 때문에 사용되는 편향 벡터 역시 별개로 존재합니다. 초기 버전을 사용하려면 `reset_after=True`로, `recuurrent_activation='sigmoid'`로 설정하십시오.

__인자__

- __units__: 양의 정수. 출력값의 차원 크기를 결정합니다.
- __activation__: GRU에서 현재 시점 입력 계산에 사용할 활성화 함수입니다. 기본값은 하이퍼볼릭탄젠트(`tanh`)이며 `None`을 전달할 경우 활성화 함수가 적용되지 않습니다(`a(x) = x`). 참고: [활성화 함수](../activations.md)
- __recurrent_activation__: GRU의 업데이트 게이트와 리셋 게이트 계산에 사용할 활성화 함수입니다. 기본값은 시그모이드(`'sigmoid'`)이며 `None`을 전달할 경우 활성화 함수가 적용되지 않습니다(`a(x) = x`). 참고: [활성화 함수](../activations.md)
- __use_bias__: `bool`. 층의 연산에 편향을 적용할지 여부를 결정합니다.
- __kernel_initializer__: `kernel` 가중치 행렬의 초기화 함수를 결정합니다. 이 가중치는 입력값에 곱해져서 선형변환하는 연산에 사용됩니다. 참고: [초기화 함수](../initializers.md)
- __recurrent_initializer__: `recurrent_kernel` 가중치 행렬의 초기화 함수를 결정합니다. 이 가중치는 이전 시점으로부터 전달받은 상태값에 곱해져서 선형변환하는 연산에 사용됩니다. 참고: [초기화 함수](../initializers.md)
- __bias_initializer__: 편향 벡터의 초기화 함수를 결정합니다. 참고: [초기화 함수](../initializers.md)
- __kernel_regularizer__: `kernel` 가중치 행렬에 적용할 규제 함수를 결정합니다. 참고: [규제 함수](../regularizers.md)
- __recurrent_regularizer__: `recurrent_kernel` 가중치 행렬에 적용할 규제 함수를 결정합니다. 참고: [규제 함수](../regularizers.md)
- __bias_regularizer__: 편향 벡터에 적용할 규제 함수를 결정합니다. 참고: [규제 함수](../regularizers.md)
- __activity_regularizer__: 층의 출력값에 적용할 규제 함수를 결정합니다. 참고: [규제 함수](../regularizers.md)
- __kernel_constraint__: `kernel` 가중치 행렬에 적용할 제약을 결정합니다. 참고: [제약](../constraints.md))
- __recurrent_constraint__: `recurrent_kernel` 가중치 행렬에 적용할 제약을 결정합니다. 참고: [제약](../constraints.md))
- __bias_constraint__: 편향 벡터에 적용할 제약을 결정합니다. 참고: [제약](../constraints.md))
- __dropout__: `0`과 `1`사이의 `float`. 입력값의 선형 변환에 사용되는 `kernel` 가중치의 값 가운데 지정한 만큼의 비율을 무작위로 `0`으로 바꾸어 탈락시킵니다.
- __recurrent_dropout__: `0`과 `1`사이의 `float`. 상태값의 선형 변환에 사용되는 `recurrent_kernel` 가중치의 값 가운데 지정한 만큼의 비율을 무작위로 `0`으로 바꾸어 탈락시킵니다.
- __implementation__: `1` 또는 `2`. 연산 모드를 설정합니다. `1`은 작은 크기의 내적과 덧셈을 많이 하는 구성, `2`는 큰 크기의 내적과 덧셈을 보다 적은 횟수로 하는 구성입니다. 이러한 설정은 하드웨어 및 어플리케이션에 따라 서로 다른 연산 성능을 가져옵니다. 기본값은 `2`입니다.
- __return_sequences__: `bool`. 시계열 가운데 모든 시점의 출력값을 반환할지 마지막 시점의 출력값만을 반환할지 결정합니다. 기본값은 `False`입니다. 
- __return_state__: `bool`. 출력과 함께 마지막 시점의 상태값도 반환할지의 여부를 결정합니다. 기본값은 `False`입니다.
- __go_backwards__: `bool`. 기본값은 `False`입니다. `True`인 경우 입력의 순서를 뒤집어 거꾸로된 순서의 처리 결과를 반환합니다. 
- __stateful__: `bool`. 기본값은 `False`입니다. `True`인 경우 현재 입력된 배치의 각 인덱스 `i`에 해당하는 입력값의 마지막 상태가 다음 배치의 각 인덱스 `i`에 해당하는 입력값의 초기 상태로 사용됩니다. 매우 긴 시계열을 연속된 배치로 나누어 처리할 때 유용합니다.  
- __unroll__: `bool`. 기본값은 `False`입니다. `True`인 경우 순환구조로 처리되는 신경망을 펼쳐서 연산의 일부를 동시에 처리합니다. 이 경우 전체를 순환구조로 처리하는 것보다 빠른 연산이 가능하지만 그만큼 많은 정보를 동시에 저장해야 하기 때문에 메모리 소모가 커집니다. 시계열 길이가 짧은 경우 적합합니다.  
- __reset_after__: 리셋 게이트를 은닉 상태에 적용하는 시점을 지정합니다. `False`인 경우 `recurrent_kernel`을 곱하기 전에 적용하며, `True`인 경우 `recurrent_kernel`을 곱한 다음 그 결과에 적용합니다(`CuDNN`과 호환되는 방식입니다). 

__참고__

- [Learning Phrase Representations using RNN Encoder-Decoder for
   Statistical Machine Translation](https://arxiv.org/abs/1406.1078)
- [On the Properties of Neural Machine Translation:
   Encoder-Decoder Approaches](https://arxiv.org/abs/1409.1259)
- [Empirical Evaluation of Gated Recurrent Neural Networks on
   Sequence Modeling](https://arxiv.org/abs/1412.3555v1)
- [A Theoretically Grounded Application of Dropout in
   Recurrent Neural Networks](https://arxiv.org/abs/1512.05287)


----
<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L2051)</span>
### LSTM

```python
keras.layers.LSTM(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)
```

Long Short-Term Memory - Hochreiter 1997.

__인자__

- __units__: 양의 정수. 출력값의 차원 크기를 결정합니다.
- __activation__: LSTM에서 현재 시점 입력을 바탕으로 후보값<sub>candidate value</sub>을 계산하는 과정 및 셀 상태<sub>csll state</sub>를 이용하여 현재 시점의 출력값을 계산하는 과정에서 사용할 활성화 함수입니다. 기본값은 하이퍼볼릭탄젠트(`tanh`)이며 `None`을 전달할 경우 활성화 함수가 적용되지 않습니다(`a(x) = x`). 참고: [활성화 함수](../activations.md)
- __recurrent_activation__: LSTM의 인풋 게이트와 포겟 게이트, 아웃풋 게이트 계산에 사용할 활성화 함수입니다. 기본값은 하드 시그모이드(`'hard_sigmoid'`)이며 `None`을 전달할 경우 활성화 함수가 적용되지 않습니다(`a(x) = x`). 참고: [활성화 함수](../activations.md)
- __use_bias__: `bool`. 층의 연산에 편향을 적용할지 여부를 결정합니다.
- __kernel_initializer__: `kernel` 가중치 행렬의 초기화 함수를 결정합니다. 이 가중치는 입력값에 곱해져서 선형변환하는 연산에 사용됩니다. 참고: [초기화 함수](../initializers.md)
- __recurrent_initializer__: `recurrent_kernel` 가중치 행렬의 초기화 함수를 결정합니다. 이 가중치는 이전 시점으로부터 전달받은 상태값에 곱해져서 선형변환하는 연산에 사용됩니다. 참고: [초기화 함수](../initializers.md)
- __bias_initializer__: 편향 벡터의 초기화 함수를 결정합니다. 참고: [초기화 함수](../initializers.md)
- __unit_forget_bias__: `bool`. `True`인 경우 [Jozefowicz et al. (2015)](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)의 제안에 따라 포겟 게이트의 편향에 `1`을 더합니다. 또한 강제적으로 `bias_initializer='zeros'`로 설정하여 나머지 게이트의 편향을 `0`으로 시작하게끔 합니다. 기본값은 `True`입니다.
- __kernel_regularizer__: `kernel` 가중치 행렬에 적용할 규제 함수를 결정합니다. 참고: [규제 함수](../regularizers.md)
- __recurrent_regularizer__: `recurrent_kernel` 가중치 행렬에 적용할 규제 함수를 결정합니다. 참고: [규제 함수](../regularizers.md)
- __bias_regularizer__: 편향 벡터에 적용할 규제 함수를 결정합니다. 참고: [규제 함수](../regularizers.md)
- __activity_regularizer__: 층의 출력값에 적용할 규제 함수를 결정합니다. 참고: [규제 함수](../regularizers.md)
- __kernel_constraint__: `kernel` 가중치 행렬에 적용할 제약을 결정합니다. 참고: [제약](../constraints.md))
- __recurrent_constraint__: `recurrent_kernel` 가중치 행렬에 적용할 제약을 결정합니다. 참고: [제약](../constraints.md))
- __bias_constraint__: 편향 벡터에 적용할 제약을 결정합니다. 참고: [제약](../constraints.md))
- __dropout__: `0`과 `1`사이의 `float`. 입력값의 선형 변환에 사용되는 `kernel` 가중치의 값 가운데 지정한 만큼의 비율을 무작위로 `0`으로 바꾸어 탈락시킵니다.
- __recurrent_dropout__: `0`과 `1`사이의 `float`. 상태값의 선형 변환에 사용되는 `recurrent_kernel` 가중치의 값 가운데 지정한 만큼의 비율을 무작위로 `0`으로 바꾸어 탈락시킵니다.
- __implementation__: `1` 또는 `2`. 연산 모드를 설정합니다. `1`은 작은 크기의 내적과 덧셈을 많이 하는 구성, `2`는 큰 크기의 내적과 덧셈을 보다 적은 횟수로 하는 구성입니다. 이러한 설정은 하드웨어 및 어플리케이션에 따라 서로 다른 연산 성능을 가져옵니다. 기본값은 `2`입니다.
- __return_sequences__: `bool`. 시계열 가운데 모든 시점의 출력값을 반환할지 마지막 시점의 출력값만을 반환할지 결정합니다. 기본값은 `False`입니다. 
- __return_state__: `bool`. 출력과 함께 마지막 시점의 상태값도 반환할지의 여부를 결정합니다. 기본값은 `False`입니다.
- __go_backwards__: `bool`. 기본값은 `False`입니다. `True`인 경우 입력의 순서를 뒤집어 거꾸로된 순서의 처리 결과를 반환합니다. 
- __stateful__: `bool`. 기본값은 `False`입니다. `True`인 경우 현재 입력된 배치의 각 인덱스 `i`에 해당하는 입력값의 마지막 상태가 다음 배치의 각 인덱스 `i`에 해당하는 입력값의 초기 상태로 사용됩니다. 매우 긴 시계열을 연속된 배치로 나누어 처리할 때 유용합니다.  
- __unroll__: `bool`. 기본값은 `False`입니다. `True`인 경우 순환구조로 처리되는 신경망을 펼쳐서 연산의 일부를 동시에 처리합니다. 이 경우 전체를 순환구조로 처리하는 것보다 빠른 연산이 가능하지만 그만큼 많은 정보를 동시에 저장해야 하기 때문에 메모리 소모가 커집니다. 시계열 길이가 짧은 경우 적합합니다.  

__참고__

- [Long short-term memory](
  http://www.bioinf.jku.at/publications/older/2604.pdf)
- [Learning to forget: Continual prediction with LSTM](
  http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
- [Supervised sequence labeling with recurrent neural networks](
  http://www.cs.toronto.edu/~graves/preprint.pdf)
- [A Theoretically Grounded Application of Dropout in
   Recurrent Neural Networks](https://arxiv.org/abs/1512.05287)
 

----
<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional_recurrent.py#L788)</span>
### ConvLSTM2D

```python
keras.layers.ConvLSTM2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=False, go_backwards=False, stateful=False, dropout=0.0, recurrent_dropout=0.0)
```

합성곱<sub>convolutional</sub> LSTM 신경망.

LSTM과 비슷하지만 입력값과 상태의 변환에 합성곱이 적용됩니다.

__인자__

- __filters__: `int`. 출력할 결과값의 차원으로 합성곱 필터의 개수를 나타냅니다. 
- __kernel_size__: `int` 또는 `int`로 이루어진 튜플/리스트. 합성곱 필터의 크기를 지정합니다.
- __strides__: `int` 또는 `int`로 이루어진 튜플/리스트. 합성곱 필터의 스트라이드를 지정합니다. 기본값은 `(1, 1)`입니다. 만약 팽창 합성곱<sub>dilated convolution</sub>을 사용하고자 할 때 스트라이드의 크기를 `1`보다 크게 지정했다면 `dilation_rate`인자는 반드시 `1`로 맞춰야 합니다.
- __padding__: `str`. 입력값의 패딩처리 여부를 `'valid'` 또는 `'same'` 가운데 하나로 지정합니다(대소문자 무관). `'valid'`는 패딩이 없는 경우, `'same'`은 출력의 형태를 입력과 같게 맞추고자 하는 경우에 사용합니다.
- __data_format__: `str`. 입력 데이터의 차원 순서를 정의하는 인자로 `'channels_last'`(기본값) 또는 `'channels_first'` 가운데 하나를 지정합니다. 입력 형태가 `(batch, time, ..., channels)`로 채널 정보가 마지막에 올 경우 `'channels_last'`를, `(batch, time, channels, ...)`로 채널 정보가 먼저 올 경우 `'channels_first'`를 선택합니다. 케라스 설정 `~/.keras/keras.json`파일에 있는 `image_data_format`값을 기본값으로 사용하며, 해당 값이 없는 경우 자동으로 `'channels_last'`를 기본값으로 적용합니다. 
- __dilation_rate__: `int` 또는 `int`로 이루어진 튜플/리스트. 팽창 합성곱 필터의 팽창비율을 결정합니다. 팽창 합성곱은 원래 조밀한 형태 그대로 입력에 적용되는 합성곱 필터를 각 원소 사이에 가로, 세로 방향으로 간격을 띄우는 방식으로 팽창시켜 성긴 대신 보다 넓은 영역에 적용될 수 있도록 변형한 합성곱입니다. 자세한 내용은 [Multi-Scale Context Aggregation by Dilated Convolutions](https://arxiv.org/abs/1511.07122v3)을 참고하십시오. 기본값은 `(1, 1)`이며, 현재 버전에서는 `dilation_rate`가 `1`보다 큰 경우 `1`보다 큰 `strides`를 지정할 수 없습니다. 
- __activation__: LSTM에서 현재 시점 입력을 바탕으로 후보값을 계산하는 과정 및 셀 상태를 이용하여 현재 시점의 출력값을 계산하는 과정에서 사용할 활성화 함수입니다. 기본값은 하이퍼볼릭탄젠트(`tanh`)이며 `None`을 전달할 경우 활성화 함수가 적용되지 않습니다(`a(x) = x`). 참고: [활성화 함수](../activations.md)
- __recurrent_activation__: LSTM의 인풋 게이트와 포겟 게이트, 아웃풋 게이트 계산에 사용할 활성화 함수입니다. 기본값은 하드 시그모이드(`'hard_sigmoid'`)이며 `None`을 전달할 경우 활성화 함수가 적용되지 않습니다(`a(x) = x`). 참고: [활성화 함수](../activations.md)
- __use_bias__: `bool`. 층의 연산에 편향을 적용할지 여부를 결정합니다.
- __kernel_initializer__: `kernel` 가중치 행렬의 초기화 함수를 결정합니다. 이 가중치는 입력값에 곱해져서 선형변환하는 연산에 사용됩니다. 참고: [초기화 함수](../initializers.md)
- __recurrent_initializer__: `recurrent_kernel` 가중치 행렬의 초기화 함수를 결정합니다. 이 가중치는 이전 시점으로부터 전달받은 상태값에 곱해져서 선형변환하는 연산에 사용됩니다. 참고: [초기화 함수](../initializers.md)
- __bias_initializer__: 편향 벡터의 초기화 함수를 결정합니다. 참고: [초기화 함수](../initializers.md)
- __unit_forget_bias__: `bool`. `True`인 경우 [Jozefowicz et al. (2015)](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)의 제안에 따라 포겟 게이트의 편향에 `1`을 더합니다. 또한 강제적으로 `bias_initializer='zeros'`로 설정하여 나머지 게이트의 편향을 `0`으로 시작하게끔 합니다. 기본값은 `True`입니다.
- __kernel_regularizer__: `kernel` 가중치 행렬에 적용할 규제 함수를 결정합니다. 참고: [규제 함수](../regularizers.md)
- __recurrent_regularizer__: `recurrent_kernel` 가중치 행렬에 적용할 규제 함수를 결정합니다. 참고: [규제 함수](../regularizers.md)
- __bias_regularizer__: 편향 벡터에 적용할 규제 함수를 결정합니다. 참고: [규제 함수](../regularizers.md)
- __activity_regularizer__: 층의 출력값에 적용할 규제 함수를 결정합니다. 참고: [규제 함수](../regularizers.md)
- __kernel_constraint__: `kernel` 가중치 행렬에 적용할 제약을 결정합니다. 참고: [제약](../constraints.md))
- __recurrent_constraint__: `recurrent_kernel` 가중치 행렬에 적용할 제약을 결정합니다. 참고: [제약](../constraints.md))
- __bias_constraint__: 편향 벡터에 적용할 제약을 결정합니다. 참고: [제약](../constraints.md))
- __return_sequences__: `bool`. 시계열 가운데 모든 시점의 출력값을 반환할지 마지막 시점의 출력값만을 반환할지 결정합니다. 기본값은 `False`입니다. 
- __go_backwards__: `bool`. 기본값은 `False`입니다. `True`인 경우 입력의 순서를 뒤집어 거꾸로된 순서의 처리 결과를 반환합니다. 
- __stateful__: `bool`. 기본값은 `False`입니다. `True`인 경우 현재 입력된 배치의 각 인덱스 `i`에 해당하는 입력값의 마지막 상태가 다음 배치의 각 인덱스 `i`에 해당하는 입력값의 초기 상태로 사용됩니다. 매우 긴 시계열을 연속된 배치로 나누어 처리할 때 유용합니다.  
- __dropout__: `0`과 `1`사이의 `float`. 입력값의 선형 변환에 사용되는 `kernel` 가중치의 값 가운데 지정한 만큼의 비율을 무작위로 `0`으로 바꾸어 탈락시킵니다.
- __recurrent_dropout__: `0`과 `1`사이의 `float`. 상태값의 선형 변환에 사용되는 `recurrent_kernel` 가중치의 값 가운데 지정한 만큼의 비율을 무작위로 0으로 바꾸어 탈락시킵니다.

__입력 형태__

- `data_format='channels_first'`의 경우
    다음과 같은 형태의 5D 텐서: `(samples, time, channels, rows, cols)`
- `data_format='channels_last'`의 경우
    다음과 같은 형태의 5D 텐서: `(samples, time, rows, cols, channels)`

__출력 형태__

- `return_sequences=True`의 경우
     - `data_format='channels_first'`이면
        다음과 같은 형태의 5D 텐서:
        `(samples, time, filters, output_row, output_col)`
     - `data_format='channels_last'`이면
        다음과 같은 형태의 5D 텐서:
        `(samples, time, output_row, output_col, filters)`
- 그 외의 경우
    - `data_format='channels_first'`이면
        다음과 같은 형태의 4D 텐서:
        `(samples, filters, output_row, output_col)`
    - `data_format='channels_last'`이면
        다음과 같은 형태의 4D 텐서:
        `(samples, output_row, output_col, filters)`

    `output_row`와 `output_col`의 값은 필터의 형태와 패딩에 따라 달라집니다.

__오류__
- __ValueError__: 유효하지 않은 인자를 전달받는 경우 발생합니다.

__참고__

- [Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting](http://arxiv.org/abs/1506.04214v1)
  현재 구현방식은 셀의 출력에 대한 피드백 루프를 포함하지 않습니다.


----
<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L780)</span>
### SimpleRNNCell

```python
keras.layers.SimpleRNNCell(units, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0)
```

SimpleRNN의 셀 클래스.

__인자__

- __units__: 양의 정수. 출력값의 차원 크기를 결정합니다.
- __activation__: 사용할 활성화 함수입니다. 기본값은 하이퍼볼릭탄젠트(`tanh`)이며 `None`을 전달할 경우 활성화 함수가 적용되지 않습니다(`a(x) = x`). 참고: [활성화 함수](../activations.md)
- __use_bias__: `bool`. 층의 연산에 편향을 적용할지 여부를 결정합니다.
- __kernel_initializer__: `kernel` 가중치 행렬의 초기화 함수를 결정합니다. 이 가중치는 입력값에 곱해져서 선형변환하는 연산에 사용됩니다. 참고: [초기화 함수](../initializers.md)
- __recurrent_initializer__: `recurrent_kernel` 가중치 행렬의 초기화 함수를 결정합니다. 이 가중치는 이전 시점으로부터 전달받은 상태값에 곱해져서 선형변환하는 연산에 사용됩니다. 참고: [초기화 함수](../initializers.md)
- __bias_initializer__: 편향 벡터의 초기화 함수를 결정합니다. 참고: [초기화 함수](../initializers.md)
- __kernel_regularizer__: `kernel` 가중치 행렬에 적용할 규제 함수를 결정합니다. 참고: [규제 함수](../regularizers.md)
- __recurrent_regularizer__: `recurrent_kernel` 가중치 행렬에 적용할 규제 함수를 결정합니다. 참고: [규제 함수](../regularizers.md)
- __bias_regularizer__: 편향 벡터에 적용할 규제 함수를 결정합니다. 참고: [규제 함수](../regularizers.md)
- __kernel_constraint__: `kernel` 가중치 행렬에 적용할 제약을 결정합니다. 참고: [제약](../constraints.md))
- __recurrent_constraint__: `recurrent_kernel` 가중치 행렬에 적용할 제약을 결정합니다. 참고: [제약](../constraints.md))
- __bias_constraint__: 편향 벡터에 적용할 제약을 결정합니다. 참고: [제약](../constraints.md))
- __dropout__: `0`과 `1`사이의 `float`. 입력값의 선형 변환에 사용되는 `kernel` 가중치의 값 가운데 지정한 만큼의 비율을 무작위로 `0`으로 바꾸어 탈락시킵니다.
- __recurrent_dropout__: `0`과 `1`사이의 `float`. 상태값의 선형 변환에 사용되는 `recurrent_kernel` 가중치의 값 가운데 지정한 만큼의 비율을 무작위로 `0`으로 바꾸어 탈락시킵니다.

    
----
<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L1164)</span>
### GRUCell

```python
keras.layers.GRUCell(units, activation='tanh', recurrent_activation='sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=2, reset_after=False)
```

GRU의 셀 클래스.

__인자__

- __units__: 양의 정수. 출력값의 차원 크기를 결정합니다.
- __activation__: GRU에서 현재 시점 입력 계산에 사용할 활성화 함수입니다. 기본값은 하이퍼볼릭탄젠트(`tanh`)이며 `None`을 전달할 경우 활성화 함수가 적용되지 않습니다(`a(x) = x`). 참고: [활성화 함수](../activations.md)
- __recurrent_activation__: GRU의 업데이트 게이트와 리셋 게이트 계산에 사용할 활성화 함수입니다. 기본값은 시그모이드(`'sigmoid'`)이며 `None`을 전달할 경우 활성화 함수가 적용되지 않습니다(`a(x) = x`). 참고: [활성화 함수](../activations.md)
- __use_bias__: `bool`. 층의 연산에 편향을 적용할지 여부를 결정합니다.
- __kernel_initializer__: `kernel` 가중치 행렬의 초기화 함수를 결정합니다. 이 가중치는 입력값에 곱해져서 선형변환하는 연산에 사용됩니다. 참고: [초기화 함수](../initializers.md)
- __recurrent_initializer__: `recurrent_kernel` 가중치 행렬의 초기화 함수를 결정합니다. 이 가중치는 이전 시점으로부터 전달받은 상태값에 곱해져서 선형변환하는 연산에 사용됩니다. 참고: [초기화 함수](../initializers.md)
- __bias_initializer__: 편향 벡터의 초기화 함수를 결정합니다. 참고: [초기화 함수](../initializers.md)
- __kernel_regularizer__: `kernel` 가중치 행렬에 적용할 규제 함수를 결정합니다. 참고: [규제 함수](../regularizers.md)
- __recurrent_regularizer__: `recurrent_kernel` 가중치 행렬에 적용할 규제 함수를 결정합니다. 참고: [규제 함수](../regularizers.md)
- __bias_regularizer__: 편향 벡터에 적용할 규제 함수를 결정합니다. 참고: [규제 함수](../regularizers.md)
- __kernel_constraint__: `kernel` 가중치 행렬에 적용할 제약을 결정합니다. 참고: [제약](../constraints.md))
- __recurrent_constraint__: `recurrent_kernel` 가중치 행렬에 적용할 제약을 결정합니다. 참고: [제약](../constraints.md))
- __bias_constraint__: 편향 벡터에 적용할 제약을 결정합니다. 참고: [제약](../constraints.md))
- __dropout__: `0`과 `1`사이의 `float`. 입력값의 선형 변환에 사용되는 `kernel` 가중치의 값 가운데 지정한 만큼의 비율을 무작위로 `0`으로 바꾸어 탈락시킵니다.
- __recurrent_dropout__: `0`과 `1`사이의 `float`. 상태값의 선형 변환에 사용되는 `recurrent_kernel` 가중치의 값 가운데 지정한 만큼의 비율을 무작위로 `0`으로 바꾸어 탈락시킵니다.
- __implementation__: `1` 또는 `2`. 연산 모드를 설정합니다. `1`은 작은 크기의 내적과 덧셈을 많이 하는 구성, `2`는 큰 크기의 내적과 덧셈을 보다 적은 횟수로 하는 구성입니다. 이러한 설정은 하드웨어 및 어플리케이션에 따라 서로 다른 연산 성능을 가져옵니다. 기본값은 `2`입니다.
- __return_sequences__: `bool`. 시계열 가운데 모든 시점의 출력값을 반환할지 마지막 시점의 출력값만을 반환할지 결정합니다. 기본값은 `False`입니다. 
- __return_state__: `bool`. 출력과 함께 마지막 시점의 상태값도 반환할지의 여부를 결정합니다. 기본값은 `False`입니다.
- __go_backwards__: `bool`. 기본값은 `False`입니다. `True`인 경우 입력의 순서를 뒤집어 거꾸로된 순서의 처리 결과를 반환합니다. 
- __stateful__: `bool`. 기본값은 `False`입니다. `True`인 경우 현재 입력된 배치의 각 인덱스 `i`에 해당하는 입력값의 마지막 상태가 다음 배치의 각 인덱스 `i`에 해당하는 입력값의 초기 상태로 사용됩니다. 매우 긴 시계열을 연속된 배치로 나누어 처리할 때 유용합니다.  
- __unroll__: `bool`. 기본값은 `False`입니다. `True`인 경우 순환구조로 처리되는 신경망을 펼쳐서 연산의 일부를 동시에 처리합니다. 이 경우 전체를 순환구조로 처리하는 것보다 빠른 연산이 가능하지만 그만큼 많은 정보를 동시에 저장해야 하기 때문에 메모리 소모가 커집니다. 시계열 길이가 짧은 경우 적합합니다.  
- __reset_after__: 리셋 게이트를 은닉 상태에 적용하는 시점을 지정합니다. `False`인 경우 `recurrent_kernel`을 곱하기 전에 적용하며, `True`인 경우 `recurrent_kernel`을 곱한 다음 그 결과에 적용합니다(`CuDNN`과 호환되는 방식입니다). 


----
<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L1765)</span>
### LSTMCell

```python
keras.layers.LSTMCell(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1)
```

LSTM의 셀 클래스.

__인자__

- __units__: 양의 정수, 출력값의 차원 크기를 결정합니다.
- __activation__: LSTM에서 현재 시점 입력을 바탕으로 후보값을 계산하는 과정 및 셀 상태를 이용하여 현재 시점의 출력값을 계산하는 과정에서 사용할 활성화 함수입니다. 기본값은 하이퍼볼릭탄젠트(`tanh`)이며 `None`을 전달할 경우 활성화 함수가 적용되지 않습니다(`a(x) = x`). 참고: [활성화 함수](../activations.md)
- __recurrent_activation__: LSTM의 인풋 게이트와 포겟 게이트, 아웃풋 게이트 계산에 사용할 활성화 함수입니다. 기본값은 하드 시그모이드(`'hard_sigmoid'`)이며 `None`을 전달할 경우 활성화 함수가 적용되지 않습니다(`a(x) = x`). 참고: [활성화 함수](../activations.md)
- __use_bias__: `bool`. 층의 연산에 편향을 적용할지 여부를 결정합니다.
- __kernel_initializer__: `kernel` 가중치 행렬의 초기화 함수를 결정합니다. 이 가중치는 입력값에 곱해져서 선형변환하는 연산에 사용됩니다. 참고: [초기화 함수](../initializers.md)
- __recurrent_initializer__: `recurrent_kernel` 가중치 행렬의 초기화 함수를 결정합니다. 이 가중치는 이전 시점으로부터 전달받은 상태값에 곱해져서 선형변환하는 연산에 사용됩니다. 참고: [초기화 함수](../initializers.md)
- __bias_initializer__: 편향 벡터의 초기화 함수를 결정합니다. 참고: [초기화 함수](../initializers.md)
- __unit_forget_bias__: `bool`. `True`인 경우 [Jozefowicz et al. (2015)](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)의 제안에 따라 포겟 게이트의 편향에 `1`을 더합니다. 또한 강제적으로 `bias_initializer='zeros'`로 설정하여 나머지 게이트의 편향을 `0`으로 시작하게끔 합니다. 기본값은 `True`입니다.
- __kernel_regularizer__: `kernel` 가중치 행렬에 적용할 규제 함수를 결정합니다. 참고: [규제 함수](../regularizers.md)
- __recurrent_regularizer__: `recurrent_kernel` 가중치 행렬에 적용할 규제 함수를 결정합니다. 참고: [규제 함수](../regularizers.md)
- __bias_regularizer__: 편향 벡터에 적용할 규제 함수를 결정합니다. 참고: [규제 함수](../regularizers.md)
- __kernel_constraint__: `kernel` 가중치 행렬에 적용할 제약을 결정합니다. 참고: [제약](../constraints.md))
- __recurrent_constraint__: `recurrent_kernel` 가중치 행렬에 적용할 제약을 결정합니다. 참고: [제약](../constraints.md))
- __bias_constraint__: 편향 벡터에 적용할 제약을 결정합니다. 참고: [제약](../constraints.md))
- __dropout__: `0`과 `1`사이의 `float`. 입력값의 선형 변환에 사용되는 `kernel` 가중치의 값 가운데 지정한 만큼의 비율을 무작위로 `0`으로 바꾸어 탈락시킵니다.
- __recurrent_dropout__: `0`과 `1`사이의 `float`. 상태값의 선형 변환에 사용되는 `recurrent_kernel` 가중치의 값 가운데 지정한 만큼의 비율을 무작위로 `0`으로 바꾸어 탈락시킵니다.
- __implementation__: `1`또는 `2`. 연산 모드를 설정합니다. `1`은 작은 크기의 내적과 덧셈을 많이 하는 구성, `2`는 큰 크기의 내적과 덧셈을 보다 적은 횟수로 하는 구성입니다. 이러한 설정은 하드웨어 및 어플리케이션에 따라 서로 다른 연산 성능을 가져옵니다. 기본값은 `2`입니다.


----
<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/cudnn_recurrent.py#L135)</span>
### CuDNNGRU

```python
keras.layers.CuDNNGRU(units, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=False, return_state=False, stateful=False)
```

[CuDNN](https://developer.nvidia.com/cudnn)을 사용한 빠른 GRU 구현 층. 

TensorFlow 백엔드로 CuDNN을 지원하는 GPU에서만 실행할 수 있습니다.

__인자__

- __units__: 양의 정수. 출력값의 차원 크기를 결정합니다..
- __kernel_initializer__: `kernel` 가중치 행렬의 초기화 함수를 결정합니다. 이 가중치는 입력값에 곱해져서 선형변환하는 연산에 사용됩니다. 참고: [초기화 함수](../initializers.md)
- __recurrent_initializer__: `recurrent_kernel` 가중치 행렬의 초기화 함수를 결정합니다. 이 가중치는 이전 시점으로부터 전달받은 상태값에 곱해져서 선형변환하는 연산에 사용됩니다. 참고: [초기화 함수](../initializers.md)
- __bias_initializer__: 편향 벡터의 초기화 함수를 결정합니다. 참고: [초기화 함수](../initializers.md)
- __kernel_regularizer__: `kernel` 가중치 행렬에 적용할 규제 함수를 결정합니다. 참고: [규제 함수](../regularizers.md)
- __recurrent_regularizer__: `recurrent_kernel` 가중치 행렬에 적용할 규제 함수를 결정합니다. 참고: [규제 함수](../regularizers.md)
- __bias_regularizer__: 편향 벡터에 적용할 규제 함수를 결정합니다. 참고: [규제 함수](../regularizers.md)
- __activity_regularizer__: 층의 출력값에 적용할 규제 함수를 결정합니다. 참고: [규제 함수](../regularizers.md)
- __kernel_constraint__: `kernel` 가중치 행렬에 적용할 제약을 결정합니다. 참고: [제약](../constraints.md))
- __recurrent_constraint__: `recurrent_kernel` 가중치 행렬에 적용할 제약을 결정합니다. 참고: [제약](../constraints.md))
- __bias_constraint__: 편향 벡터에 적용할 제약을 결정합니다. 참고: [제약](../constraints.md))
- __return_sequences__: `bool`. 시계열 가운데 모든 시점의 출력값을 반환할지 마지막 시점의 출력값만을 반환할지 결정합니다. 기본값은 `False`입니다. 
- __return_state__: `bool`. 출력과 함께 마지막 시점의 상태값도 반환할지의 여부를 결정합니다. 기본값은 `False`입니다.
- __stateful__: `bool`. 기본값은 `False`입니다. `True`인 경우 현재 입력된 배치의 각 인덱스 `i`에 해당하는 입력값의 마지막 상태가 다음 배치의 각 인덱스 `i`에 해당하는 입력값의 초기 상태로 사용됩니다. 매우 긴 시계열을 연속된 배치로 나누어 처리할 때 유용합니다.  

    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/cudnn_recurrent.py#L328)</span>
### CuDNNLSTM

```python
keras.layers.CuDNNLSTM(units, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=False, return_state=False, stateful=False)
```

[CuDNN](https://developer.nvidia.com/cudnn)을 사용한 빠른 LSTM 구현 층.

TensorFlow 백엔드로 CuDNN을 지원하는 GPU에서만 실행할 수 있습니다.

__인자__

- __units__: 양의 정수. 출력값의 차원 크기를 결정합니다.
- __kernel_initializer__: `kernel` 가중치 행렬의 초기화 함수를 결정합니다. 이 가중치는 입력값에 곱해져서 선형변환하는 연산에 사용됩니다. 참고: [초기화 함수](../initializers.md)
- __unit_forget_bias__: `bool`. `True`인 경우 [Jozefowicz et al. (2015)](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)의 제안에 따라 포겟 게이트의 편향에 `1`을 더합니다. 또한 강제적으로 `bias_initializer="zeros"`로 설정하여 나머지 게이트의 편향을 `0`으로 시작하게끔 합니다. 기본값은 `True`입니다.
- __recurrent_initializer__: `recurrent_kernel` 가중치 행렬의 초기화 함수를 결정합니다. 이 가중치는 이전 시점으로부터 전달받은 상태값에 곱해져서 선형변환하는 연산에 사용됩니다. 참고: [초기화 함수](../initializers.md)
- __bias_initializer__: 편향 벡터의 초기화 함수를 결정합니다. 참고: [초기화 함수](../initializers.md)
- __kernel_regularizer__: `kernel` 가중치 행렬에 적용할 규제 함수를 결정합니다. 참고: [규제 함수](../regularizers.md)
- __recurrent_regularizer__: `recurrent_kernel` 가중치 행렬에 적용할 규제 함수를 결정합니다. 참고: [규제 함수](../regularizers.md)
- __bias_regularizer__: 편향 벡터에 적용할 규제 함수를 결정합니다. 참고: [규제 함수](../regularizers.md)
- __activity_regularizer__: 층의 출력값에 적용할 규제 함수를 결정합니다. 참고: [규제 함수](../regularizers.md)
- __kernel_constraint__: `kernel` 가중치 행렬에 적용할 제약을 결정합니다. 참고: [제약](../constraints.md))
- __recurrent_constraint__: `recurrent_kernel` 가중치 행렬에 적용할 제약을 결정합니다. 참고: [제약](../constraints.md))
- __bias_constraint__: 편향 벡터에 적용할 제약을 결정합니다. 참고: [제약](../constraints.md))
- __return_sequences__: `bool`. 시계열 가운데 모든 시점의 출력값을 반환할지 마지막 시점의 출력값만을 반환할지 결정합니다. 기본값은 `False`입니다. 
- __return_state__: `bool`. 출력과 함께 마지막 시점의 상태값도 반환할지의 여부를 결정합니다. 기본값은 `False`입니다.
- __stateful__: `bool`. 기본값은 `False`입니다. `True`인 경우 현재 입력된 배치의 각 인덱스 `i`에 해당하는 입력값의 마지막 상태가 다음 배치의 각 인덱스 `i`에 해당하는 입력값의 초기 상태로 사용됩니다. 매우 긴 시계열을 연속된 배치로 나누어 처리할 때 유용합니다.  
    

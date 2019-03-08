
## 활성화 함수의 사용

활성화 함수는 'Acitvation'레이어 혹은 모든 전방 레이어에서 지원하는 'activation'인수를 통해 사용 될 수 있습니다. 

```python
from keras.layers import Activation, Dense

model.add(Dense(64))
model.add(Activation('tanh'))
```

위의 코드는 다음의 코드와 같습니다:

```python
model.add(Dense(64, activation='tanh'))
```

활성화 함수로는 TensorFlow/Theano/CNTK 함수들을 요소별로 전달 할 수도 있습니다:

```python
from keras import backend as K

model.add(Dense(64, activation=K.tanh))
```

## 사용 가능한 활성화 함수

### softmax


```python
keras.activations.softmax(x, axis=-1)
```


Softmax 활성화 함수.

__요소 (Arguments)__

- __x__: Input tensor.
- __axis__: Integer, axis along which the softmax normalization is applied.

__반환문 (Returns)__

Tensor, output of softmax transformation.

__예외처리 (Raises)__

- __ValueError__: In case `dim(x) == 1`.
    
----

### elu


```python
keras.activations.elu(x, alpha=1.0)
```


Exponential linear unit.

__요소 (Arguments)__

- __x__: Input tensor.
- __alpha__: A scalar, slope of negative section.

__반환문 (Returns)__

The exponential linear activation: `x` if `x > 0` and
`alpha * (exp(x)-1)` if `x < 0`.

__참조 (References)__

- [Fast and Accurate Deep Network Learning by Exponential
   Linear Units (ELUs)](https://arxiv.org/abs/1511.07289)
    
----

### selu


```python
keras.activations.selu(x)
```


Scaled Exponential Linear Unit (SELU).

SELU 는 다음과 같습니다: `scale * elu(x, alpha)`, alpha와 scale은
미리 정해진 상수입니다.

'alpha'와 'scale' 값은 가중치가 정확하게 초기화되고('lecun_normal' initialize' 참조) 
입력 횟수가 "충분히 많은 한"(자세한 내용은 참조), 입력의 평균과 분산을 두 개의 연속 
레이어 사이에 보존할 수 있도록 선택합니다.


__요소 (Arguments)__

- __x__: A tensor or variable to compute the activation function for.

__반환문 (Returns)__

   The scaled exponential unit activation: `scale * elu(x, alpha)`.

__메모 (Note)__

- To be used together with the initialization "lecun_normal".
- To be used together with the dropout variant "AlphaDropout".

__참 (Reference)__

- [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
    
----

### softplus


```python
keras.activations.softplus(x)
```


Softplus activation function.

__요소 (Arguments)__

- __x__: Input tensor.

__반환문 (Returns)__

The softplus activation: `log(exp(x) + 1)`.
    
----

### softsign


```python
keras.activations.softsign(x)
```


Softsign activation function.

__요소 (Arguments)__

- __x__: Input tensor.

__반환문 (Returns)__

The softsign activation: `x / (abs(x) + 1)`.
    
----

### relu


```python
keras.activations.relu(x, alpha=0.0, max_value=None, threshold=0.0)
```


Rectified Linear Unit.

디폴트 값으로 요소(element-wise) 'max(x, 0)'를 반환합니다.

그렇지 않으면 다음과 같다.

`f(x) = max_value` for `x >= max_value`,
`f(x) = x` for `threshold <= x < max_value`,
`f(x) = alpha * (x - threshold)` otherwise.

__요소 (Arguments)__

- __x__: Input tensor.
- __alpha__: float. Slope of the negative part. Defaults to zero.
- __max_value__: float. Saturation threshold.
- __threshold__: float. Threshold value for thresholded activation.

__반환문 (Returns)__

A tensor.
    
----

### tanh


```python
keras.activations.tanh(x)
```


쌍곡탄젠트 (Hyperbolic tangent) 활성화 함수.

----

### sigmoid


```python
keras.activations.sigmoid(x)
```


시그모이드 (Sigmoid) 활성화 함수.

----

### hard_sigmoid


```python
keras.activations.hard_sigmoid(x)
```


Hard sigmoid 활성화 함수.

Faster to compute than sigmoid activation.

__요소 (Arguments)__

- __x__: Input tensor.

__반환문 (Returns)__

Hard sigmoid activation:

- `0` if `x < -2.5`
- `1` if `x > 2.5`
- `0.2 * x + 0.5` if `-2.5 <= x <= 2.5`.

----

### exponential


```python
keras.activations.exponential(x)
```


Exponential (base e) 활성화 함수.

----

### linear


```python
keras.activations.linear(x)
```


Linear (i.e. identity) 활성화 함수.


## "고급 활성화" (Advanced Activations)에 대하여

단순한 TensorFlow/Theano/CNTK 기능(예를 들어 상태를 유지하는 학습 가능한 활성화)보다 복잡한 활성화는 [Advanced Activation layers](layers/advanced-activations.md) 으로 제공되며, 'Keras.layers.advanced_activations' 모듈에서 확인할 수 있습니다. 여기에는 "PReLU"와 "LeakyReLU"가 포함됩니다.

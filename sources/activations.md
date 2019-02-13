
## 활성화 함수의 사용

활성화 함수는 `Activation` layer, 또는 모든 forward layer가 지원하고 있는 `activation` argument을 통해 사용될 수 있습니다.

```python
from keras.layers import Activation, Dense

model.add(Dense(64))
model.add(Activation('tanh'))
```

이는 아래의 코드와 같습니다. :

```python
model.add(Dense(64, activation='tanh'))
```

또한 활성화 함수로 요소별 TensorFlow/Theano/CNTK 함수를 쓸 수 있습니다.

```python
from keras import backend as K

model.add(Dense(64, activation=K.tanh))
```

## 사용가능한 활성화 함수들

### softmax


```python
keras.activations.softmax(x, axis=-1)
```


소프트맥스 활성화 함수(Softmax activation function).

__Arguments__

- __x__: 입력 텐서.
- __axis__: 정수형, softmax 일반화(normalization)가 적용되는 축.

__Returns__

텐서, 소프트맥스(softmax) 변환의 출력값.

__Raises__

- __ValueError__: `dim(x) == 1`인 경우.
    
----

### elu


```python
keras.activations.elu(x, alpha=1.0)
```


 지수 선형 단위(Exponential Linear Unit(ELU)).

__Arguments__

- __x__: 입력 텐서.
- __alpha__: 스칼라, 음의 구간의 경사.

__Returns__

지수 선형 활성화 함수 : `x` if `x > 0` and
`alpha * (exp(x)-1)` if `x < 0`.

__References__

- [Fast and Accurate Deep Network Learning by Exponential
   Linear Units (ELUs)](https://arxiv.org/abs/1511.07289)
    
----

### selu


```python
keras.activations.selu(x)
```


 스케일 지수 선형 단위(Scaled Exponential Linear Unit (SELU)).

SELU 는 다음과 같습니다 : `scale * elu(x, alpha)`, alpha와 scale는 미리 정의된 상수입니다. 
가중치가 정확하게 초기화되고 입력 횟수가 "충분히 크다"는 한, 
입력의 평균과 분산을 2개의 연속 레이어 사이에 보존하도록 `alpha`와 `scale`의 값을 선택합니다.
(더 많은 정보를 위해 reference를 참고하십시오.)

__Arguments__

- __x__: 활성화 함수를 계산하기 위한 텐서 혹은 변수

__Returns__

   스케일 지수 선형 단위 함수(The scaled exponential unit activation): `scale * elu(x, alpha)`.

__Note__

- "lecun_normal" 초기화와 함께 사용하기.
- "AlphaDropout" dropout 변수와 함께 사용하기.

__References__

- [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
    
----

### softplus


```python
keras.activations.softplus(x)
```


소프트플러스 활성화 함수(Softplus activation function).

__Arguments__

- __x__: 입력 텐서.

__Returns__

소프트플러스 활성화 함수(The softplus activation): `log(exp(x) + 1)`.
    
----

### softsign


```python
keras.activations.softsign(x)
```


소프트사인 활성화 함수(Softsign activation function).

__Arguments__

- __x__: 입력 텐서.

__Returns__

소프트사인 활성화 함수(The softsign activation): `x / (abs(x) + 1)`.
    
----

### relu


```python
keras.activations.relu(x, alpha=0.0, max_value=None, threshold=0.0)
```


Rectified Linear Unit.

With default values, it returns element-wise `max(x, 0)`.

Otherwise, it follows:
`f(x) = max_value` for `x >= max_value`,
`f(x) = x` for `threshold <= x < max_value`,
`f(x) = alpha * (x - threshold)` otherwise.

__Arguments__

- __x__: Input tensor.
- __alpha__: float. Slope of the negative part. Defaults to zero.
- __max_value__: float. Saturation threshold.
- __threshold__: float. Threshold value for thresholded activation.

__Returns__

A tensor.
    
----

### tanh


```python
keras.activations.tanh(x)
```


Hyperbolic tangent activation function.

----

### sigmoid


```python
keras.activations.sigmoid(x)
```


Sigmoid activation function.

----

### hard_sigmoid


```python
keras.activations.hard_sigmoid(x)
```


Hard sigmoid activation function.

Faster to compute than sigmoid activation.

__Arguments__

- __x__: Input tensor.

__Returns__

Hard sigmoid activation:

- `0` if `x < -2.5`
- `1` if `x > 2.5`
- `0.2 * x + 0.5` if `-2.5 <= x <= 2.5`.

----

### exponential


```python
keras.activations.exponential(x)
```


Exponential (base e) activation function.

----

### linear


```python
keras.activations.linear(x)
```


Linear (i.e. identity) activation function.


## On "Advanced Activations"

Activations that are more complex than a simple TensorFlow/Theano/CNTK function (eg. learnable activations, which maintain a state) are available as [Advanced Activation layers](layers/advanced-activations.md), and can be found in the module `keras.layers.advanced_activations`. These include `PReLU` and `LeakyReLU`.

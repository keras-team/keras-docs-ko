## 활성화 함수의 사용법

활성화 함수<sub>activations</sub>는 `Activation` layer이나 forward-pass에 사용되는 layers에서 지원하는 `activation` argument로 사용 가능합니다.

```python
from keras.layers import Activation, Dense

model.add(Dense(64))
model.add(Activation('tanh'))
```

이것은 다음과 같습니다.

```python
model.add(Dense(64, activation='tanh'))
```
여러분은 Tensorflow/Theano/CNTK 의 element-wise 함수도 활성화 함수로 사용할 수 있습니다.

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

__Arguments__

- __x__: Input tensor.
- __axis__: 정수, softmax normalization이 적용되는 axis.


__Returns__

Softmax 변환으로 생성된 tensor.

__Raises__

- __ValueError__: `dim(x) == 1`일 때.

----

### elu


```python
keras.activations.elu(x, alpha=1.0)
```


Exponential linear unit.

__Arguments__

- __x__: Input tensor.
- __alpha__: 스칼라, 음수 부분의 기울기.

__Returns__

Exponential linear unit의 활성값:  
`x > 0` 이면 `x`,  
`x < 0` 이면 `alpha * (exp(x)-1)`.

__References__

- [Fast and Accurate Deep Network Learning by Exponential
   Linear Units (ELUs)](https://arxiv.org/abs/1511.07289)

----

### selu


```python
keras.activations.selu(x)
```


Scaled Exponential Linear Unit (SELU).

SELU는 `scale * elu(x, alpha)`와 같으며, `alpha`와 `scale`은
미리 정해지는 상수입니다. weights가 올바르게 초기화되고(`lecun_normal`를 확인해주십시오)
inputs 수가 "충분히 많다"면(참고 자료에서 더 많은 정보를 확인해주십시오) `alpha`와 `scale`의 값은
입력의 평균값과 분산값이 두 개의 연속되는 layers 사이에서 보존되도록 결정됩니다.

__Arguments__

- __x__: 활성 함수를 적용하려는 tensor 또는 변수.

__Returns__

Scaled exponential linear unit의 활성값 `scale * elu(x, alpha)`.

__Note__

- "lecun_normal"과 함께 사용되어야 합니다.
- "AlphaDropout"과 함께 이용되어야 합니다.

__References__

- [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)

----

### softplus


```python
keras.activations.softplus(x)
```


Softplus 활성화 함수.

__Arguments__

- __x__: Input tensor.

__Returns__

Softplus의 활성값 `log(exp(x) + 1)`.

----

### softsign


```python
keras.activations.softsign(x)
```


Softsign 활성화 함수.

__Arguments__

- __x__: 입력 텐서.

__Returns__

Softsign의 활성값 `x / (abs(x) + 1)`.

----

### relu


```python
keras.activations.relu(x, alpha=0.0, max_value=None, threshold=0.0)
```


Rectified Linear Unit.

디폴트 인수들을 사용하면 element-wise `max(x, 0)`를 반환합니다.

다른 인수를 사용하면 다음과 같습니다.  
`x >= max_value` 일 때 `f(x) = max_value`,  
`threshold <= x < max_value` 일 때 `f(x) = x`,  
그 외는 `f(x) = alpha * (x - threshold)`.

__Arguments__

- __x__: Input tensor.
- __alpha__: 부동소수. 음수 부분의 기울기. 디폴트 값은 0.
- __max_value__: 부동소수. 포화 임계값.
- __threshold__: 부동소수. 활성화를 위한 임계치.

__Returns__

A tensor.

----

### tanh


```python
keras.activations.tanh(x)
```


Hyperbolic tangent 활성화 함수.

__Arguments__

- __x__: Input tensor.

__Returns__

The hyperbolic activation:
`tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`


----

### sigmoid


```python
keras.activations.sigmoid(x)
```


Sigmoid 활성화 함수.

__Arguments__

- __x__: Input tensor.

__Returns__

The sigmoid activation: `1 / (1 + exp(-x))`.

----

### hard_sigmoid


```python
keras.activations.hard_sigmoid(x)
```


Hard sigmoid activation function.

sigmoid 활성화 함수보다 연산 속도가 빠릅니다.

__Arguments__

- __x__: Input tensor.

__Returns__

Hard sigmoid의 활성값:

- `x < -2.5` 이면 `0`
- `x > 2.5` 이면 `1`
- `-2.5 <= x <= 2.5` 이면 `0.2 * x + 0.5`.

----

### exponential


```python
keras.activations.exponential(x)
```


지수(밑은 e) 활성화 함수.

__Arguments__

- __x__: Input tensor.

__Returns__

Exponential activation: `exp(x)`.

----

### linear


```python
keras.activations.linear(x)
```


선형(즉, 항등) 활성화 함수.

__Arguments__

- __x__: Input tensor.

__Returns__

Input tensor, unchanged.


## "고급 활성화 함수"에 대하여

간단한 TensorFlow/Theano/CNTK 활성화 함수보다 더 복잡한 함수들(eg. 학습 가능한 파라미터를 가진 활성화 함수)은 [Advanced Activation layers](layers/advanced-activations.md) 에서 확인할 수 있으며, `keras.layers.advanced_activations` 모듈에서 찾을 수 있습니다. 이는 `PReLU`와 `LeakyReLU`를 포함합니다.

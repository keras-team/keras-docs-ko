
## Activations의 이용

Activations는 `Activation` 층이나 앞선 층에서 지원하는 모든 `activation` argument로 이용 가능합니다:

```python
from keras.layers import Activation, Dense

model.add(Dense(64))
model.add(Activation('tanh'))
```

이것은 다음과 같습니다:

```python
model.add(Dense(64, activation='tanh'))
```

여러분은 Tensorflow/Theano/CNTK 의 element-wise 함수도 활성함수로 이용할 수 있습니다:

```python
from keras import backend as K

model.add(Dense(64, activation=K.tanh))
```

## 사용 가능한 activations

### softmax


```python
keras.activations.softmax(x, axis=-1)
```


Softmax 활성화 함수.

__Arguments__

- __x__: 입력 텐서.
- __axis__: 정수, softmax 정규화가 적용되는 축

__Returns__

텐서, softmax 변환의 출력.

__Raises__

- __ValueError__: `dim(x) == 1`일 때.
    
----

### elu


```python
keras.activations.elu(x, alpha=1.0)
```


Exponential linear unit.

__Arguments__

- __x__: 입력 텐서.
- __alpha__: 스칼라, 음수 부분의 기울기

__Returns__

The exponential linear activation: `x > 0` 라면 `x`,
`x < 0`라면 `alpha * (exp(x)-1)`.

__References__

- [Fast and Accurate Deep Network Learning by Exponential
   Linear Units (ELUs)](https://arxiv.org/abs/1511.07289)
    
----

### selu


```python
keras.activations.selu(x)
```


Scaled Exponential Linear Unit (SELU).

SELU는 다음과 같습니다: `scale * elu(x, alpha)`, 여기서 `alpha`와 `scale`은 
미리 정해지는 상수입니다. `alpha`와 `scale`의 값은 입력의 평균값과 분산값이
두 개의 연속되는 층 사이에서 보존되도록 결정됩니다. 조건은 가중치의 초기치
설정이 올바르게 되고(`lecun_normal` 초기치 설정 참조) 입력의 수가 "충분히
클 때"입니다(references에서 더 많은 정보를 확인해주십시오).

__Arguments__

- __x__: 활성 함수를 적용하는 텐서, 또는 변수.

__Returns__

   scaled exponential unit activation의 값: `scale * elu(x, alpha)`.

__Note__

- "lecun_normal"과 함께 사용되어야 합니다.
- "AlphaDropout"드롭아웃 변수와 함께 이용되어야 합니다.

__References__

- [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
    
----

### softplus


```python
keras.activations.softplus(x)
```


Softplus 활성화 함수.

__Arguments__

- __x__: 입력 텐서.

__Returns__

softplus 활성 함수: `log(exp(x) + 1)`.
    
----

### softsign


```python
keras.activations.softsign(x)
```


Softsign 활성화 함수.

__Arguments__

- __x__: 입력 텐서.

__Returns__

softsign 활성화 함수: `x / (abs(x) + 1)`.
    
----

### relu


```python
keras.activations.relu(x, alpha=0.0, max_value=None, threshold=0.0)
```


Rectified Linear Unit.

디폴트 값들로는, element-wise `max(x, 0)`를 반환합니다.

값을 준다면, 다음을 따릅니다:
`x >= max_value` 일 때 `f(x) = max_value`,
`threshold <= x < max_value` 일 때 `f(x) = x`,
그 외는 `f(x) = alpha * (x - threshold)`.

__Arguments__

- __x__: 입력 텐서.
- __alpha__: 부동소수점. 음수 부분의 기울기. 디폴트는 0
- __max_value__: 부동소수점. 포화 임계값
- __threshold__: 부동소수점. 임계값 활성화를 위한 임계치

__Returns__

텐서.
    
----

### tanh


```python
keras.activations.tanh(x)
```


Hyperbolic tangent activation function.

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


Sigmoid activation function.

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

sigmoid 활성화 함수보다 컴퓨팅 속도가 빠릅니다.

__Arguments__

- __x__: 입력 텐서.

__Returns__

Hard sigmoid activation:

- `x < -2.5` 라면 `0`
- `x > 2.5` 라면 `1`
- `-2.5 <= x <= 2.5` 라면 `0.2 * x + 0.5`.

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


선형(i.e. identity) 활성화 함수.

__Arguments__

- __x__: Input tensor.

__Returns__

Input tensor, unchanged.
    

## "고급 Activations"에 대하여

간단한 TensorFlow/Theano/CNTK 활성화 함수보다 더 복잡한 함수들(eg. 상태를 유지하는 학습 가능한 활성화 함수) 는 [Advanced Activation layers](layers/advanced-activations.md) 에서 확인할 수 있으며, `keras.layers.advanced_activations` 모듈에서 찾을 수 있습니다. 이는 `PReLU`와 `LeakyReLU`를 포함합니다.

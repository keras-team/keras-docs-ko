## 활성화 함수의 사용법<sub>Usage of activations</sub>

활성화 함수는 `Activation` 층<sub>layer</sub>이나 포워드 패스를 사용하는 모든 층에서 `activation` 인자로 사용 가능합니다.

```python
from keras.layers import Activation, Dense

model.add(Dense(64))
model.add(Activation('tanh'))
```

위의 코드는 아래와 동일합니다.

```python
model.add(Dense(64, activation='tanh'))
```

TensorFlow, Theano, CNTK에서 제공하는 원소별<sub>element-wise</sub> 연산도 활성화 함수로 사용할 수 있습니다.

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

__인자__

- __x__: 입력 텐서.
- __axis__: `int`. Softmax 정규화<sub>normalization</sub>가 적용되는 축<sub>axis</sub>.


__반환값__

Softmax 변환으로 생성된 텐서.  
`f(x) = exp(x) / sum(exp(x))`

__오류__

- __ValueError__: `dim(x) == 1`인 경우 발생합니다.

----

### elu


```python
keras.activations.elu(x, alpha=1.0)
```


Exponential Linear Unit(ELU).

__인자__

- __x__: 입력 텐서.
- __alpha__: `float`. `x < 0`인 경우의 기울기. 기본값은 `1.0`.

__반환값__

ELU의 활성값.  
- `x > 0`인 경우, `f(x) = x`  
- `x < 0`인 경우, `f(x) = alpha * (exp(x) - 1)`

__참고__

- [Fast and Accurate Deep Network Learning by Exponential
   Linear Units (ELUs)](https://arxiv.org/abs/1511.07289)

----

### selu


```python
keras.activations.selu(x)
```


Scaled Exponential Linear Unit(SELU).

SELU는 `scale * elu(x, alpha)`와 같습니다. `alpha`와 `scale`은
미리 정해지는 상수입니다. 가중치<sub>weights</sub>가 올바르게 초기화되고(`lecun_normal` 참조)
입력 수가 '충분히 많다'면 `alpha`와 `scale`의 값은
입력의 평균과 분산이 연속되는 두 개의 층에서 보존되도록 결정됩니다(참고자료 참조).

__인자__

- __x__: 입력 텐서.

__반환값__

SELU의 활성값  
`f(x) = scale * elu(x, alpha)`

__유의 사항__

- `lecun_normal`과 함께 사용되어야 합니다.
- `AlphaDropout`과 함께 사용되어야 합니다.

__참고__

- [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)

----

### softplus


```python
keras.activations.softplus(x)
```


Softplus 활성화 함수.

__인자__

- __x__: 입력 텐서.

__반환값__

Softplus의 활성값.  
`f(x) = log(exp(x) + 1)`

----

### softsign


```python
keras.activations.softsign(x)
```


Softsign 활성화 함수.

__인자__

- __x__: 입력 텐서.

__반환값__

Softsign의 활성값.  
`f(x) = x / (abs(x) + 1)`

----

### relu


```python
keras.activations.relu(x, alpha=0.0, max_value=None, threshold=0.0)
```


Rectified Linear Unit(ReLU).

인자의 기본값을 사용하면 원소별로 연산된 `max(x, 0)`를 반환합니다.

다른 인자를 사용하면 ReLU는 다음과 같습니다.
- `x >= max_value`인 경우, `f(x) = max_value` 
- `threshold <= x < max_value`인 경우, `f(x) = x`  
- 나머지 경우, `f(x) = alpha * (x - threshold)`

__인자__

- __x__: 입력 텐서.
- __alpha__: `float`. `x < 0`인 경우의 기울기. 기본값은 `0.0`.
- __max_value__: `float`. 포화 임계값.
- __threshold__: `float`. 활성화가 일어나는 임계값.

__반환값__

ReLU 변환으로 생성된 텐서.  
- `x >= max_value`인 경우, `f(x) = max_value` 
- `threshold <= x < max_value`인 경우, `f(x) = x`  
- 나머지 경우, `f(x) = alpha * (x - threshold)`

----

### tanh


```python
keras.activations.tanh(x)
```


Hyperbolic Tangent 활성화 함수.

__인자__

- __x__: 입력 텐서.

__반환값__

Hyperbolic Tangent의 활성값.  
`f(x) = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`


----

### sigmoid


```python
keras.activations.sigmoid(x)
```


Sigmoid 활성화 함수.

__인자__

- __x__: 입력 텐서.

__반환값__

Sigmoid의 활성값.  
`f(x) = 1 / (1 + exp(-x))`

----

### hard_sigmoid


```python
keras.activations.hard_sigmoid(x)
```


Hard sigmoid 활성화 함수.

Sigmoid 활성화 함수보다 연산 속도가 빠릅니다.

__인자__

- __x__: 입력 텐서.

__반환값__

Hard sigmoid의 활성값.

- `x < -2.5`인 경우, `f(x) = 0`
- `x > 2.5`인 경우, `f(x) = 1`
- `-2.5 <= x <= 2.5`인 경우, `f(x) = 0.2 * x + 0.5`

----

### exponential


```python
keras.activations.exponential(x)
```


(밑이 e인) 지수 활성화 함수.

__인자__

- __x__: 입력 텐서.

__반환값__

Exponential의 활성값.
`f(x) = exp(x)`

----

### linear


```python
keras.activations.linear(x)
```


선형(즉, 항등) 활성화 함수.

__인자__

- __x__: 입력 텐서.

__반환값__

변하지 않은 입력 텐서.


## 고급 활성화 함수에 대하여

간단한 TensorFlow, Theano, CNTK의 활성화 함수보다 더 복잡한 함수들(예: 학습 가능한 파라미터를 가진 활성화 함수)은 [고급 활성화 함수의 사용법](layers/advanced-activations.md)에서 확인할 수 있으며, `keras.layers.advanced_activations` 모듈에서 찾을 수 있습니다. `PReLU`와 `LeakyReLU`도 여기서 찾을 수 있습니다.

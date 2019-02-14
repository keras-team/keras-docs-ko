
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


정류 선형 단위(Rectified Linear Unit).

기본적으로, `max(x, 0)`을 반환합니다.

그 외에는, 다음과 같습니다 :
`f(x) = max_value` 은 `x >= max_value` 로,
`f(x) = x` 은 `threshold <= x < max_value` 로,
`f(x) = alpha * (x - threshold)` 그 외의 값으로 반환합니다.

__Arguments__

- __x__: 입력 텐서.
- __alpha__: float 타입, 음의 기울기, 기본값은 0.
- __max_value__: float 타입, 포화(Saturation) 임계값.
- __threshold__: float 타입, 활성화에 대한 임계값.

__Returns__

텐서.
    
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

sigmoid 활성화 함수보다 계산 속도가 빠릅니다.

__Arguments__

- __x__: 입력 텐서.

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


지수 (base e) 활성화 함수.

----

### linear


```python
keras.activations.linear(x)
```


선형 (i.e. identity) 활성화 함수.


## "심화된(Advanced) 활성화 함수들"

단순한 TensorFlow/Theano/CNTK 함수(예. 상태를 유지하는 학습가능한 활성화 함수)보다 복잡한 활성화 함수는 [Advanced Activation layers](layers/advanced-activations.md)에서 찾을 수 있으며, `keras.layers.advanced_activations` 모듈에서 확인할 수 있습니다. 이 모듈에서는 `PReLU` 와 `LeakyReLU`를 포함하고 있습니다.
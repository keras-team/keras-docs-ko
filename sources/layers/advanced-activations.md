<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/advanced_activations.py#L19)</span>
### LeakyReLU

```python
keras.layers.LeakyReLU(alpha=0.3)
```

Leaky Rectified Linear Unit 활성화 함수입니다.

유닛이 활성화되지 않는 경우 작은 그래디언트를 허용합니다.  
- `x < 0`인 경우, `f(x) = alpha * x`  
- `x >= 0`인 경우, `f(x) = x`

__입력 형태__

입력값의 형태는 임의로 지정됩니다. `LeakyReLU`을 모델의 첫 번째 층으로 사용하는 경우 키워드 인자로 `input_shape` 인자<sub>argument</sub>(샘플 축<sub>axis</sub>을 제외한 `int` 튜플)를 지정해야 합니다.

__출력 형태__

입력 형태와 동일합니다.

__인자__

- __alpha__: 음이 아닌 `float`. 음의 부분 기울기 계수.

__참고__

- [Rectifier Nonlinearities Improve Neural Network Acoustic Models](
   https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf)

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/advanced_activations.py#L59)</span>
### PReLU

```python
keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)
```

Parametric Rectified Linear Unit 활성화 함수입니다.

PReLU는 다음과 같습니다.  
- `x < 0`인 경우, `f(x) = alpha * x`  
- `x >= 0`인 경우, `f(x) = x`

`alpha`는 x와 동일한 형태를 가진 학습된 배열입니다.

__입력 형태__

입력값의 형태는 임의로 지정됩니다. `PReLU`를 모델의 첫 번째 층으로 사용하는 경우 키워드 인자로 `input_shape` 인자(샘플 축을 제외한 `int` 튜플)를 지정해야 합니다.

__출력 형태__

입력 형태와 동일합니다.

__인자__

- __alpha_initializer__: 가중치<sub>weights</sub>를 위한 초기화 함수<sub>initializer</sub>.
- __alpha_regularizer__: 가중치를 위한 규제 함수<sub>regularizer</sub>.
- __alpha_constraint__: 가중치의 제약<sub>constraint</sub>.
- __shared_axes__: 활성화 함수에 대해 학습 가능한 파라미터들을 공유할 축. 예를 들어, 만일 입력 특징 맵<sub>feature map</sub>들이 2D 합성곱<sub>convolution</sub>으로부터 생성되어 `(batch, height, width, channels)`의 형태를 가진다면, `shared_axes=[1, 2]`로 설정하여 각 필터가 하나의 매개 변수 세트를 공유하도록 할 수 있습니다.

__참고__

- [Delving Deep into Rectifiers: Surpassing Human-Level Performance on
   ImageNet Classification](https://arxiv.org/abs/1502.01852)

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/advanced_activations.py#L153)</span>
### ELU

```python
keras.layers.ELU(alpha=1.0)
```

Exponential Linear Unit 활성화 함수입니다.

ELU는 다음과 같습니다.  
- `x < 0`인 경우, `f(x) =  alpha * (exp(x) - 1)`  
- `x >= 0`인 경우, `f(x) = x`

__입력 형태__

입력값의 형태는 임의로 지정됩니다. `ELU` 모델의 첫 번째 층으로 사용하는 경우 키워드 인자로 `input_shape` 인자(샘플 축을 제외한 `int` 튜플)를 지정해야 합니다.

__출력 형태__

입력 형태와 동일합니다.

__인자__

- __alpha__: `x < 0`인 경우 곱할 배수<sub>factor</sub>.

__참고__

- [Fast and Accurate Deep Network Learning by Exponential Linear Units
   (ELUs)](https://arxiv.org/abs/1511.07289v1)

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/advanced_activations.py#L193)</span>
### ThresholdedReLU

```python
keras.layers.ThresholdedReLU(theta=1.0)
```

ThresholdedReLU 활성화 함수입니다.

ThresholdedReLU는 다음과 같습니다.  
- `x > theta`인 경우, `f(x) = x`  
- `x <= theta`인 경우,  `f(x) = 0`

__입력 형태__

입력값의 형태는 임의로 지정됩니다. `ThresholdedReLU`를 모델의 첫 번째 층으로 사용하는 경우 키워드 인자로 `input_shape` 인자(샘플 축을 제외한 `int` 튜플)를 지정해야 합니다.

__출력 형태__

입력 형태와 동일합니다.

__인자__

- __theta__: 음이 아닌 `float`. 활성화가 이루어지는 임계값.

__참고__

- [Zero-Bias Autoencoders and the Benefits of Co-Adapting Features](
   https://arxiv.org/abs/1402.3337)

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/advanced_activations.py#L233)</span>
### Softmax

```python
keras.layers.Softmax(axis=-1)
```

Softmax 활성화 함수입니다.

__입력 형태__

입력값의 형태는 임의로 지정됩니다. `Softmax`를 모델의 첫 번째 층으로 사용하는 경우 키워드 인자로 `input_shape` 인자(샘플 축을 제외한 `int` 튜플)를 지정해야 합니다.

__출력 형태__

입력 형태와 동일합니다.

__인자__

- __axis__: `int`. softmax 정규화<sub>normalization</sub>가 적용되는 축.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/advanced_activations.py#L265)</span>
### ReLU

```python
keras.layers.ReLU(max_value=None, negative_slope=0.0, threshold=0.0)
```

Rectified Linear Unit 활성화 함수입니다.

인자들의 기본값을 사용하면 원소별로<sub>element-wise</sub> 연산된 `max(x, 0)`를 반환합니다.

다른 인자를 사용하면 다음과 같습니다.  
- `x >= max_value`인 경우, `f(x) = max_value`  
- `threshold <= x < max_value`인 경우, `f(x) = x`  
- 나머지 경우, `f(x) = negative_slope * (x - threshold)`.

__입력 형태__

입력값의 형태는 임의로 지정됩니다. `ReLU`를 모델의 첫 번째 층으로 사용하는 경우 키워드 인자로 `input_shape` 인자(샘플 축을 제외한 `int` 튜플)를 지정해야 합니다.

__출력 형태__

입력 형태와 동일합니다.

__인자__

- __max_value__: 음이 아닌 `float`. 최대 활성화 값.
- __negative_slope__: 음이 아닌 `float`. 음의 부분 기울기 계수.
- __threshold__: `float`. 임계값이 정해진 활성화 함수인 경우,  임계값.

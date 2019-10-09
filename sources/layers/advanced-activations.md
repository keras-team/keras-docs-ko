<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/advanced_activations.py#L19)</span>
### LeakyReLU

```python
keras.layers.LeakyReLU(alpha=0.3)
```

ReLU(Rectified Linear Unit) 활성화 함수의 leaky version입니다.

Unit이 활성화되지 않는 경우 작은 gradient를 허용합니다.
`x < 0인 경우 f(x) = alpha * x`,  
`x >= 0인 경우 f(x) = x`.

__Input shape__

임의입니다. 이 layer를 model의 첫 번째 layer로 사용할 때 키워드 argument `input_shape` (샘플 axis를 제외한 정수 튜플)를 사용하십시오.

__Output shape__

Input shape와 동일합니다.

__Arguments__

- __alpha__: 음이 아닌 부동소수. 음의 부분 기울기 계수입니다.

__References__

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
`x < 0인 경우 f(x) = alpha * x`,  
`x >= 0인 경우 f(x) = x`,  
`alpha`는 x와 동일한 shape를 가진 학습된 배열입니다.

__Input shape__

임의입니다. 이 layer를 model의 첫 번째 layer로 사용할 때 키워드 argument `input_shape` (샘플 axis를 제외한 정수 튜플)를 사용하십시오.

__Output shape__

Input shape와 동일합니다.

__Arguments__

- __alpha_initializer__: weights를 위한 initializer 함수입니다.
- __alpha_regularizer__: weights를 위한 regularizer 함수입니다.
- __alpha_constraint__: weights의 constraint입니다.
- __shared_axes__: 활성화 함수에 대해 학습 가능한 parameter들을 공유할 axis를 의미합니다. 예를 들어, 만일 입력 feature map들이 `(batch, height, width, channels)`의 output shape으로 2D convolution으로부터 생성된 것이며, 각 필터가 하나의 매개 변수 세트를 공유하도록 하고 싶은 경우, `shared_axes=[1, 2]`로 설정하십시오.

__References__

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
`x < 0인 경우 f(x) =  alpha * (exp(x) - 1.)`,  
`x >= 0인 경우 f(x) = x`.

__Input shape__

임의입니다. 이 layer를 model의 첫 번째 layer로 사용할 때 키워드 argument `input_shape` (샘플 axis를 제외한 정수 튜플)를 사용하십시오.

__Output shape__

Input shape와 동일합니다.

__Arguments__

- __alpha__: 음의 부분 factor에 대한 값입니다.

__References__

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
`x > theta인 경우 f(x) = x`,  
`그렇지 않은 경우 f(x) = 0`.

__Input shape__

임의입니다. 이 layer를 model의 첫 번째 layer로 사용할 때 키워드 argument `input_shape` (샘플 axis를 제외한 정수 튜플)를 사용하십시오.

__Output shape__

Input shape와 동일합니다.

__Arguments__

- __theta__: 음이 아닌 부동소수. 활성화가 이루어지는 임계값 위치입니다.

__References__

- [Zero-Bias Autoencoders and the Benefits of Co-Adapting Features](
   https://arxiv.org/abs/1402.3337)

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/advanced_activations.py#L233)</span>
### Softmax

```python
keras.layers.Softmax(axis=-1)
```

Softmax 활성화 함수입니다.

__Input shape__

임의입니다. 이 layer를 model의 첫 번째 layer로 사용할 때 키워드 argument `input_shape` (샘플 axis를 제외한 정수 튜플)를 사용하십시오.

__Output shape__

Input shape와 동일합니다.

__Arguments__

- __axis__: softmax normalization가 적용되는 축의 정숫값.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/advanced_activations.py#L265)</span>
### ReLU

```python
keras.layers.ReLU(max_value=None, negative_slope=0.0, threshold=0.0)
```

Rectified Linear Unit 활성화 함수입니다.

디폴트 인수들을 사용하면 element-wise `max(x, 0)`를 반환합니다.

다른 인수를 사용하면 다음과 같습니다.
`x >= max_value`인 경우 `f(x) = max_value`,  
`threshold <= x < max_value`인 경우 `f(x) = x`,  
그렇지 않은 경우 `f(x) = negative_slope * (x - threshold)`.

__Input shape__

임의입니다. 이 layer를 model의 첫 번째 layer로 사용할 때 키워드 argument `input_shape` (샘플 axis를 제외한 정수 튜플)를 사용하십시오.

__Output shape__

Input shape와 동일합니다.

__Arguments__

- __max_value__: 음이 아닌 부동소수. 최대 활성화 값을 의미합니다.
- __negative_slope__: 음이 아닌 부동소수. 음의 부분 기울기 계수입니다.
- __threshold__: 부동소수. 임계값이 정해진 활성화를 위한 임계값을 의미합니다.


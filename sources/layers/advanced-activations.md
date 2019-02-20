<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/advanced_activations.py#L19)</span>
### LeakyReLU

```python
keras.layers.LeakyReLU(alpha=0.3)
```

ReLU(Rectified Linear Unit) 활성화 함수의 Leaky version입니다. 

Unit이 활성화되지 않는 경우 작은 기울기를 허용합니다:
`x < 0인 경우 f(x) = alpha * x`,
`x >= 0인 경우 f(x) = x`.

__입력 크기__

임의입니다. 이 레이어를 모델의 첫 번째 레이어로 사용할 때 키워드 인자 `input_shape` (정수 튜플, 샘플 축 미포함)를 사용하십시오. 

__출력 크기__

입력 크기와 동일합니다. 

__인자 설명__

- __alpha__: float >= 0이어야 합니다. 음의 방향 기울기에 대한 계수입니다. 

__참고 자료__

- [Rectifier Nonlinearities Improve Neural Network Acoustic Models](
   https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf)
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/advanced_activations.py#L59)</span>
### PReLU

```python
keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)
```

Parametric Rectified Linear Unit 활성화 함수입니다. 

다음을 따라야 합니다:
`x < 0인 경우 f(x) = alpha * x`,
`x >= 0인 경우 f(x) = x`,
여기서 `alpha`는 x와 동일한 크기를 가진 학습된 배열입니다. 

__입력 크기__

임의입니다. 이 레이어를 모델의 첫 번째 레이어로 사용할 때, 키워드 인자 `input_shape` (정수 튜플, 샘플 축 미포함)를 사용하십시오. 

__출력 크기__

입력 크기와 동일합니다. 

__인자 설명__

- __alpha_initializer__: weights를 위한 초기화 함수입니다. 
- __alpha_regularizer__: weights를 위한 규제기입니다.
- __alpha_constraint__: weights에 대한 제약 조건입니다. 
- __shared_axes__: 활성화 함수에 대해 학습 가능한 매개 변수들을 공유할 축을 의미합니다. 예를 들어, 입력 feature map이 output shape(batch, height, width, channels) 형태로 2D convolution을 거쳐 나온 것일때, 각 필터가 하나의 매개 변수 세트를 공유하도록 하고 싶으신 경우 'shared_axes=[1, 2]'로 설정하십시오. 

__참고 자료__

- [Delving Deep into Rectifiers: Surpassing Human-Level Performance on
   ImageNet Classification](https://arxiv.org/abs/1502.01852)
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/advanced_activations.py#L153)</span>
### ELU

```python
keras.layers.ELU(alpha=1.0)
```

지수 선형 유닛(Exponential Linear Unit) 활성화 함수입니다.

다음을 따라야 합니다:
`x < 0인 경우 f(x) =  alpha * (exp(x) - 1.)`,
`x >= 0인 경우 f(x) = x`.

__입력 크기__

임의입니다. 이 레이어를 모델의 첫 번째 레이어로 사용할 때 키워드 인자 `input_shape` (정수 튜플, 샘플 축 미포함)를 사용하십시오. 

__출력 크기__

입력 크기와 동일합니다. 

__인자 설명__

- __alpha__: 음수 인자에 대한 크기입니다.

__참고 자료__

- [Fast and Accurate Deep Network Learning by Exponential Linear Units
   (ELUs)](https://arxiv.org/abs/1511.07289v1)
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/advanced_activations.py#L193)</span>
### ThresholdedReLU

```python
keras.layers.ThresholdedReLU(theta=1.0)
```

임계값이 있는 ReLU 활성화 함수입니다.

다음을 따라야 합니다:
`x > theta인 경우 f(x) = x`,
`그렇지 않은 경우 f(x) = 0`.

__입력 크기__

임의입니다. 이 레이어를 모델의 첫 번째 레이어로 사용할 때 키워드 인자 `input_shape` (정수 튜플, 샘플 축 미포함)를 사용하십시오. 

__출력 크기__

입력 크기와 동일합니다. 

__인자 설명__

- __theta__: float >= 0이어야 합니다. 활성화가 이뤄질 임계값 위치입니다. 

__참고 자료__

- [Zero-Bias Autoencoders and the Benefits of Co-Adapting Features](
   https://arxiv.org/abs/1402.3337)
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/advanced_activations.py#L233)</span>
### Softmax

```python
keras.layers.Softmax(axis=-1)
```

Softmax 활성화 함수입니다. 

__입력 크기__

임의입니다. 이 레이어를 모델의 첫 번째 레이어로 사용할 때 키워드 인자 `input_shape` (정수 튜플, 샘플 축 미포함)를 사용하십시오. 

__출력 크기__

입력 크기와 동일합니다. 

__인자 설명__

- __axis__: integer이어야 합니다. 축을 따라 softmax 정규화(normalization)가 적용됩니다.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/advanced_activations.py#L265)</span>
### ReLU

```python
keras.layers.ReLU(max_value=None, negative_slope=0.0, threshold=0.0)
```

정류된 선형 유닛 활성화 함수(Rectified Linear Unit activation function)입니다. 

기본값을 사용하면 요소별로 `max(x, 0)`를 반환합니다. 

그렇지 않다면, 다음을 따라야 합니다:
`x >= max_value`인 경우 `f(x) = max_value`,
`threshold <= x < max_value`인 경우 `f(x) = x`,
그렇지 않은 경우 `f(x) = negative_slope * (x - threshold)`.

__입력 크기__

임의입니다. 이 레이어를 모델의 첫 번째 레이어로 사용할 때 키워드 인자 `input_shape` (정수 튜플, 샘플 축 미포함)를 사용하십시오. 

__출력 크기__

입력 크기와 동일합니다. 

__인자 설명__

- __max_value__: float >= 0 이어야 합니다. 최대 활성화 값을 의미합니다. 
- __negative_slope__: float >= 0 이어야 합니다. 음의 방향 기울기에 대한 계수입니다.
- __threshold__: float 범위여야 합니다. 임계값이 정해진 활성화(Thresholded activation)를 위한 임계값을 의미합니다. 
    

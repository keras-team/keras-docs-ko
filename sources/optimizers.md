
## Optimizers의 용법

optimizer는 Keras 모델을 컴파일하기 위해 요구되는 두 인자 중 하나 입니다:

```python
from keras import optimizers

model = Sequential()
model.add(Dense(64, kernel_initializer='uniform', input_shape=(10,)))
model.add(Activation('softmax'))

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)
```

위의 예제와 같이 `model.compile()`에 전달하기 전에 optimizer의 인스턴스를 만들거나, 또는 이름을 이용해서 호출할 수 있습니다. 후자의 경우에는 optimizer를 위한 디폴트 파라메터가 사용됩니다.

```python
# 이름으로 optimizer를 전달: 디폴트 파라메터가 사용됨
model.compile(loss='mean_squared_error', optimizer='sgd')
```

---

## 모든 Keras의 Optimizer에 공통적인 파라메터

`clipnorm` 그리고 `clipvalue` 파라메터는 경사도 자르기(gradient clipping)을 조절하기 위한 모든 optimizer에서 사용될 수 있습니다:

```python
from keras import optimizers

# 경사도가 자르게될 모든 파라메터에 대한
# Norm의 최대값을 1로 설정
sgd = optimizers.SGD(lr=0.01, clipnorm=1.)
```

```python
from keras import optimizers

# A경사도가 자르게될 모든 파라메터의
# 최대값을 0.5, 그리고
# 최소값을 -0.5로 설정
sgd = optimizers.SGD(lr=0.01, clipvalue=0.5)
```

---

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/optimizers.py#L157)</span>
### SGD

```python
keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
```

Stochastic Gradient Descent optimizer.

모멘텀(momentum), 학습률 붕괴(learning rate decay), 그리고 네스테로프 모멘텀(Nesterov momentum)의 지원을 포함합니다.

__인자__

- __lr__: >= 0인 부동소수. 학습률 .
- __momentum__: >= 0인 부동소수. SGD를 적절한 방향으로 가속화 하고, 흔들림(진동)을 약화시키기 위한 파라메터 입니다.
- __decay__: >= 0인 부동소수. 각 업데이트마다의 학습률 붕괴값 입니다.
- __nesterov__: 불리언. 네스테로프 모멘텀을 적용할지 말지를 설정합니다.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/optimizers.py#L220)</span>
### RMSprop

```python
keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
```

RMSProp optimizer.

이 optimizer의 파라메터를 디폴트 값으로 사용하는 것이 권고 됩니다.
(자유롭게 튜닝될 수 있는 학습률은 제외)

이 optimizer는 보통 순환 신경망(Recurrent Neural Networks)을 위한 좋은 선택입니다.

__인자__

- __lr__:  >= 0인 부동소수. 학습률 입니다.
- __rho__: >= 0인 부동소수.
- __epsilon__: >= 0인 부동소수. Fuzzy 인자. `None`으로 설정되면, `K.epsilon()`가 사용됩니다.
- __decay__: >= 0인 부동소수. 각 업데이트마다의 학습률 붕괴값 입니다.

__참고__

- [rmsprop: Divide the gradient by a running average of its recent magnitude](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/optimizers.py#L288)</span>
### Adagrad

```python
keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
```

Adagrad optimizer.

Adagrad는 학습 도중 얼마나 빈번히 파라메터가 업데이트되는 연관성에의해  
파라메터에 특정한 학습률과 함께 사용되는 optmizer 입니다.
파라메터가 더 많이 업데이트될 수록, 더 작은 학습률이 사용됩니다.

이 optimizer의 파라메터는 디폴트 값으로 사용하는 것이 권고 됩니다.

__인자__

- __lr__: >= 0인 부동소수. 학습률 .
- __epsilon__: >= 0인 부동소수. `None`으로 설정되면, `K.epsilon()`가 사용됩니다.
- __decay__: >= 0인 부동소수. 각 업데이트마다의 학습률 붕괴값 입니다.

__참고__

- [Adaptive Subgradient Methods for Online Learning and Stochastic
   Optimization](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/optimizers.py#L353)</span>
### Adadelta

```python
keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
```

Adadelta optimizer.

Adadelta는 Adagrad를 확장한 보다 견고한 optimizer로
모든 과거의 경사도를 축적하는것 대신에, 경사도 업데이트의 이동창에 기반하여 학습률을 조절합니다.
이 방법을 사용하면, Adadelta는 많은 업데이트가 이뤄진 후 일지라도, 학습을 계속합니다.
Adagrad와 비교해 볼 때, Adadelta의 원래 버전에서는 초기 학습률을 설정할 필요가 없었습니다.
현재의 버전에서는 다른 optimizer 처럼 초기 학습률과 붕괴 인자가 설정될 수 있습니다.

이 optimizer의 파라메터는 디폴트 값으로 사용하는 것이 권고 됩니다.

__인자__

- __lr__: >= 0인 부동소수. 학습률. 디폴트 값은 1 입니다..
    디폴트 값으로 사용하는 것이 권고 됩니다.
- __rho__: >= 0인 부동소수.. Adadelta의 붕괴 인자로 각 시간에 대하여 유지되는 경사도의 부분에 상응합니다.
- __epsilon__: >= 0인 부동소수. Fuzzy 인자. `None`으로 설정되면, `K.epsilon()`가 사용됩니다.
- __decay__: >= 0인 부동소수. 초기 학습률 붕괴값 입니다.

__참고__

- [Adadelta - an adaptive learning rate method](
   https://arxiv.org/abs/1212.5701)
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/optimizers.py#L436)</span>
### Adam

```python
keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
```

Adam optimizer.

디폴트 파라메터는 논문에서 언급된 내용을 따릅니다.

__인자__

- __lr__: >= 0인 부동소수. 학습률. 디폴트 값은 1 입니다.
- __beta_1__: 0 < beta < 1인 부동소수. 일반적으로 1에 가깝게 설정 됩니다.
- __beta_2__: 0 < beta < 1인 부동소수. 일반적으로 1에 가깝게 설정 됩니다.
- __epsilon__: >= 0인 부동소수. Fuzz 인자. `None`으로 설정되면, `K.epsilon()`가 사용됩니다.
- __decay__: >= 0인 부동소수. 각 업데이트마다의 학습률 붕괴값 입니다.
- __amsgrad__: 불리언. 이 알고리즘의 변종인 "On the Convergence of Adam and Beyond" 논문에서 소개된 
AMSGrad를 적용할지 말지 설정합니다.

__참조__

- [Adam - A Method for Stochastic Optimization](
   https://arxiv.org/abs/1412.6980v8)
- [On the Convergence of Adam and Beyond](
   https://openreview.net/forum?id=ryQu7f-RZ)
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/optimizers.py#L527)</span>
### Adamax

```python
keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
```

Adam 논문의 섹션7에 소개된 Adamax optimizer.

Adam의 무한대 norm에 기반한 변종 입니다.
디폴트 파라메터는 논문에서 언급된 내용을 따릅니다.

__인자__

- __lr__: >= 0인 부동소수. 학습률. 디폴트 값은 1 입니다.
- __beta_1/beta_2__: 0 < beta < 1인 부동소수. 일반적으로 1에 가깝게 설정 됩니다.
- __epsilon__: >= 0인 부동소수. Fuzz 인자. `None`으로 설정되면, `K.epsilon()`가 사용됩니다.
- __decay__: >= 0인 부동소수. 각 업데이트마다의 학습률 붕괴값 입니다.

__참고__

- [Adam - A Method for Stochastic Optimization](
   https://arxiv.org/abs/1412.6980v8)
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/optimizers.py#L605)</span>
### Nadam

```python
keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
```

Nesterov Adam optimizer.

Adam이 근본적으로는 모멘텀이 적용된 RMSprop인 것과 마찬가지로,
Nadam은 네스테로프 모멘텀이 적용된 Adam RMSprop 이라고 볼 수 있습니다.

디폴트 파라메터는 논문에서 언급된 내용을 따릅니다.
이 optimizer의 파라메터는 디폴트 값으로 사용하는 것이 권고 됩니다.

__인자__

- __lr__: >= 0인 부동소수. 학습률. 디폴트 값은 1 입니다.
- __beta_1/beta_2__:  0 < beta < 1인 부동소수. 일반적으로 1에 가깝게 설정 됩니다.
- __epsilon__: >= 0인 부동소수. Fuzz 인자. `None`으로 설정되면, `K.epsilon()`가 사용됩니다.

__참고__

- [Nadam report](http://cs229.stanford.edu/proj2015/054_report.pdf)
- [On the importance of initialization and momentum in deep learning](
   http://www.cs.toronto.edu/~fritz/absps/momentum.pdf)
    

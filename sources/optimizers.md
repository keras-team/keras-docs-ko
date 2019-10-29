## 최적화 함수의 사용법<sub>Usage of optimizers</sub>

최적화 함수<sub>opti</sub>는 손실 함수<sub>loss</sub>와 더불어 케라스 모델을 컴파일<sub>compile</sub>하기 위한 인자입니다.

```python
from keras import optimizers

model = Sequential()
model.add(Dense(64, kernel_initializer='uniform', input_shape=(10,)))
model.add(Activation('softmax'))

# 최적화 함수의 객체를 만듭니다.
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)
```

최적화 함수는 위의 예시와 같이 객체를 만들어 `model.compile()`의 인자로 전달하거나, 아래와 같이 문자열 형태로 전달할 수 있습니다.
문자열을 전달할 경우, 해당 최적화 함수의 모든 인자는 기본값으로 사용됩니다.

```python
# 최적화 함수를 문자열 형태로 전달하는 경우는
# 기본값이 사용됩니다. 
model.compile(loss='mean_squared_error', optimizer='sgd')
```

---

## 모든 케라스 최적화 함수에 사용되는 공통 인자

모든 최적화 함수는 `clipnorm`과 `clipvalue` 인자를 통해 그래디언트 클리핑(gradient clipping)을 조절할 수 있습니다.

```python
from keras import optimizers

# 모든 그래디언트의 노름의 크기는 최대 1입니다.
sgd = optimizers.SGD(lr=0.01, clipnorm=1.)
```

```python
from keras import optimizers

# 모든 그래디언트의 노름의 크기는 최소 -0.5, 최대 0.5로 제한됩니다.
sgd = optimizers.SGD(lr=0.01, clipvalue=0.5)
```

---

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/optimizers.py#L164)</span>
### SGD

```python
keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False)
```

확률적 경사 하강법<sub>Stochastic Gradient Descent</sub> 최적화 함수

모멘텀과 네스테로프 모멘텀<sub>nesterov momentum</sub> 인자를 지정할 수 있습니다.

__인자__

- __learning_rate__: `float`>=0. 학습률.
- __momentum__: `float`>=0. 적절한 방향으로 SGD의 학습 속도를 가속화하며, 
흔들림(진동)<sub>oscillations</sub>을 줄여줍니다.
- __nesterov__: `boolean`. 네스테로프 모멘텀의 적용 여부를 설정합니다.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/optimizers.py#L229)</span>
### RMSprop

```python
keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
```

RMSProp 최적화 함수.

이 최적화 함수를 사용할 때, 학습률을 제외한 모든 인자에서 기본값 사용을 권장합니다.

__인자__

- __learning_rate__: `float`>=0. 학습률.
- __rho__: `float`>=0.

__참고__

- [rmsprop: Divide the gradient by a running average of its recent magnitude](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/optimizers.py#L303)</span>
### Adagrad

```python
keras.optimizers.Adagrad(learning_rate=0.01)
```

Adagrad 최적화 함수.

Adagrad는 매개변수마다 다르게 적용되는 학습률이 사용되며,
매개변수의 값이 갱신되는 빈도에 따라 학습률이 결정됩니다.
매개변수가 더 자주 갱신될수록, 더 작은 학습률이 사용됩니다.

이 최적화 함수를 사용하는 경우, 기본값 사용을 권장합니다.

__인자__

- __learning_rate__: `float`>=0. 학습률.

__참고__

- [Adaptive Subgradient Methods for Online Learning and Stochastic
   Optimization](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/optimizers.py#L376)</span>
### Adadelta

```python
keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95)
```

Adadelta 최적화 함수.

Adadelta는 Adagrad를 확장하여 더욱 견고한 최적화함수입니다.
이전의 모든 그래디언트를 축적하는 대신, 그래디언트 갱신의 이동창<sub>moving window</sub>에 기반하여 학습률을 조절합니다.
이 방법을 사용함으로써 많은 갱신이 이뤄지더라도 학습을 계속할 수 있습니다.
Adagrad와 비교해 볼 때, Adadelta의 기존 버전에서는 초기 학습률을 설정할 필요가 없었습니다.
하지만 현재 버전에서는 다른 케라스 최적화함수와 같이 초기 학습률을 설정할 수 있습니다.

이 최적화 함수를 사용하는 경우, 기본값 사용을 권장합니다.

__인자__

- __learning_rate__: `float`>=0. 학습률. 초기 학습률은 1이며, 기본값 사용을 권장합니다.
- __rho__: `float`>=0. 학습률 감소에 쓰이는 값으로서, 
해당 시점에 유지되는 그래디언트 비율을 나타냅니다.

__참고__

- [Adadelta - an adaptive learning rate method](
   https://arxiv.org/abs/1212.5701)
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/optimizers.py#L467)</span>
### Adam

```python
keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
```

Adam 최적화 함수.

제공되는 모든 인자의 기본값은 논문에서 언급된 내용을 따릅니다.

__인자__

- __learning_rate__: `float`>=0. 학습률.
- __beta_1__: `float`, 0 < beta_1 < 1. 일반적으로 1에 가까운 수를 지정합니다.
- __beta_2__: `float`, 0 < beta_2 < 1. 일반적으로 1에 가까운 수를 지정합니다.
- __amsgrad__: `boolean`. Adam의 변형된 형태인 AMSGrad의 적용 여부를 설정합니다.
    AMSGrad는 "On the Convergence of Adam and Beyond" 논문에서 소개되었습니다.

__참조__

- [Adam - A Method for Stochastic Optimization](
   https://arxiv.org/abs/1412.6980v8)
- [On the Convergence of Adam and Beyond](
   https://openreview.net/forum?id=ryQu7f-RZ)
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/optimizers.py#L567)</span>
### Adamax

```python
keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
```

Adam 논문의 섹션 7에 소개된 Adamax 최적화 함수.

무한 노름(infinity norm)에 기반한 Adam의 변형된 형태입니다.
제공되는 모든 인자의 기본값은 논문에서 언급된 내용을 따릅니다.

__인자__

- __learning_rate__: `float`>=0. 학습률.
- __beta_1__: `float`, 0 < beta_1 < 1. 일반적으로 1에 가까운 수를 지정합니다.
- __beta_2__: `float`, 0 < beta_2 < 1. 일반적으로 1에 가까운 수를 지정합니다.


__참고__

- [Adam - A Method for Stochastic Optimization](
   https://arxiv.org/abs/1412.6980v8)
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/optimizers.py#L645)</span>
### Nadam

```python
keras.optimizers.Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999)
```

네스테로프 Adam 최적화 함수.

Adam이 본질적으로는 모멘텀이 적용된 RMSprop인 것처럼,
Nadam은 네스테로프 모멘텀이 적용된 RMSprop입니다.

제공되는 모든 인자의 기본값은 논문에서 언급된 내용을 따릅니다.
이 최적화 함수를 사용하는 경우, 기본값 사용을 권장합니다.

__인자__

- __learning_rate__: `float`>=0. 학습률.
- __beta_1__: `float`, 0 < beta_1 < 1. 일반적으로 1에 가까운 수를 지정합니다.
- __beta_2__: `float`, 0 < beta_2 < 1. 일반적으로 1에 가까운 수를 지정합니다.

__참고__

- [Nadam report](http://cs229.stanford.edu/proj2015/054_report.pdf)
- [On the importance of initialization and momentum in deep learning](
   http://www.cs.toronto.edu/~fritz/absps/momentum.pdf)
    

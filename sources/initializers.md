## 초기값 생성 함수 사용법<sub>Usage of initializers</sub>

초기값 생성 함수<sub>initializer</sub>는 케라스 층<sub>layer</sub>의 파라미터 초기값을 어떤 방식으로 생성할 것인지를 결정합니다. 모든 층에서 다 똑같은 것은 아니지만, 대부분의 경우 `kernel_initializer`와 `bias_initializer` 인자<sub>argument</sub>를 사용해서 가중치<sub>weight</sub>와 편향<sub>bias</sub>의 초기값 생성 함수를 지정합니다.

```python
model.add(Dense(64,
                kernel_initializer='random_uniform',
                bias_initializer='zeros'))
```

## 케라스가 제공하는 초기값 생성 함수<sub>Available initializers</sub>
아래의 함수들은 `keras.initializers` 모듈에 내장되어 있습니다.

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L14)</span>
### Initializer 
```python
keras.initializers.Initializer()
```
초기값 생성 함수의 기본 클래스로 모든 초기값 생성 함수는 이 클래스를 상속받습니다.

----
<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L33)</span>
### Zeros 
```python
keras.initializers.Zeros()
```
지정한 파라미터 값을 모두 `0`으로 생성합니다.

----
<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L41)</span>
### Ones
```python
keras.initializers.Ones()
```
지정한 파라미터 값을 모두 `1`로 생성합니다.

----
<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L49)</span>
### Constant
```python
keras.initializers.Constant(value=0)
```
지정한 파라미터를 특정한 상수(`value`)값으로 생성합니다.

__인자__
- __value__: `float`. 생성할 파라미터의 값입니다.
    
----
<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L66)</span>
### RandomNormal
```python
keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
```
지정한 평균 및 표준편차를 따르는 정규분포<sub>normal distribution</sub>로부터 대상 파라미터 값을 무작위로 생성합니다.

__인자__
- __mean__: `int` 혹은 `float`형식의 스칼라 값 또는 같은 형식의 스칼라 텐서. 파라미터 생성에 사용할 정규분포의 평균을 정합니다. 기본값은 `0`입니다.
- __stddev__: `int` 혹은 `float`형식의 스칼라 값 또는 같은 형식의 스칼라 텐서. 파라미터 생성에 사용할 정규분포의 표준편차를 정합니다. 기본값은 `0.05`입니다.
- __seed__: `int`. 무작위 생성에 사용할 시드를 정합니다. 기본값은 `None`입니다.
    
----
<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L94)</span>
### RandomUniform

```python
keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
```
지정한 범위의 균등분포<sub>uniform distribution</sub>로부터 대상 파라미터 값을 무작위로 생성합니다.

__인자__
- __minval__: `int` 혹은 `float`형식의 스칼라 값 또는 같은 형식의 스칼라 텐서. 무작위 생성에 사용할 수치 범위의 최솟값입니다. 기본값은 `-0.05`입니다.
- __maxval__: `int` 혹은 `float`형식의 스칼라 값 또는 같은 형식의 스칼라 텐서. 무작위 생성에 사용할 수치 범위의 최댓값입니다. 기본값은 `0.05`입니다.
- __seed__: `int`. 무작위 생성에 사용할 시드를 정합니다. 기본값은 `None`입니다.
    
----
<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L122)</span>
### TruncatedNormal
```python
keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)
```
정규분포 안의 일부 제한된 영역에서 파라미터 값을 무작위로 생성합니다. 정규분포로부터 무작위 값을 선택한다는 점에서 `RandomNormal`과 비슷하지만, 선택 범위를 평균으로부터 `2`표준편차 안쪽으로 제한합니다. 신경망의 가중치 또는 필터 초기값 생성에 쓸 방식으로 권장합니다.

__인자__
- __mean__: `int` 혹은 `float`형식의 스칼라 값 또는 같은 형식의 스칼라 텐서. 파라미터 생성에 사용할 정규분포의 평균을 정합니다. 기본값은 `0`입니다.
- __stddev__: `int` 혹은 `float`형식의 스칼라 값 또는 같은 형식의 스칼라 텐서. 파라미터 생성에 사용할 정규분포의 표준편차를 정합니다. 기본값은 `0.05`입니다.
- __seed__: `int`. 무작위 생성에 사용할 시드를 정합니다. 기본값은 `None`입니다.
    
----
<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L155)</span>
### VarianceScaling
```python
keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)
```
가중치 텐서의 규모에 따라 초기값 생성에 사용할 분포를 조정합니다.  

먼저 정규분포(`distribution="normal"`)를 선택할 경우, `0`을 평균으로 `sqrt(scale / n)`의 표준편차를 가진 정규분포의 `+-2`표준편차 범위 안에서 값을 선택합니다. 이때 `n`값은 `mode`인자 지정에 따라 다음과 같이 달라집니다.
- `mode = "fan_in"`의 경우 `n`은 가중치 텐서의 입력 차원 크기입니다. 
- `mode = "fan_out"`의 경우 `n`은 가중치 텐서의 출력 차원 크기입니다. 
- `mode = "fan_avg"`의 경우 입력 차원, 출력 차원 크기의 평균값이 됩니다.

또는 균등분포(`distribution="uniform"`)를 선택할 경우 `[-limit, limit]`의 범위를 가진 균등분포로부터 값이 선택됩니다. 여기서 `limit`은 `sqrt(3 * scale / n)`으로 정의됩니다. 위와 마찬가지로 `n`값은 `mode`인자에 따라 정해집니다.

__인자__
- __scale__: 스케일링 팩터(양의 `float`)입니다. 기본값은 `1.0`입니다.
- __mode__: `"fan_in"`, `"fan_out"`, `"fan_avg"` 중 하나입니다. 기본값은 `"fan_in"`입니다.
- __distribution__: `"normal"`과 `"uniform"` 가운데에서 분포의 종류를 정합니다. 기본값은 `"normal"`입니다.
- __seed__: `int`. 무작위 생성에 사용할 시드를 정합니다. 기본값은 `None`입니다.

__오류__
- __ValueError__: `"scale"`, `"mode"` 혹은 `"distribution"` 인자에 잘못된 값을 전달한 경우 발생합니다.
    
----
<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L229)</span>
### Orthogonal
```python
keras.initializers.Orthogonal(gain=1.0, seed=None)
```
무작위 직교행렬<sub>orthogonal matrix</sub>을 파라미터의 초기값으로 생성합니다. 먼저 평균`0` 표준편차`1`의 정규분포로부터 무작위 텐서를 생성한 뒤, 해당 텐서를 특이값분해<sub>singular value decomposition</sub>하여 얻은 직교행렬을 원하는 파라미터의 형태에 맞게 변형하여 사용합니다. 직교행렬의 가중치 초기값 활용에 대해서는 다음의 [논문](http://arxiv.org/abs/1312.6120)을 참고하십시오.

__인자__
- __gain__: 생성될 직교행렬에 곱할 배수입니다. 기본값은 `1.`입니다.
- __seed__: `int`. 무작위 생성에 사용할 시드를 정합니다. 기본값은 `None`입니다.

__참조__
- [Exact solutions to the nonlinear dynamics of learning in deep linear neural networks](http://arxiv.org/abs/1312.6120)
    
----
<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L267)</span>
### Identity
```python
keras.initializers.Identity(gain=1.0)
```
지정한 파라미터를 단위행렬로 생성합니다. 단위행렬의 특성상 2D 행렬에만 사용할 수 있습니다. 만약 사용하고자 하는 행렬이 정사각형이 아니라면 `0`을 채운 행 또는 열을 추가하여 형태를 맞춥니다. 

__인자__
- __gain__: 생성할 단위행렬에 곱할 배수입니다. 기본값은 `1.`입니다.
    
  
----
### glorot_normal
```python
keras.initializers.glorot_normal(seed=None)
```
Glorot 정규분포 방식으로 파라미터의 초기값을 생성합니다. Xavier 정규분포 방식이라고도 불리며, 가중치 텐서의 크기에 따라 값을 조절하는 방식의 하나입니다.  
  
가중치 텐서의 입력 차원 크기를 `fan_in`, 출력 차원 크기를 `fan_out`이라고 할 때, 평균 `0`과 `sqrt(2 / (fan_in + fan_out))`의 표준편차를 가진 정규분포의 `+-2`표준편차 범위 안에서 무작위로 값을 추출합니다.

__인자__
- __seed__: `int`. 무작위 생성에 사용할 시드를 정합니다. 기본값은 `None`입니다.

__반환값__  

Glorot 정규분포 방식을 따르는 초기값 생성 함수.

__참조__
- [Understanding the difficulty of training deep feedforward neural networks](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)
    
----
### glorot_uniform
```python
keras.initializers.glorot_uniform(seed=None)
```
Glorot 균등분포 방식으로 파라미터의 초기값을 생성합니다. Xavier 균등분포 방식이라고도 불리며, 가중치 텐서의 크기에 따라 값을 조절하는 방식의 하나입니다.  

`[-limit, limit]`의 범위를 가진 균등분포로부터 값이 선택됩니다. 가중치 텐서의 입력 차원 크기를 `fan_in`, 출력 차원 크기를 `fan_out`이라고 할 때, `limit`은 `sqrt(6 / (fan_in + fan_out))`으로 구합니다.

__인자__
- __seed__: `int`. 무작위 생성에 사용할 시드를 정합니다. 기본값은 `None`입니다.

__반환값__  

Glorot 균등분포 방식을 따르는 초기값 생성 함수.

__참조__
- [Understanding the difficulty of training deep feedforward neural networks](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)
    
----
### he_normal
```python
keras.initializers.he_normal(seed=None)
```
He 정규분포 방식으로 파라미터의 초기값을 생성합니다. 가중치 텐서의 크기에 따라 값을 조절하는 방식의 하나입니다.  
  
가중치 텐서의 입력 차원 크기를 `fan_in`이라고 할 때, 평균 `0`과 `sqrt(2 / fan_in)`의 표준편차를 가진 정규분포의 `+-2`표준편차 범위 안에서 무작위로 값을 추출합니다.

__인자__
- __seed__: `int`. 무작위 생성에 사용할 시드를 정합니다. 기본값은 `None`입니다.

__반환값__  

He 정규분포 방식을 따르는 초기값 생성 함수.

__참조__
- [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](http://arxiv.org/abs/1502.01852)

----
### he_uniform
```python
keras.initializers.he_uniform(seed=None)
```
He 정규분포 방식으로 파라미터의 초기값을 생성합니다. 가중치 텐서의 크기에 따라 값을 조절하는 방식의 하나입니다.  
  
`[-limit, limit]`의 범위를 가진 균등분포로부터 값이 선택됩니다. 가중치 텐서의 입력 차원 크기를 `fan_in`이라고 할 때, `limit`은 `sqrt(6 / fan_in)`으로 구합니다.

__인자__
- __seed__: `int`. 무작위 생성에 사용할 시드를 정합니다. 기본값은 `None`입니다.

__반환값__  

He 균등분포 방식을 따르는 초기값 생성 함수.

__참조__
- [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](http://arxiv.org/abs/1502.01852)

----
### lecun_normal
```python
keras.initializers.lecun_normal(seed=None)
```
LeCun 정규분포 방식으로 파라미터의 초기값을 생성합니다. 가중치 텐서의 크기에 따라 값을 조절하는 방식의 하나입니다.  
  
가중치 텐서의 입력 차원 크기를 `fan_in`이라고 할 때, 평균 `0`과 `sqrt(1 / fan_in)`의 표준편차를 가진 정규분포의 `+-2`표준편차 범위 안에서 무작위로 값을 추출합니다.

__인자__
- __seed__: `int`. 무작위 생성에 사용할 시드를 정합니다. 기본값은 `None`입니다.

__반환값__  

LeCun 정규분포 방식을 따르는 초기값 생성 함수.

__참조__
- [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
- [Efficient Backprop](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)

----
### lecun_uniform
```python
keras.initializers.lecun_uniform(seed=None)
```
LeCun 정규분포 방식으로 파라미터의 초기값을 생성합니다. 가중치 텐서의 크기에 따라 값을 조절하는 방식의 하나입니다.  
  
`[-limit, limit]`의 범위를 가진 균등분포로부터 값이 선택됩니다. 가중치 텐서의 입력 차원 크기를 `fan_in`이라고 할 때, `limit`은 `sqrt(3 / fan_in)`으로 구합니다.

__인자__
- __seed__: `int`. 무작위 생성에 사용할 시드를 정합니다. 기본값은 `None`입니다.

__반환값__  

LeCun 균등분포 방식을 따르는 초기값 생성 함수.

__참조__
- [Efficient BackProp](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)

    
----
신경망의 각 층에서 초기값 생성 함수는 해당 초기값 생성 함수의 이름과 동일한 문자열<sub>string</sub> 또는 호출 가능한 함수<sub>callable</sub>의 형태로 지정할 수 있습니다. 

```python
from keras import initializers

model.add(Dense(64, kernel_initializer=initializers.random_normal(stddev=0.01))) # 함수 호출로 지정합니다.

# also works; will use the default parameters.
model.add(Dense(64, kernel_initializer='random_normal'))                         # 문자열로 지정합니다.
```

----
## 사용자 정의 초기값 생성 함수<sub>Using custom initializers</sub>
사용자 정의 함수를 만들어 사용하는 경우, 인자로 `shape`(초기값을 생성할 파라미터의 형태)와 `dtype`(생성할 값의 자료형)을 전달받아야 합니다.

```python
from keras import backend as K

def my_init(shape, dtype=None):                 # 형태와 자료형 인자를 명시합니다.
    return K.random_normal(shape, dtype=dtype)

model.add(Dense(64, kernel_initializer=my_init))
```

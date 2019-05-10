## 초기값 설정기의 사용법

초기값 설정은 케라스 레이어의 초기 난수 가중치를 설정하는 방식을 규정합니다.

초기값 설정기를 레이어에 전달하는 키워드 인수는 레이어 종류에 따라 다릅니다. 보통은 간단히 `kernel_initializer`와 `bias_initializer`를 사용합니다:

```python
model.add(Dense(64,
                kernel_initializer='random_uniform',
                bias_initializer='zeros'))
```

## 사용가능한 초기값 설정기

`keras.initializers` 모듈의 일부로 내장된 다음과 같은 초기값 설정기를 사용하실 수 있습니다:

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L14)</span>
### 초기값 설정기

```python
keras.initializers.Initializer()
```

초기값 설정기 베이스 클래스: 모든 초기값 설정기가 이 클래스에서 상속받습니다.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L33)</span>
### Zeros

```python
keras.initializers.Zeros()
```

모든 값이 0인 텐서를 생성하는 초기값 설정기입니다.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L41)</span>
### Ones

```python
keras.initializers.Ones()
```

모든 값이 1인 텐서를 생성하는 초기값 설정기입니다.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L49)</span>
### Constant

```python
keras.initializers.Constant(value=0)
```

모든 값이 특정 상수인 텐서를 생성하는 초기값 설정기입니다.

__인수__

- __value__: float; 생성 텐서의 값.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L66)</span>
### RandomNormal

```python
keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
```

정규분포에 따라 텐서를 생성하는 초기값 설정기입니다.

__인수__

- __mean__: 파이썬 스칼라 혹은 스칼라 텐서. 생성할 난수값의 평균입니다.
- __stddev__: 파이썬 스칼라 혹은 스칼라 텐서. 생성할 난수값의 표준편차입니다.
- __seed__: 파이썬 정수. 난수 생성기에 시드를 전달하는데 사용됩니다.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L94)</span>
### RandomUniform

```python
keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
```

균등분포에 따라 텐서를 생성하는 초기값 설정기입니다.

__인수__

- __minval__: 파이썬 스칼라 혹은 스칼라 텐서. 난수값을 생성할 범위의 하한선입니다.
- __maxval__: 파이썬 스칼라 혹은 스칼라 텐서. 난수값을 생성할 범위의 상한선입니다. float 자료형의 경우 디폴트값은 1입니다.
- __seed__: 파이썬 정수. 난수 생성기에 시드를 전달하는데 사용됩니다.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L122)</span>
### TruncatedNormal

```python
keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)
```

절단된 정규분포에 따라 텐서를 생성하는 초기값 설정기입니다.

이 값은 `RandomNormal`에서 생성된 값과 비슷하지만
평균으로부터 두 표준편차 이상 차이나는 값은 폐기 후 다시 뽑습니다.
이 초기값 설정기는 신경망 가중치와 필터에
사용할 것을 권장드립니다.

__인수__

- __mean__: 파이썬 스칼라 혹은 스칼라 텐서. 생성할 난수값의
  평균입니다.
- __stddev__: 파이썬 스칼라 혹은 스칼라 텐서. 생성할 난수값의
  표준편차입니다.
- __seed__: 파이썬 정수. 난수 생성기에 시드를 전달하는데 사용됩니다.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L155)</span>
### VarianceScaling

```python
keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)
```

가중치의 형태에 따라 규모를 조절할 수 있는 초기값 설정기입니다.

`distribution="normal"`로 분포를 설정 시, 0을 중심으로 `stddev = sqrt(scale / n)`의
표준편차를 가진 절단된 정규분포에 따라 샘플이 생성되는데, 여기서 n이란:

- mode = "fan_in"의 경우 가중치 텐서의 입력 유닛의 수입니다.
- mode = "fan_out"의 경우 가중치 텐서의 출력 유닛의 수입니다.
- mode = "fan_avg"의 경우 입력 유닛과 출력 유닛의 수의 평균을 말합니다.

`distribution="uniform"`로 분포를 설정 시,
[-limit, limit]의 범위내 균등분포에 따라 샘플이 생성되는데,
여기서 limit은 `limit = sqrt(3 * scale / n)`로 정의됩니다.

__인수__

- __scale__: 스케일링 인수 (양수 float).
- __mode__: "fan_in", "fan_out", 그리고 "fan_avg" 중 하나.
- __distribution__: 사용할 무작위 분포. "normal"과 "uniform" 중 하나입니다.
- __seed__: 파이썬 정수. 난수 생성기에 시드를 전달하는데 사용됩니다.

__오류보고__

- __ValueError__: "scale", mode" 혹은 "distribution" 인수에 무효한 값을
  전달했을 때 오류를 알려줍니다.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L229)</span>
### Orthogonal

```python
keras.initializers.Orthogonal(gain=1.0, seed=None)
```

무작위 직교행렬을 생성하는 초기값 설정기입니다.

__인수__

- __gain__: 직교행렬에 적용할 승법 인수입니다.
- __seed__: 파이썬 정수. 난수 생성기에 시드를 전달하는데 사용됩니다.

__참조__

- [Exact solutions to the nonlinear dynamics of learning in deep
   linear neural networks](http://arxiv.org/abs/1312.6120)
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L267)</span>
### Identity

```python
keras.initializers.Identity(gain=1.0)
```

단위행렬을 생성하는 초기값 설정기입니다.

2D 행렬에만 사용할 수 있습니다.
만약 사용하고자 하는 행렬이 정사각형이 아니라면,
추가적으로 행/열을 만들어 0을 채웁니다.

__인수__

- __gain__: 단위행렬에 적용할 승법 인수입니다.
    
----

### lecun_uniform


```python
keras.initializers.lecun_uniform(seed=None)
```


LeCun 균등분포 초기값 설정기.

[-limit, limit]의 범위내 균등분포에 따라 샘플이 생성되는데,
여기서 `limit`은 `sqrt(3 / fan_in)`으로 정의되고,
`fan_in`은 가중치 텐서의 입력 유닛의 수를 말합니다.

__인수__

- __seed__: 파이썬 정수. 난수 생성기에 시드를 전달하는데 사용됩니다.

__반환값__

초기값 설정기.

__참조__

- [Efficient BackProp](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
    
----

### glorot_normal


```python
keras.initializers.glorot_normal(seed=None)
```


Glorot 정규분포 초기값 설정기, Xavier 정규분포 초기값 설정기라고도 합니다.

0을 중심으로 `stddev = sqrt(2 / (fan_in + fan_out))`의 표준편차를 가진 
절단된 정규분포에 따라 샘플이 생성되는데, 
여기서 `fan_in`이란 가중치 텐서의 입력 유닛의 수를,
`fan_out`은 가중치 텐서의 출력 유닛의 수를 의미합니다.

__인수__

- __seed__: 파이썬 정수. 난수 생성기에 시드를 전달하는데 사용됩니다.

__반환값__

초기값 설정기.

__참조__

- [Understanding the difficulty of training deep feedforward neural
   networks](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)
    
----

### glorot_uniform


```python
keras.initializers.glorot_uniform(seed=None)
```


Glorot 균등분포 초기값 설정기, Xavier 균등분포 초기값 설정기라고도 합니다.

[-limit, limit]의 범위내 균등분포에 따라 샘플이 생성되는데,
여기서 `limit`은 `sqrt(6 / (fan_in + fan_out))`로 정의되고
`fan_in`은 가중치 텐서의 입력 유닛의 수를,
`fan_out`은 가중치 텐서의 출력 유닛의 수를 의미합니다.

__인수__

- __seed__: 파이썬 정수. 난수 생성기에 시드를 전달하는데 사용됩니다.

__반환값__

초기값 설정기.

__참조__

- [Understanding the difficulty of training deep feedforward neural
   networks](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)
    
----

### he_normal


```python
keras.initializers.he_normal(seed=None)
```


He 정규분포 초기값 설정기.

0을 중심으로 `stddev = sqrt(2 / fan_in)`의 표준편차를 가진
절단된 정규분포에 따라 샘플이 생성되는데,
여기서 `fan_in`이란 가중치 텐서의 입력 유닛의 수를 의미합니다.

__인수__

- __seed__: 파이썬 정수. 난수 생성기에 시드를 전달하는데 사용됩니다.

__반환값__

초기값 설정기.

__참조__

- [Delving Deep into Rectifiers: Surpassing Human-Level Performance on
   ImageNet Classification](http://arxiv.org/abs/1502.01852)
    
----

### lecun_normal


```python
keras.initializers.lecun_normal(seed=None)
```


LeCun 정규분포 초기값 설정기

0을 중심으로 `stddev = sqrt(1 / fan_in)`의 표준편차를 가진 
절단된 정규분포에 따라 샘플이 생성되는데,
여기서 `fan_in`이란 가중치 텐서의 입력 유닛의 수를 의미합니다.

__인수__

- __seed__: 파이썬 정수. 난수 생성기에 시드를 전달하는데 사용됩니다.

__반환값__

초기값 설정기.

__참조__

- [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
- [Efficient Backprop](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
    
----

### he_uniform


```python
keras.initializers.he_uniform(seed=None)
```


He 균등분포 분산 스케일링 초기값 설정기.

[-limit, limit]의 범위내 균등분포에 따라 샘플이 생성되는데,
여기서 `limit`은 `sqrt(6 / fan_in)`으로 정의되고,
`fan_in`은 가중치 텐서의 입력 유닛의 수를 의미합니다.

__인수__

- __seed__: 파이썬 정수. 난수 생성기에 시드를 전달하는데 사용됩니다.

__반환값__

초기값 설정기.

__참조__

- [Delving Deep into Rectifiers: Surpassing Human-Level Performance on
   ImageNet Classification](http://arxiv.org/abs/1502.01852)
    


초기값 설정기는 문자열(위의 사용가능한 초기값 설정기와 매치되어야 합니다)이나 callable 로 전달할 수 있습니다:

```python
from keras import initializers

model.add(Dense(64, kernel_initializer=initializers.random_normal(stddev=0.01)))

# also works; will use the default parameters.
model.add(Dense(64, kernel_initializer='random_normal'))
```


## 커스텀 초기값 설정기

커스텀 callable을 사용하는 경우, 인수로 `shape`(초기값 설정할 변수의 형태)과 `dtype`(생성된 값의 자료형)을 전달받아야 합니다:

```python
from keras import backend as K

def my_init(shape, dtype=None):
    return K.random_normal(shape, dtype=dtype)

model.add(Dense(64, kernel_initializer=my_init))
```

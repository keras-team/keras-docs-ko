## 이니셜라이저 사용법

초기화는 케라스 레이어의 랜덤한 초기값 지정을 세팅하는 방법을 정의합니다.

이니셜라이저를 인자로 넘겨주기 위해 쓰이는 키워드는 레이어에 따라 다릅니다. 보통, 간단하게는 `kernel_initializer` 와 `bias_initializer` 입니다:

```python
model.add(Dense(64,
                kernel_initializer='random_uniform',
                bias_initializer='zeros'))
```

## 사용가능한 이니셜라이저들

다음의 내장 이니셜라이저들은 `keras.initializers` 모듈의 일부로써 사용가능합니다.

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L14)</span>
### Initializer

```python
keras.initializers.Initializer()
```

이니셜라이저 기본 클래스: 모든 이니셜라이저는 클래스로부터 상속받습니다.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L33)</span>
### Zeros

```python
keras.initializers.Zeros()
```
텐서를 모두 0으로 세팅하는 이니셜라이저.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L41)</span>
### Ones

```python
keras.initializers.Ones()
```

텐서를 모두 1로 세팅하는 이니셜라이저.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L49)</span>
### Constant

```python
keras.initializers.Constant(value=0)
```

텐서를 특정값으로 세팅하는 이니셜라이저.



__Arguments__

- __value__: float; 제너레이터 텐서의 값
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L66)</span>
### RandomNormal

```python
keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
```

텐서를 정규분포로 채우는 이니셜라이저.

__Arguments__

- __mean__: 파이썬 스칼라 또는 스칼라 텐서. 생성된 랜덤값의 평균.
- __stddev__: 파이썬 스칼라 또는 스칼라 텐서. 랜덤값의 표준편차.
- __seed__: 파이썬 정수. 랜덤값 생성시의 seed를 위한 값.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L94)</span>
### RandomUniform

```python
keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
```

텐서를 uniform(균일) 분포로 채운 이니셜라이저.

__Arguments__

- __minval__: 파이썬 스칼라 또는 스칼라 텐서. 생성된 랜덤 값 범위의 하한.
- __maxval__: 파이썬 스칼라 또는 스칼라 텐서. 생성된 랜덤 값 범위의 상한, 기본값은 float 타입 1.
- __seed__: 파이썬 정수. 랜덤값 생성시의 seed를 위한 값.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L122)</span>
### TruncatedNormal

```python
keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)
```

Truncated 정규 분포를 생성하기 위한 이니셜라이저.

이 값들은 2배의 표준편차 이상의 값이 폐기되고 다시 구해진다는 점을 빼면 `RandomNormal`에서 뽑힌 값들과 유사합니다. 
본 이니셜라이저는 신경망의 웨이트와 필터 세팅에 추천합니다.

__Arguments__

- __mean__: 파이썬 스칼라 또는 스칼라 텐서. 랜덤값의 평균.
- __stddev__: 파이썬 스칼라 또는 스칼라 텐서. 랜덤값의 표준편차.
- __seed__: 파이썬 정수. 랜덤값 생성시의 seed를 위한 값.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L155)</span>
### VarianceScaling

```python
keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)
```

웨이트의 모양에 맞게 눈금 크기를 조절할 수 있는 이니셜라이저.

`distribution="normal"` 옵션은 샘플들이 중점이 0 인 truncated 정규분포로 선택되며, 표준편차에서 `stddev = sqrt(scale / n)` n은 다음을 뜻합니다:

- mode 가 "fan\_in" 일 때의 웨이트 텐서에서 입력단위의 개수
- mode 가 "fan\_out" 일 때의 출력단위의 개수
- mode 가 "fan\_avg" 일 때의 입력과 출력의 평균

`distribution="uniform"` 옵션에서 샘플들은 [-limit, limit], `limit = sqrt(3 * scale / n)` 범위에서 균등분포로 선택됩니다.

__Arguments__

- __scale__: 스케일링 인자(양수 float 타입).
- __mode__: "fan\_in", "fan\_out", "fan\_avg" 셋 중 하나.
- __distribution__: 사용할 랜덤 분포 선택. "normal", "uniform"중 하나.
- __seed__: 파이썬 정수. 랜덤값 생성시의 seed를 위한 값.

__Raises__

- __ValueError__: "scale", "mode" 또는 "distribution" 인자에서 유효하지 않은 값이 입력된 경우 발생.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L229)</span>
### Orthogonal

```python
keras.initializers.Orthogonal(gain=1.0, seed=None)
```

랜덤 직교행렬을 생성하는 이니셜라이저.

__Arguments__

- __gain__: 직교행렬에 적용할 곱셈인자.
- __seed__: 파이썬 정수. 랜덤값 생성시의 seed를 위한 값.

__References__

- [Exact solutions to the nonlinear dynamics of learning in deep
   linear neural networks](http://arxiv.org/abs/1312.6120)
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L267)</span>
### Identity

```python
keras.initializers.Identity(gain=1.0)
```

단위행렬을 생성하는 이니셜라이저.

2차원 행렬에서만 사용가능.
만일 원하는 행렬이 정사각형이 아닌 경우에 행렬에 0이 채워집니다.

__Arguments__

- __gain__: 단위행렬에 적용할 곱셈인자.
    
----

### lecun_uniform


```python
keras.initializers.lecun_uniform(seed=None)
```

LeCun 균등 이니셜라이저.

[-limit, limit]에서 균등 분포로 샘플을 선택합니다. 단, `limit`은 `sqrt(3 / fan_in)`이고, `fan_in`은 웨이트 텐서의 입력단위의 개수입니다.

__Arguments__

- __seed__: 파이썬 정수. 랜덤값 생성시의 seed를 위한 값.

__Returns__

이니셜라이저를 리턴합니다.

__References__

- [Efficient BackProp](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
    
----

### glorot_normal


```python
keras.initializers.glorot_normal(seed=None)
```
Glorot 정규 이니셜라이저 또는 Xavier 정규 이니셜라이저.

표준편차가 `stddev = sqrt(2 / (fan_in + fan_out))`이고 중심이 0인 Truncated 정규분포로 샘플을 선택합니다.
단, `fan_in`은 웨이트 텐서의 입력단위의 개수이고, `fan_out`은 웨이트 텐서의 출력단위의 개수입니다.

__Arguments__

- __seed__: 파이썬 정수. 랜덤값 생성시의 seed를 위한 값.

__Returns__

이니셜라이저를 리턴합니다.

__References__

- [Understanding the difficulty of training deep feedforward neural
   networks](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)
    
----

### glorot_uniform


```python
keras.initializers.glorot_uniform(seed=None)
```

Glorot 균등 이니셜라이저 또는 Xavier 균등 이니셜라이저

[-limit, limit] 구간에서 균등 분포로 생성된 값을 선택합니다.
단, `limit`은 `sqrt(6 / (fan_in + fan_out))`이고 
`fan_in`은 웨이트 텐서의 입력단위의 개수, 
`fan_out`은 웨이트 텐서의 출력단위의 개수입니다.

__Arguments__

- __seed__: 파이썬 정수. 랜덤값 생성시의 seed를 위한 값.

__Returns__

이니셜라이저를 리턴합니다.

__References__

- [Understanding the difficulty of training deep feedforward neural
   networks](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)
    
----

### he_normal


```python
keras.initializers.he_normal(seed=None)
```


He 정규 이니셜라이저.

표준편차가 `stddev = sqrt(2 / fan_in)`로 정의되고 0 중심의 truncated 정규 분포에서 샘플을 선택합니다.
단, `fan_in`는 웨이트 텐서의 입력단위의 개수입니다.


__Arguments__

- __seed__: 파이썬 정수. 랜덤값 생성시의 seed를 위한 값.

__Returns__

이니셜라이저를 리턴합니다.

__References__

- [Delving Deep into Rectifiers: Surpassing Human-Level Performance on
   ImageNet Classification](http://arxiv.org/abs/1502.01852)
    
----

### lecun_normal


```python
keras.initializers.lecun_normal(seed=None)
```

LeCun 정규 이니셜라이저.


표준편차가 `stddev = sqrt(1 / fan_in)`로 정의되고 0 중심의 truncated 정규 분포에서 샘플을 선택합니다.
단 `fan_in`는 웨이트 텐서의 입력단위의 개수입니다.

__Arguments__

- __seed__: 파이썬 정수. 랜덤값 생성시의 seed를 위한 값.

__Returns__

이니셜라이저를 리턴합니다.

__References__

- [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
- [Efficient Backprop](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
    
----

### he_uniform


```python
keras.initializers.he_uniform(seed=None)
```

He 균등 분산 스케일링 이니셜라이저.

[-limit, limit]구간에서 균등분포로 샘플을 선택합니다.
단, `limit`은 `sqrt(6 / fan_in)`이고,
`fan_in`은 웨이트 텐서의 입력단위의 개수입니다.

__Arguments__

- __seed__: 파이썬 정수. 랜덤값 생성시의 seed를 위한 값.

__Returns__

이니셜라이저를 리턴합니다.

__References__

- [Delving Deep into Rectifiers: Surpassing Human-Level Performance on
   ImageNet Classification](http://arxiv.org/abs/1502.01852)
    

이니셜라이저는 문자열(위에 사용가능한 이니셜라이저중 하나와 일치해야 함)이나 호출가능 매개변수로 전달할 수 있습니다.:


```python
from keras import initializers

model.add(Dense(64, kernel_initializer=initializers.random_normal(stddev=0.01)))

# also works; will use the default parameters.
model.add(Dense(64, kernel_initializer='random_normal'))
```


## 사용자 커스텀 이니셜라이저 사용하기


커스텀 호출가능한 인자를 전달할 때 반드시 `shape`(초기화할 변수의 shape)와 `dtype`(생성된 값의 dtype)인자를 취할 수 있도록 해야 합니다.:

```python
from keras import backend as K

def my_init(shape, dtype=None):
    return K.random_normal(shape, dtype=dtype)

model.add(Dense(64, kernel_initializer=my_init))
```

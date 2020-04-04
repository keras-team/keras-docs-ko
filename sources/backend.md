# 케라스 백엔드

## "백엔드"는 무엇인가요? 

케라스는 딥러닝 모델을 블록쌓기처럼 구성요소를 쌓아 만들 수 있게끔 해주는 높은 차원의 모델 수준 라이브러리입니다. 케라스는 단독으로는 텐서들의 곱셈이나 합성곱<sub>Convolution</sub>과 같은 낮은 차원의 직접적인 연산을 지원하지 않습니다. 대신에 텐서 연산에 특화된 라이브러리들을 가져와 "백엔드 엔진"으로 사용합니다. 케라스는 하나의 텐서 연산 라이브러리를 골라 그에 의지하는 대신 모듈 방식으로 문제를 처리하여 서로 다른 백엔드 엔진을 케라스와 매끄럽게 연결하게끔 합니다.

현재 케라스는 **TensorFlow**, **Theano**, 그리고 **CNTK**의 세 가지 백엔드를 지원합니다.
- [TensorFlow](http://www.tensorflow.org/)는 구글에서 만든 오픈소스 심볼릭 텐서 연산 프레임워크입니다.
- [Theano](http://deeplearning.net/software/theano/)은 Université de Montréal LISA Lab에서 개발한 오픈소스 심볼릭 텐서 연산 프레임워크입니다. 
- [CNTK](https://www.microsoft.com/en-us/cognitive-toolkit/)마이크로소프트에서 만든 딥러닝 개발용 오픈소스 툴킷입니다.

앞으로 더 많은 백엔드 옵션을 지원할 예정입니다.


----
## 한 백엔드에서 다른 백엔드로의 전환

케라스를 한 번이라도 실행한 적이 있다면 아래의 위치에서 케라스 설정 파일을 찾을 수 있습니다. **윈도우<sub>Windows</sub> 사용자**의 경우 주소의 `$HOME`을 `%USERPROFILE%`로 바꿔 넣어주시면 됩니다. 

`$HOME/.keras/keras.json`

만약 파일이 없는 경우 해당 위치에 설정 파일을 만들 수 있습니다.  

기본 설정 파일의 내용은 다음과 같습니다.

```
{
    "image_data_format": "channels_last",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}
```

`backend` 필드의 값을 `"theano"`, `"tensorflow"` 또는 `"cntk"`로 바꿔주는 것만으로 새로운 백엔드를 사용해 케라스 코드를 실행할 수 있습니다. 

또는 아래와 같이 환경 변수 `KERAS_BACKEND`를 정의해 설정 파일의 내용을 대체할 수도 있습니다. 

```bash
KERAS_BACKEND=tensorflow python -c "from keras import backend"
Using TensorFlow backend.
```

Keras에서는 `"tensorflow"`, `"theano"` 그리고 `"cntk"`외에도 사용자가 지정한 임의의 백엔드를 사용할 수 있습니다. 만약 `my_module`이라는 이름의 Python 모듈을 백엔드로 사용하고자 한다면 `keras.json` 파일의 `"backend"` 변수 값을 아래와 같이 바꿔주어야 합니다.  

```
{
    "image_data_format": "channels_last",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "my_package.my_module"
}
```

사용하고자 하는 외부 백엔드는 반드시 검증된 것이어야 하며 `placeholder`, `variable` 그리고 `function`의 세 함수들을 지원해야 합니다.

만약 필수 항목이 누락되어 백엔드를 사용할 수 없는 경우라면 어느 항목의 누락으로 오류가 발생했는지를 로그에 기록합니다.


----
## keras.json 상세

`keras.json` 설정 파일은 아래의 내용으로 이루어져 있습니다.

```
{
    "image_data_format": "channels_last",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}
```

각각의 설정은 `$HOME/.keras/keras.json` (윈도우 사용자의 경우 `%USERPROFILE%/.keras/keras.json`) 파일을 수정하여 변경할 수 있습니다.  

* `image_data_format`: `str`. 이미지 데이터 처리시 입력받을 데이터의 차원 순서를 정의하는 인자로 `'channels_last'` 또는 `'channels_first'` 가운데 하나를 지정합니다. (`keras.backend.image_data_format()`은 이 지정값을 반환합니다.)
  - 2D 데이터의 경우 (예: 이미지) `"channels_last"`는 `(rows, cols, channels)`의 차원 순서를, `"channels_first"`는 `(channels, rows, cols)`의 차원 순서를 따릅니다. 
  - 3D 데이터의 경우 `"channels_last"`는 `(conv_dim1, conv_dim2, conv_dim3, channels)`의 차원 순서를, `"channels_first"`는 `(channels, conv_dim1, conv_dim2, conv_dim3)`의 차원 순서를 따릅니다.
* `epsilon`: `float`. 0으로 나눗셈을 해서 발생하는 오류를 피하기 위해 특정 연산에서 분모에 더할 작은 상수값을 정합니다.
* `floatx`: `str`. `"float16"`, `"float32"`, 또는 `"float64"` 가운데 실수값의 기본 부동소수점 단위를 지정합니다.
* `backend`: `str`. `"tensorflow"`, `"theano"`, 또는 `"cntk"` 가운데 기본 백엔드를 지정합니다.

----

## 추상화된 케라스 백엔드를 사용하여 새로운 코드 작성하기

만약 Theano(`th`)와 Tensorflow(`tf`) 모두와 호환이 되는 케라스 모듈을 작성하고자 한다면 아래와 같이 추상화된 백엔드 API를 사용해야 합니다. 

다음과 같이 백엔드 모듈을 불러올 수 있습니다.

```python
from keras import backend as K
```

아래는 입력의 `placeholder` 인스턴스를 만드는 코드입니다. `tf.placeholder()`, `th.tensor.matrix()` 또는 `th.tensor.tensor()` 등을 실행하는 것과 같습니다.

```python
inputs = K.placeholder(shape=(2, 4, 5))
# 이하 역시 작동합니다.
inputs = K.placeholder(shape=(None, 4, 5))
# 이하 역시 작동합니다.
inputs = K.placeholder(ndim=3)
```

아래는 변수<sub>variable</sub> 인스턴스를 만드는 코드입니다. `tf.Variable()` 또는 `th.shared()`를 실행하는 것과 같습니다.

```python
import numpy as np
val = np.random.random((3, 4, 5))
var = K.variable(value=val)

# 모든 초기값이 0인 변수를 생성.
var = K.zeros(shape=(3, 4, 5))
# 모든 초기값이 1인 변수를 생성.
var = K.ones(shape=(3, 4, 5))
```

구현에 필요한 대부분의 텐서 연산들은 사용법이 TensorFlow나 Theano와 크게 다르지 않습니다. 

```python
# 무작위 초기값을 가진 텐서를 생성하기.
b = K.random_uniform_variable(shape=(3, 4), low=0, high=1) # 균등 분포
c = K.random_normal_variable(shape=(3, 4), mean=0, scale=1) # 가우시안 분포
d = K.random_normal_variable(shape=(3, 4), mean=0, scale=1)

# 텐서 사칙연산
a = b + c * K.abs(d)
c = K.dot(a, K.transpose(b))
a = K.sum(b, axis=1)
a = K.softmax(b)
a = K.concatenate([b, c], axis=-1)
# 등등...
```


----
## 백엔드 함수들

### backend

```python
keras.backend.backend()
```

현재 사용중인 백엔드의 이름을 반환합니다.(예: "tensorflow")

__반환값__ 
  
`str`. 현재 사용 중인 케라스 백엔드 이름.

__예시__
```python
>>> keras.backend.backend()
'tensorflow'
```


---
### symbolic

```
keras.backend.symbolic(func)
```

텐서플로우 2.0에서 케라스 그래프로 진입하기 위해 사용하는 데코레이터 함수입니다.

__인자__
- __func__: 데코레이터를 적용할 함수입니다. 

__반환값__ 
    
데코레이터가 적용된 함수.


---
### eager

```
keras.backend.eager(func)
```
텐서플로우 2.0에서 케라스 그래프를 빠져나오기 위해 사용하는 함수입니다.

__인자__
- __func__: 데코레이터를 적용할 함수.

__반환값__ 
    
데코레이터가 적용된 함수.


----
### get_uid

```python
keras.backend.get_uid(prefix='')
```

그래프 안에서 사용할 문자열 접두어<sub>prefix</sub>에 정수형 고유값을 UID로 부여합니다. 

__인자__
- __prefix__: 그래프의 문자열 접두어.

__반환값__ 
    

고유 정수 식별자(uid)

__예시__
```
>>> keras.backend.get_uid('dense')
1
>>> keras.backend.get_uid('dense')
2
```


----
### reset_uids

```python
keras.backend.reset_uids()
```

그래프의 식별자(uid)를 초기화합니다.


----
### manual_variable_initialization

```python
keras.backend.manual_variable_initialization(value)
```

변수 초기화의 수동/자동 여부를 결정하는 플래그 값입니다.  

변수를 인스턴스화 할 때 자동으로 초기화 할지, 별도로 사용자가 직접 초기화 할지를 결정합니다. 기본값은 자동 초기화입니다.

__인자__
- __value__: `bool`값.


----
### epsilon

```python
keras.backend.epsilon()
```

fuzz factor(엡실론: 0으로 나누는 오류를 방지하기 위해 분모에 더하는 작은 상수값)을 반환합니다.

__반환값__ 
    
단일 실수(부동소수점) 값.

__예시__
```python
>>> keras.backend.epsilon()
1e-07
```


----
### set_epsilon

```python
keras.backend.set_epsilon(e)
```

fuzz factor(엡실론: 0으로 나누는 오류를 방지하기 위해 분모에 더하는 작은 상수값)를 설정합니다.

__인자__
- __e__: `float`. 엡실론의 새로운 값.

__예시__
```python
>>> from keras import backend as K
>>> K.epsilon()
1e-07
>>> K.set_epsilon(1e-05)
>>> K.epsilon()
1e-05
```


----
### floatx

```python
keras.backend.floatx()
```

실수 표현에 사용하는 부동소수점 유형의 기본값을 문자열로 반환합니다. (예: 'float16', 'float32', 'float64').

__반환값__ 
    
`str`. 현재 설정된 부동소수점 유형의 기본값.

__예시__
```python
>>> keras.backend.floatx()
'float32'
```


----
### set_floatx

```python
keras.backend.set_floatx(floatx)
```

실수 표현에 사용할 부동소수점 유형의 기본값을 설정합니다.

__인자__
- __floatx__: `str`. 'float16', 'float32', 'float64' 가운데 하나를 입력합니다.

__예시__
```python
>>> from keras import backend as K
>>> K.floatx()
'float32'
>>> K.set_floatx('float16')
>>> K.floatx()
'float16'
```


----
### cast_to_floatx

```python
keras.backend.cast_to_floatx(x)
```

NumPy 배열을 케라스에 지정된 기본 실수형 타입으로 변환합니다.

__인자__
- __x__: NumPy 배열.

__반환값__ 
    
변환된 NumPy 배열.

__예시__

```python
>>> from keras import backend as K
>>> K.floatx()
'float32'
>>> arr = numpy.array([1.0, 2.0], dtype='float64')
>>> arr.dtype
dtype('float64')
>>> new_arr = K.cast_to_floatx(arr)
>>> new_arr
array([ 1.,  2.], dtype=float32)
>>> new_arr.dtype
dtype('float32')
```


----
### image_data_format

```python
keras.backend.image_data_format()
```

케라스가 처리할 이미지 데이터 유형의 기본값을 반환합니다.

__반환값__ 

`'channels_first'` 또는 `'channels_last'`의 문자열.

__예시__

```python
>>> keras.backend.image_data_format()
'channels_first'
```


----
### set_image_data_format

```python
keras.backend.set_image_data_format(data_format)
```

케라스가 처리할 이미지 데이터 유형의 기본값을 지정합니다. 데이터의 차원 순서를 정의하는 인자로, 예를 들어 2D 데이터의 경우 (예: 이미지) `"channels_last"`는 `(rows, cols, channels)`의 차원 순서를, `"channels_first"`는 `(channels, rows, cols)`의 차원 순서를 따릅니다. 3D 데이터의 경우 `"channels_last"`는 `(conv_dim1, conv_dim2, conv_dim3, channels)`의 차원 순서를, `"channels_first"`는 `(channels, conv_dim1, conv_dim2, conv_dim3)`의 차원 순서를 따릅니다.

__인자__

- __data_format__: `str`. `'channels_first'` 또는 `'channels_last'`.

__예시__

```python
>>> from keras import backend as K
>>> K.image_data_format()
'channels_first'
>>> K.set_image_data_format('channels_last')
>>> K.image_data_format()
'channels_last'
```


----
### learning_phase

```python
keras.backend.learning_phase()
```

학습 단계를 나타내는 플래그를 반환합니다.

`bool`형식의 텐서로, 학습할 때와 테스트할 때의 작동이 달라지는 모든 케라스 함수들에 전달되어 현재 어떤 단계에 있는지를 알립니다(0=시험, 1=학습).

__반환값__ 
    
학습 단계 (스칼라 정수 텐서 또는 파이썬 정수형).


----
### set_learning_phase

```python
keras.backend.set_learning_phase(value)
```

학습 단계 변수를 주어진 값으로 고정합니다.

__인자__
- __value__: 0 또는 1(`int`)의 학습 단계 값(0=시험, 1=학습). 

__오류__
- __ValueError__: `value` 가 `0` 또는 `1`이 아닌 경우.


----
### clear_session

```python
keras.backend.clear_session()
```

현재의 케라스 그래프를 없애고 새 그래프를 만듭니다. 오래된 모델/층과의 혼란을 피할 때 유용합니다.

    
----
### is_sparse

```python
keras.backend.is_sparse(tensor)
```

입력한 텐서가 희소<sub>sparse</sub> 텐서인지 아닌지를 확인합니다. 

__인자__
- __tensor__: 한 개의 텐서 인스턴스.

__반환값__ 
    
불리언.

__예시__

```python
>>> from keras import backend as K
>>> a = K.placeholder((2, 2), sparse=False)
>>> print(K.is_sparse(a))
False
>>> b = K.placeholder((2, 2), sparse=True)
>>> print(K.is_sparse(b))
True
```


----
### to_dense

```python
keras.backend.to_dense(tensor)
```

희소 텐서를 밀집<sub>dense</sub> 텐서로 바꿔줍니다.

__인자__
- __tensor__: (희소 형태일 가능성이 있는) 텐서 인스턴스.

__반환값__ 
    
한 개의 밀집 텐서.

__예시__

```python
>>> from keras import backend as K
>>> b = K.placeholder((2, 2), sparse=True)
>>> print(K.is_sparse(b))
True
>>> c = K.to_dense(b)
>>> print(K.is_sparse(c))
False
```


----
### variable

```python
keras.backend.variable(value, dtype=None, name=None, constraint=None)
```

변수 인스턴스를 생성합니다.

__인자__
- __value__: NumPy 배열, 텐서의 초기 값.
- __dtype__: 텐서의 자료형.
- __name__: 텐서의 이름(선택사항).
- __constraint__: 최적화 함수<sub>optimizer</sub>로 변수 값을 업데이트한 다음에 적용할 제약 함수입니다(선택사항).

__반환값__ 
    
(케라스 메타데이터가 포함된) 변수 인스턴스.

__예시__
```python
>>> from keras import backend as K
>>> val = np.array([[1, 2], [3, 4]])
>>> kvar = K.variable(value=val, dtype='float64', name='example_var')
>>> K.dtype(kvar)
'float64'
>>> print(kvar)
example_var
>>> K.eval(kvar)
array([[ 1.,  2.],
       [ 3.,  4.]])
```


----
### constant

```python
keras.backend.constant(value, dtype=None, shape=None, name=None)
```

상수 텐서를 만듭니다.

__인자__
- __value__: 상수 값(또는 리스트)
- __dtype__: 텐서의 자료형.
- __shape__: 텐서의 형태(선택사항).
- __name__: 텐서의 이름(선택사항).

__반환값__ 
    
상수 텐서
    
    
----
### is_keras_tensor

```python
keras.backend.is_keras_tensor(x)
```

`x`가 케라스 텐서인지 아닌지 여부를 확인합니다.

"케라스 텐서"란 케라스의 층(`Layer` 클래스) 또는 `Input`에 의해 반환된 텐서입니다.

__인자__
- __x__: 확인할 텐서.

__반환값__ 
    
주어진 인자가 케라스 텐서인지의 여부를 나타내는 `bool`값.

__오류__
- __ValueError__: `x`가 심볼릭 텐서가 아닌 경우.

__예시__
```python
>>> from keras import backend as K
>>> from keras.layers import Input, Dense
>>> np_var = numpy.array([1, 2])
>>> K.is_keras_tensor(np_var) # NumPy 배열은 심볼릭 텐서가 아니므로 오류를 반환합니다.
ValueError
>>> k_var = tf.placeholder('float32', shape=(1,1))
>>> # 케라스 바깥에서 생성된 변수는 케라스 텐서가 아닙니다.
>>> K.is_keras_tensor(k_var)
False
>>> keras_var = K.variable(np_var)
>>> # 케라스 층이 아닌 케라스 백엔드에 의해 생성된 변수는 케라스 텐서가 아닙니다.
>>> K.is_keras_tensor(keras_var)
False
>>> keras_placeholder = K.placeholder(shape=(2, 4, 5))
>>> # 플레이스홀더는 케라스 텐서가 아닙니다.
>>> K.is_keras_tensor(keras_placeholder)
False
>>> keras_input = Input([10])
>>> K.is_keras_tensor(keras_input) # Input은 케라스 텐서입니다.
True
>>> keras_layer_output = Dense(10)(keras_input)
>>> # 모든 케라스 층의 출력은 케라스 텐서입니다.
>>> K.is_keras_tensor(keras_layer_output)
True
```
    
    
----
### is_tensor

```python
keras.backend.is_tensor(x)
```

`x`가 현재 사용하고 있는 백엔드에서 생성한 텐서인지 여부를 확인합니다. 

__인자__
- __x__: 확인할 텐서.

__반환값__ 
    
주어진 인자가 백엔드의 텐서인지의 여부를 나타내는 `bool`값.


----
### placeholder

```python
keras.backend.placeholder(shape=None, ndim=None, dtype=None, sparse=False, name=None)
```

플레이스홀더 인스턴스를 생성합니다.

__인자__
- __shape__: 플레이스홀더의 형태. `int`로 이루어진 튜플로, `None`이 포함될 수 있습니다.
- __ndim__: 텐서의 축의 개수(=총 차원의 수). 인스턴스를 생성하려면 적어도 {`shape`, `ndim`} 중 하나는 반드시 입력해야 합니다. 두 인자를 다 입력한 경우 `shape`가 사용됩니다.
- __dtype__: 플레이스홀더의 자료형.
- __sparse__: `bool`. 플레이스홀더가 희소<sub>sparse</sub>타입이어야 하는지의 여부를 결정합니다.
- __name__: `str`. 플레이스홀더의 이름을 정합니다(선택사항).

__반환값__ 
    
(케라스 메타데이터가 포함된) 텐서 인스턴스.

__예시__
```python
>>> from keras import backend as K
>>> input_ph = K.placeholder(shape=(2, 4, 5))
>>> input_ph._keras_shape
(2, 4, 5)
>>> input_ph
<tf.Tensor 'Placeholder_4:0' shape=(2, 4, 5) dtype=float32>
```


----
### is_placeholder

```python
keras.backend.is_placeholder(x)
```

`x`가 플레이스홀더인지 아닌지 여부를 확인합니다.

__인자__
- __x__: 확인할 플레이스홀더.

__반환값__ 
    
주어진 인자가 플레이스홀더인지의 여부를 나타내는 `bool`값.


----
### shape

```python
keras.backend.shape(x)
```

텐서 또는 변수의 (심볼릭) 형태를 반환합니다.

__인자__
- __x__: 한 개의 텐서 또는 변수. 

__반환값__ 
    
(심볼릭) 형태 값을 저장한 텐서.

__예시__
```python
# TensorFlow 예시
>>> from keras import backend as K
>>> tf_session = K.get_session()
>>> val = np.array([[1, 2], [3, 4]])
>>> kvar = K.variable(value=val)
>>> inputs = keras.backend.placeholder(shape=(2, 4, 5))
>>> K.shape(kvar)
<tf.Tensor 'Shape_8:0' shape=(2,) dtype=int32>
>>> K.shape(inputs)
<tf.Tensor 'Shape_9:0' shape=(3,) dtype=int32>
# 형태값 텐서를 정수로 변환합니다(또는 K.int_shape(x)를 사용할 수도 있습니다).
>>> K.shape(kvar).eval(session=tf_session)
array([2, 2], dtype=int32)
>>> K.shape(inputs).eval(session=tf_session)
array([2, 4, 5], dtype=int32)
```
    
    
----
### int_shape

```python
keras.backend.int_shape(x)
```

텐서 또는 변수의 형태를 `int` 또는 `None` 값을 포함하는 튜플 형식으로 반환합니다.

__인자__
- __x__: 텐서 또는 변수. 

__반환값__ 
    
`int` 또는 `None`을 포함한 튜플.

__예시__
```python
>>> from keras import backend as K
>>> inputs = K.placeholder(shape=(2, 4, 5))
>>> K.int_shape(inputs)
(2, 4, 5)
>>> val = np.array([[1, 2], [3, 4]])
>>> kvar = K.variable(value=val)
>>> K.int_shape(kvar)
(2, 2)
```

__NumPy 구현__

```python
def int_shape(x): 
    return x.shape # int_shape의 출력값은 NumPy의 .shape 출력과 같습니다.
```


----
### ndim

```python
keras.backend.ndim(x)
```

텐서의 축의 개수(=총 차원의 수)를 정수값으로 반환합니다. 

__인자__
- __x__: 텐서 또는 변수.

__반환값__ 
    
축의 개수. 정수형(스칼라값)으로 반환합니다.

__예시__
```python
>>> from keras import backend as K
>>> inputs = K.placeholder(shape=(2, 4, 5))
>>> val = np.array([[1, 2], [3, 4]])
>>> kvar = K.variable(value=val)
>>> K.ndim(inputs)
3
>>> K.ndim(kvar)
2
```

__NumPy 구현__

```python
def ndim(x):
    return x.ndim
```


----
### size
```
keras.backend.size(x, name=None)
```

텐서의 크기(원소의 개수)를 반환합니다.

__인자__
- __x__:  텐서 또는 변수.
- __name__: 해당 연산에 이름을 부여합니다(선택사항).

__반환값__ 
    
입력 텐서의 원소 개수를 값으로 갖는 텐서.

__예시__
```
>>> from keras import backend as K
>>> val = np.array([[1, 2], [3, 4]]) # 원소의 개수는 4개입니다.
>>> kvar = K.variable(value=val)
>>> K.size(inputs)
<tf.Tensor: id=9, shape=(), dtype=int32, numpy=4> # 원소의 개수인 4의 값을 갖는 텐서를 반환합니다.
```


----
### dtype

```python
keras.backend.dtype(x)
```

케라스 텐서 또는 변수의 자료형을 문자열로 반환합니다.

__인자__
- __x__: 텐서 또는 변수. 

__반환값__ 
    
`str`. `x`의 자료형.

__예시__

```python
>>> from keras import backend as K
>>> K.dtype(K.placeholder(shape=(2,4,5)))
'float32'
>>> K.dtype(K.placeholder(shape=(2,4,5), dtype='float32'))
'float32'
>>> K.dtype(K.placeholder(shape=(2,4,5), dtype='float64'))
'float64'
# Keras variable
>>> kvar = K.variable(np.array([[1, 2], [3, 4]]))
>>> K.dtype(kvar)
'float32_ref'
>>> kvar = K.variable(np.array([[1, 2], [3, 4]]), dtype='float32')
>>> K.dtype(kvar)
'float32_ref'
```

__NumPy 구현__

```python
def dtype(x):
    return x.dtype.name
```


----
### eval

```python
keras.backend.eval(x)
```

변수(텐서)의 값을 NumPy 배열의 형태로 가져옵니다. 

__인자__
- __x__: 한 개의 변수. 

__반환값__ 
    
NumPy 배열.

__예시__
```python
>>> from keras import backend as K
>>> kvar = K.variable(np.array([[1, 2], [3, 4]]), dtype='float32')
>>> K.eval(kvar)
array([[ 1.,  2.],
       [ 3.,  4.]], dtype=float32)
```

__Numpy 적용__

```python
def eval(x):
    return x
```


----
### zeros

```python
keras.backend.zeros(shape, dtype=None, name=None)
```

모든 값이 0인 변수 인스턴스를 생성합니다.

__인자__
- __shape__: `int`로 이루어진 튜플. 변수의 형태를 지정합니다.
- __dtype__: `str`. 변수의 자료형을 지정합니다.
- __name__: `str`. 변수의 이름을 지정합니다.

__반환값__ 
    
`0.0`으로 채워진 변수(케라스 메타데이터 포함).
`shape`가 심볼릭인 경우 변수를 반환하는 대신 형태가 동적으로 변하는 텐서를 반환힙니다. 

__예시__
```python
>>> from keras import backend as K
>>> kvar = K.zeros((3,4))
>>> K.eval(kvar)
array([[ 0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.]], dtype=float32)
```

__NumPy 구현__
```python
def zeros(shape, dtype=floatx(), name=None):
    return np.zeros(shape, dtype=dtype)
```


----
### ones

```python
keras.backend.ones(shape, dtype=None, name=None)
```

모든 값이 1인 변수 인스턴스를 생성합니다.

__인자__
- __shape__: `int`로 이루어진 튜플. 변수의 형태를 지정합니다.
- __dtype__: `str`. 변수의 자료형을 지정합니다.
- __name__: `str`. 변수의 이름을 지정합니다.

__반환값__ 
    
`1.0`으로 채워진 변수(케라스 메타데이터 포함).
`shape`가 심볼릭인 경우 변수를 반환하는 대신 형태가 동적으로 변하는 텐서를 반환힙니다. 

__예시__

```python
>>> from keras import backend as K
>>> kvar = K.ones((3,4))
>>> K.eval(kvar)
array([[ 1.,  1.,  1.,  1.],
       [ 1.,  1.,  1.,  1.],
       [ 1.,  1.,  1.,  1.]], dtype=float32)
```

__NumPy 구현__
```python
def ones(shape, dtype=floatx(), name=None):
    return np.ones(shape, dtype=dtype)
```


----
### eye

```python
keras.backend.eye(size, dtype=None, name=None)
```

인스턴스를 생성합니다. 

__인자__
- __size__: `int`. 행과 열의 수를 지정합니다. 
- __dtype__: `str`. 변수의 자료형을 지정합니다. 
- __name__: `str`. 변수의 이름을 지정합니다.

__반환값__ 
    
단위행렬 형태의 케라스 변수. 

__예시__

```python
>>> from keras import backend as K
>>> kvar = K.eye(3)
>>> K.eval(kvar)
array([[ 1.,  0.,  0.],
       [ 0.,  1.,  0.],
       [ 0.,  0.,  1.]], dtype=float32)
```

__NumPy 구현__

```python
def eye(size, dtype=None, name=None):
    return np.eye(size, dtype=dtype)
```


----
### zeros_like

```python
keras.backend.zeros_like(x, dtype=None, name=None)
```

지정한 별도의 텐서와 형태가 같으면서 모든 값이 0인 텐서를 생성합니다. 

__인자__
- __x__: 케라스 변수 또는 케라스 텐서. 
- __dtype__: `str`. 변수의 자료형을 지정합니다. `None`(기본값)인 경우 `x`의 자료형을 따릅니다.
- __name__: `str`. 변수의 이름을 지정합니다.


__반환값__ 
    
`x`와 형태가 같으며 값이 0인 케라스 변수.

__예시__

```python
>>> from keras import backend as K
>>> kvar = K.variable(np.random.random((2,3)))
>>> kvar_zeros = K.zeros_like(kvar)
>>> K.eval(kvar_zeros)
array([[ 0.,  0.,  0.],
       [ 0.,  0.,  0.]], dtype=float32)
```

__NumPy 구현__

```python
def zeros_like(x, dtype=floatx(), name=None):
    return np.zeros_like(x, dtype=dtype)
```


----

### ones_like

```python
keras.backend.ones_like(x, dtype=None, name=None)
```

지정한 별도의 텐서와 형태가 같으면서 모든 값이 1인 텐서를 생성합니다. 

__인자__
- __x__: 케라스 변수 또는 케라스 텐서. 
- __dtype__: `str`. 변수의 자료형을 지정합니다. `None`(기본값)인 경우 `x`의 자료형을 따릅니다.
- __name__: `str`. 변수의 이름을 지정합니다.


__반환값__ 
    
`x`와 형태가 같으며 값이 1인 케라스 변수.

__예시__

```python
>>> from keras import backend as K
>>> kvar = K.variable(np.random.random((2,3)))
>>> kvar_ones = K.ones_like(kvar)
>>> K.eval(kvar_ones)
array([[ 1.,  1.,  1.],
       [ 1.,  1.,  1.]], dtype=float32)
```

__NumPy 구현__

```python
def ones_like(x, dtype=floatx(), name=None):
    return np.ones_like(x, dtype=dtype)
```


----
### identity

```python
keras.backend.identity(x, name=None)
```

입력 텐서와 내용이 같은 텐서를 반환합니다.

__인자__
- __x__: 입력 텐서.
- __name__: `str`. 생성할 변수의 이름을 지정합니다.

__반환값__ 
    
`x`와 형태 및 값이 동일한 텐서.
    
    
----
### random_uniform_variable

```python
keras.backend.random_uniform_variable(shape, low, high, dtype=None, name=None, seed=None)
```

균등분포에서 무작위 값을 추출하여 변수 인스턴스를 생성합니다.

__인자__
- __shape__: `int`로 이루어진 튜플. 생성할 변수의 형태입니다.
- __low__: `float`. 출력 범위의 최솟값을 지정합니다.
- __high__: `float`. 출력 범위의 최댓값을 지정합니다. 
- __dtype__: `str`. 변수의 자료형을 지정합니다.
- __name__: `str`. 변수의 이름을 지정합니다.
- __seed__: `int`. 무작위 값 생성에 사용할 seed값을 지정합니다.

__반환값__ 
    
균등분포에서 추출한 표본값으로 이루어진 케라스 변수.  
  
__예시__

```python
# TensorFlow example
>>> kvar = K.random_uniform_variable((2,3), 0, 1)
>>> kvar
<tensorflow.python.ops.variables.Variable object at 0x10ab40b10>
>>> K.eval(kvar)
array([[ 0.10940075,  0.10047495,  0.476143  ],
       [ 0.66137183,  0.00869417,  0.89220798]], dtype=float32)
```

__NumPy 구현__

```python
def random_uniform_variable(shape, low, high, dtype=None, name=None, seed=None):
    return (high - low) * np.random.random(shape).astype(dtype) + low
```


----
### random_normal_variable


```python
keras.backend.random_normal_variable(shape, mean, scale, dtype=None, name=None, seed=None)
```

정규분포에서 무작위 값을 추출하여 변수 인스턴스를 생성합니다.

__인자__

- __shape__: `int`로 이루어진 튜플. 생성할 변수의 형태입니다.
- __mean__: `float`. 사용할 정규분포의 평균을 지정합니다.
- __scale__: `float`. 사용할 정규분포의 표준편차를 지정합니다.
- __dtype__: `str`. 변수의 자료형을 지정합니다.
- __name__: `str`. 변수의 이름을 지정합니다.
- __seed__: `int`. 무작위 값 생성에 사용할 seed값을 지정합니다.

__반환값__ 
    
정규분포에서 추출한 표본값으로 이루어진 케라스 변수. 

__예시__

```python
# TensorFlow example
>>> kvar = K.random_normal_variable((2,3), 0, 1)
>>> kvar
<tensorflow.python.ops.variables.Variable object at 0x10ab12dd0>
>>> K.eval(kvar)
array([[ 1.19591331,  0.68685907, -0.63814116],
       [ 0.92629528,  0.28055015,  1.70484698]], dtype=float32)
```

__NumPy 구현__

```python
def random_normal_variable(shape, mean, scale, dtype=None, name=None, seed=None):
    return scale * np.random.randn(*shape).astype(dtype) + mean
```


----
### count_params

```python
keras.backend.count_params(x)
```

케라스 변수 또는 텐서의 원소 개수를 반환합니다. 

__인자__

- __x__: 케라스 텐서 또는 변수.

__반환값__ 
    
`int`.`x`의 원소 개수(배열의 각 차원 크기를 모두 곱한 값).

__예시__

```python
>>> kvar = K.zeros((2,3))
>>> K.count_params(kvar)
6
>>> K.eval(kvar)
array([[ 0.,  0.,  0.],
       [ 0.,  0.,  0.]], dtype=float32)
```

__NumPy 구현__

```python
def count_params(x):
    return x.size
```


----
### cast

```python
keras.backend.cast(x, dtype)
```

텐서의 자료형을 변경합니다.

케라스 변수를 입력할 경우 자료형은 지정한대로 변환되지만 케라스 텐서로 바뀌어 반환됩니다. 

__인자__

- __x__: 케라스 텐서 또는 변수.
- __dtype__: `str`. `'float16'`, `'float32'`, 또는 `'float64'` 가운데 하나를 지정합니다.

__반환값__ 

지정한 `dtype` 자료형의 케라스 텐서.

__예시__

```python
>>> from keras import backend as K
>>> input = K.placeholder((2, 3), dtype='float32')
>>> input
<tf.Tensor 'Placeholder_2:0' shape=(2, 3) dtype=float32>
# It doesn't work in-place as below.
>>> K.cast(input, dtype='float16')
<tf.Tensor 'Cast_1:0' shape=(2, 3) dtype=float16>
>>> input
<tf.Tensor 'Placeholder_2:0' shape=(2, 3) dtype=float32>
# you need to assign it.
>>> input = K.cast(input, dtype='float16')
>>> input
<tf.Tensor 'Cast_2:0' shape=(2, 3) dtype=float16>
```
----

### update

```python
keras.backend.update(x, new_x)
```

`x`의 값을 `new_x`로 갱신합니다.

__인자__

- __x__: 한 개의 변수.
- __new_x__: x와 형태가 같은 텐서.  

__반환값__ 
    
값이 갱신된 변수 `x`.
    
    
----
### update_add

```python
keras.backend.update_add(x, increment)
```

`x`에 `increment`를 더해 값을 갱신합니다. 

__인자__

- __x__: 변수.
- __increment__: x와 같은 형태의 텐서.

__반환값__ 
    
값이 갱신된 변수 `x`.


----
### update_sub

```python
keras.backend.update_sub(x, decrement)
```

`x`에서 `decrement`를 빼 값을 갱신합니다.

__인자__

- __x__: 변수.
- __decrement__: x와 같은 형태의 텐서.

__반환값__ 
    
값이 갱신된 변수 `x`.
    
    
----
### moving_average_update

```python
keras.backend.moving_average_update(x, value, momentum)
```

변수의 이동평균을 계산합니다. 

__인자__

- __x__: 변수.
- __value__: `x`와 같은 형태의 텐서.
- __momentum__: 이동평균의 모멘텀.

__반환값__ 
    
변수를 업데이트하는 연산.
    
----
### dot

```python
keras.backend.dot(x, y)
```

두 개의 텐서(또는 변수)를 내적한 결과를 텐서로 반환합니다. 
차원의 개수가 서로 다른 텐서를 내적할 경우 Theano의 계산 방식을 따라 결과를 생성합니다. 
(예: `(2, 3) * (4, 3, 5) = (2, 4, 5)`, `(3, 10) * (2, 5, 10, 4) = (3, 2, 5, 4)`)

__인자__

- __x__: 텐서 또는 변수.
- __y__: 텐서 또는 변수.

__반환값__ 
    
`x` 과 `y`를 내적한 결과 텐서.

__예시__

```python
# 텐서 간의 내적
>>> x = K.placeholder(shape=(2, 3))
>>> y = K.placeholder(shape=(3, 4))
>>> xy = K.dot(x, y)
>>> xy
<tf.Tensor 'MatMul_9:0' shape=(2, 4) dtype=float32>
```

```python
# 텐서 간의 내적
>>> x = K.placeholder(shape=(32, 28, 3))
>>> y = K.placeholder(shape=(3, 4))
>>> xy = K.dot(x, y)
>>> xy
<tf.Tensor 'MatMul_9:0' shape=(32, 28, 4) dtype=float32>
```

```python
# Theano 방식의 연산 예
>>> x = K.random_uniform_variable(shape=(2, 3), low=0, high=1)
>>> y = K.ones((4, 3, 5))
>>> xy = K.dot(x, y)
>>> K.int_shape(xy)
(2, 4, 5)
```
__NumPy 구현__

```python
def dot(x, y):
    return np.dot(x, y)
```


----
### batch_dot

```python
keras.backend.batch_dot(x, y, axes=None)
```

배치별 내적을 수행합니다.

`x`와 `y`가 배치 데이터일 때 (예: (batch_size, :)) `batch_dot`을 사용하여 `x`와 `y`의 내적을 계산하면 각 배치별 내적의 결과를 얻을 수 있습니다. `batch_dot`은 입력 값보다 차원 수가 작은 텐서 또는 변수를 반환합니다. 만약 차원의 수가 1로 줄어드는 경우 `expand_dims`를 적용하여 반환할 차원 수가 2 이상이 되도록 합니다.

__인자__

- __x__: `ndim >= 2` 조건을 만족하는 케라스 텐서 또는 변수.
- __y__: `ndim >= 2` 조건을 만족하는 케라스 텐서 또는 변수. 
- __axes__: `int` 또는 `int`로 이루어진 튜플. `x`와 `y`의 순서대로 각각 감소 대상인 차원(내적 연산의 대상이 되는 차원)을 지정합니다. 단일한 정수를 입력하는 경우 `x`와 `y` 모두에 같은 차원이 지정됩니다.


__반환값__  

`x`의 형태에서 내적 대상을 제외한 나머지 차원과 `y`의 형태에서 배치 및 내적 대상을 제외한 나머지 차원을 이어붙인 형태의 텐서. 결과의 차원 수가 1인 경우(batch_size만 남을 경우) 결과를 (batch_size,1)로 바꾸어 반환합니다. 

__예시__

`x = [[1, 2], [3, 4]]`, `y = [[5, 6], [7, 8]]`일 때 `batch_dot(x, y, axes=1) = [[17], [53]]`으로 `x.dot(y.T)`의 결과를 반환합니다. `axes=1` 인자에 따라 `x`와 `y`의 1번째 차원끼리 곱하기 위해 `y`를 전치하기 때문입니다.  
  
__유사코드<sub>pseudo code</sub>:__
```
inner_products = []
for xi, yi in zip(x, y):
    inner_products.append(xi.dot(yi))
result = stack(inner_products)
```

__형태 변환에 대한 설명:__  

`x`의 모양은`(100, 20)`, `y`의 모양은`(100, 30, 20)`일 때 `axis=(1, 2)`인 경우 다음과 같은 규칙에 따라 출력 형태가 결정됩니다.

- `x.shape [0]`: 100 : 배치 크기의 차원으로 출력 형태의 첫 번째 차원 값이 됩니다.
- `x.shape [1]`: 20 : `axis`인자에서 지정한대로 내적 연산에서 합산될 차원입니다. 따라서 출력 형태에는 추가되지 않습니다. 
- `y.shape [0]`: 100 : 배치 크기의 차원입니다. 이미 `x`에서 가져온 차원이므로 출력 형태에 추가되지 않습니다. 이와 같이 항상 `y`의 첫 번째 차원은 배제됩니다.
- `y.shape [1]`: 30 : 출력 형태에 추가됩니다.
- `y.shape [2]`: 20 : `axis`인자에서 지정한대로 내적 연산에서 합산될 차원입니다. 따라서 출력 형태에는 추가되지 않습니다.  

결과적으로 `output_shape` =`(100, 30)`이 됩니다.

```python
>>> x_batch = K.ones(shape=(32, 20, 1))
>>> y_batch = K.ones(shape=(32, 30, 20))
>>> xy_batch_dot = K.batch_dot(x_batch, y_batch, axes=(1, 2))
>>> K.int_shape(xy_batch_dot)
(32, 1, 30)
```

__NumPy 구현__

<details>
<summary>NumPy 구현 보기</summary>

```python
def batch_dot(x, y, axes=None):
    if x.ndim < 2 or y.ndim < 2:
        raise ValueError('Batch dot requires inputs of rank 2 or more.')

    if isinstance(axes, int):
        axes = [axes, axes]
    elif isinstance(axes, tuple):
        axes = list(axes)

    if axes is None:
        if y.ndim == 2:
            axes = [x.ndim - 1, y.ndim - 1]
        else:
            axes = [x.ndim - 1, y.ndim - 2]

    if any([isinstance(a, (list, tuple)) for a in axes]):
        raise ValueError('Multiple target dimensions are not supported. ' +
                         'Expected: None, int, (int, int), ' +
                         'Provided: ' + str(axes))

    # Handle negative axes
    if axes[0] < 0:
        axes[0] += x.ndim
    if axes[1] < 0:
        axes[1] += y.ndim

    if 0 in axes:
        raise ValueError('Can not perform batch dot over axis 0.')

    if x.shape[0] != y.shape[0]:
        raise ValueError('Can not perform batch dot on inputs'
                         ' with different batch sizes.')

    d1 = x.shape[axes[0]]
    d2 = y.shape[axes[1]]
    if d1 != d2:
        raise ValueError('Can not do batch_dot on inputs with shapes ' +
                         str(x.shape) + ' and ' + str(y.shape) +
                         ' with axes=' + str(axes) + '. x.shape[%d] != '
                         'y.shape[%d] (%d != %d).' % (axes[0], axes[1], d1, d2))

    result = []
    axes = [axes[0] - 1, axes[1] - 1]  # ignore batch dimension
    for xi, yi in zip(x, y):
        result.append(np.tensordot(xi, yi, axes))
    result = np.array(result)

    if result.ndim == 1:
        result = np.expand_dims(result, -1)

    return result
```
</details>
----


### transpose

```python
keras.backend.transpose(x)
```

텐서를 전치한 결과를 반환합니다.

__인자__

- __x__: 텐서 또는 변수. 

__반환값__ 

`x`를 전치한 텐서.

__예시__

```python
>>> var = K.variable([[1, 2, 3], [4, 5, 6]])
>>> K.eval(var)
array([[ 1.,  2.,  3.],
       [ 4.,  5.,  6.]], dtype=float32)
>>> var_transposed = K.transpose(var)
>>> K.eval(var_transposed)
array([[ 1.,  4.],
       [ 2.,  5.],
       [ 3.,  6.]], dtype=float32)
```

```python
>>> inputs = K.placeholder((2, 3))
>>> inputs
<tf.Tensor 'Placeholder_11:0' shape=(2, 3) dtype=float32>
>>> input_transposed = K.transpose(inputs)
>>> input_transposed
<tf.Tensor 'transpose_4:0' shape=(3, 2) dtype=float32>

```

__NumPy 구현__

```python
def transpose(x):
    return np.transpose(x)
```


----
### gather

```python
keras.backend.gather(reference, indices)
```
`reference`로 지정한 텐서에서 `indices`로 지정한 인덱스의 원소들을 뽑아 하나의 텐서로 가져옵니다. 

__인자__

- __reference__: 대상을 추출할 텐서.
- __indices__: 추출 대상의 인덱스를 지정한 정수 텐서.

__반환값__  
  
`reference`와 자료형이 동일한 텐서.

__NumPy 구현__

```python
def gather(reference, indices):
    return reference[indices]
```


----
### max

```python
keras.backend.max(x, axis=None, keepdims=False)
```

텐서의 원소 가운데 최댓값을 반환합니다.

__인자__

- __x__: 텐서 또는 변수. 
- __axis__: 단일 `int` 값 또는 (*rank(x) = `x`의 차원 개수일 때) [-rank(x), rank(x)) 범위의 `int`로 이루어진 튜플. 최댓값을 찾을 방향의 축을 지정합니다. `None`인 경우 텐서 전체에서의 최댓값을 찾습니다.
- __keepdims__: `bool`. 출력값에서 `x`차원의 개수를 유지할지 여부를 지정합니다. `keepdims`를 `True`로 지정하면 입력값과 출력값 사이에 축소된 차원을 없애는 대신 크기 1을 부여해 차원을 남깁니다. `keepdims`가 `False`인 경우 축소된 차원을 없앰에 따라 출력 텐서의 차원 개수가 1만큼 감소합니다. 기본값은 `False`입니다.

__반환값__ 

`x`의 최댓값으로 이루어진 텐서. 

__NumPy 구현__

```python
def max(x, axis=None, keepdims=False):
    if isinstance(axis, list):
        axis = tuple(axis)
    return np.max(x, axis=axis, keepdims=keepdims)
```


----
### min

```python
keras.backend.min(x, axis=None, keepdims=False)
```

텐서의 원소 가운데 최댓값을 반환합니다.

__인자__

- __x__: 텐서 또는 변수. 
- __axis__: 단일 `int` 값 또는 (*rank(x) = `x`의 차원 개수일 때) [-rank(x), rank(x)) 범위의 `int`로 이루어진 튜플. 최솟값을 찾을 방향의 축을 지정합니다. `None`인 경우 텐서 전체에서의 최솟값을 찾습니다.
- __keepdims__: `bool`. 출력값에서 `x`차원의 개수를 유지할지 여부를 지정합니다. `keepdims`를 `True`로 지정하면 입력값과 출력값 사이에 축소된 차원을 없애는 대신 크기 1을 부여해 차원을 남깁니다. `keepdims`가 `False`인 경우 축소된 차원을 없앰에 따라 출력 텐서의 차원 개수가 1만큼 감소합니다. 기본값은 `False`입니다.

__반환값__ 
    
`x`의 최솟값으로 이루어진 텐서. 

__NumPy 구현__

```python
def min(x, axis=None, keepdims=False):
    if isinstance(axis, list):
        axis = tuple(axis)
    return np.min(x, axis=axis, keepdims=keepdims)
```
----


### sum

```python
keras.backend.sum(x, axis=None, keepdims=False)
```

지정된 축의 방향을 따라 텐서의 원소를 더해서 반환합니다.

__인자__

- __x__: 텐서 또는 변수. 
- __axis__: 단일 `int` 값 또는 (*rank(x) = `x`의 차원 개수일 때) [-rank(x), rank(x)) 범위의 `int`로 이루어진 튜플. 지정한 축을 따라 이동하면서 나머지 방향의 원소들끼리 더합니다. 예를 들어 형태가 (3, 4)인 텐서를 합산할 때 `axis=0`으로 지정할 경우 길이가 4인 텐서를 출력합니다. `None`인 경우 텐서의 모든 원소를 더합니다.
- __keepdims__: `bool`. 출력값에서 `x`차원의 개수를 유지할지 여부를 지정합니다. `keepdims`를 `True`로 지정하면 입력값과 출력값 사이에 축소된 차원을 없애는 대신 크기 1을 부여해 차원을 남깁니다. `keepdims`가 `False`인 경우 축소된 차원을 없앰에 따라 출력 텐서의 차원 개수가 1만큼 감소합니다. 예를 들어 형태가 (3, 4)인 텐서를 합산할 때 `axis=0, keepdims=True`로 지정할 경우 형태가 (1, 4)인 텐서를 출력합니다. 기본값은 `False`입니다. 

__반환값__ 
    
'x'의 합으로 이루어진 텐서. 

__NumPy 구현__

```python
def sum(x, axis=None, keepdims=False):
    if isinstance(axis, list):
        axis = tuple(axis)
    return np.sum(x, axis=axis, keepdims=keepdims)
```


----
### prod

```python
keras.backend.prod(x, axis=None, keepdims=False)
```

지정된 축의 방향을 따라 텐서의 원소를 곱해서 반환합니다.


__인자__

- __x__: 텐서 또는 변수. 
- __axis__: 단일 `int` 값 또는 (*rank(x) = `x`의 차원 개수일 때) [-rank(x), rank(x)) 범위의 `int`로 이루어진 튜플. 지정한 축을 따라 이동하면서 나머지 방향의 원소들끼리 곱합니다. 예를 들어 형태가 (3, 4)인 텐서를 곱할 때 `axis=0`으로 지정할 경우 길이가 4인 텐서를 출력합니다. `None`인 경우 텐서의 모든 원소를 곱합니다.
- __keepdims__: `bool`. 출력값에서 `x`차원의 개수를 유지할지 여부를 지정합니다. `keepdims`를 `True`로 지정하면 입력값과 출력값 사이에 축소된 차원을 없애는 대신 크기 1을 부여해 차원을 남깁니다. `keepdims`가 `False`인 경우 축소된 차원을 없앰에 따라 출력 텐서의 차원 개수가 1만큼 감소합니다. 예를 들어 형태가 (3, 4)인 텐서를 합산할 때 `axis=0, keepdims=True`로 지정할 경우 형태가 (1, 4)인 텐서를 출력합니다. 기본값은 `False`입니다.

__반환값__ 
    
'x'의 원소들의 곱으로 이루어진 텐서.

__NumPy 구현__

```python
def prod(x, axis=None, keepdims=False):
    if isinstance(axis, list):
        axis = tuple(axis)
    return np.prod(x, axis=axis, keepdims=keepdims)
```
----


### cumsum

```python
keras.backend.cumsum(x, axis=0)
```

지정된 축의 방향으로 텐서의 원소를 누적합하여 반환합니다.

__인자__

- __x__: 텐서 또는 변수.
- __axis__: `int`. 값을 합산할 방향을 나타내는 축.

__반환값__  
`x`의 원소를 누적한 합의 텐서.

__NumPy 구현__

```python
def cumsum(x, axis=0):
    return np.cumsum(x, axis=axis)
```
----


### cumprod

```python
keras.backend.cumprod(x, axis=0)
```


지정된 축의 방향으로 텐서의 원소를 누적곱하여 반환합니다.

__인자__

- __x__: 텐서 또는 변수.
- __axis__: `int`. 값을 곱할 방향을 나타내는 축.

__반환값__  
`x`의 원소를 누적한 합의 텐서.

__NumPy 구현__

```python
def cumprod(x, axis=0):
    return np.cumprod(x, axis=axis)
```
----


### var

```python
keras.backend.var(x, axis=None, keepdims=False)
```

텐서 가운데 지정한 축 방향에 있는 원소들끼리의 분산을 구합니다.

__인자__

- __x__: 텐서 또는 변수. 
- __axis__: 단일 `int` 값 또는 (*rank(x) = `x`의 차원 개수일 때) [-rank(x), rank(x)) 범위의 `int`로 이루어진 튜플. 분산을 구할 값들이 속한 방향을 지정합니다. 예를 들어 형태가 (3, 4)인 텐서의 분산을 구할 때 `axis=0`으로 지정할 경우 각 0번째 축 방향에 해당하는 3개의 원소끼리의 분산을 구하게 되므로 결과적으로 길이가 4인 텐서를 출력합니다. `None`인 경우 모든 원소 사이의 분산을 구합니다.
- __keepdims__: `bool`. 출력값에서 `x`차원의 개수를 유지할지 여부를 지정합니다. `keepdims`를 `True`로 지정하면 입력값과 출력값 사이에 축소된 차원을 없애는 대신 크기 1을 부여해 차원을 남깁니다. `keepdims`가 `False`인 경우 축소된 차원을 없앰에 따라 출력 텐서의 차원 개수가 1만큼 감소합니다. 예를 들어 형태가 (3, 4)인 텐서를의 분산을 구할 때 `axis=0, keepdims=True`로 지정할 경우 형태가 (1, 4)인 텐서를 출력합니다. 기본값은 `False`입니다.
 
__반환값__ 
    
`x`의 원소들의 분산으로 이루어진 텐서.

__NumPy 구현__

```python
def var(x, axis=None, keepdims=False):
    if isinstance(axis, list):
        axis = tuple(axis)
    return np.var(x, axis=axis, keepdims=keepdims)
```
----
### std

```python
keras.backend.std(x, axis=None, keepdims=False)
```

텐서 가운데 지정한 축 방향에 있는 원소들끼리의 표준편차를 구합니다.

__인자__

- __x__: 텐서 또는 변수. 
- __axis__: 단일 `int` 값 또는 (*rank(x) = `x`의 차원 개수일 때) [-rank(x), rank(x)) 범위의 `int`로 이루어진 튜플. 표준편차를 구할 값들이 속한 방향을 지정합니다. 예를 들어 형태가 (3, 4)인 텐서의 표준편차를 구할 때 `axis=0`으로 지정할 경우 각 0번째 축 방향에 해당하는 3개의 원소끼리의 표준편차를 구하게 되므로 결과적으로 길이가 4인 텐서를 출력합니다. `None`인 경우 모든 원소 사이의 표준편차를 구합니다.
- __keepdims__: `bool`. 출력값에서 `x`차원의 개수를 유지할지 여부를 지정합니다. `keepdims`를 `True`로 지정하면 입력값과 출력값 사이에 축소된 차원을 없애는 대신 크기 1을 부여해 차원을 남깁니다. `keepdims`가 `False`인 경우 축소된 차원을 없앰에 따라 출력 텐서의 차원 개수가 1만큼 감소합니다. 예를 들어 형태가 (3, 4)인 텐서의 표준편차를 구할 때 `axis=0, keepdims=True`로 지정할 경우 형태가 (1, 4)인 텐서를 출력합니다. 기본값은 `False`입니다.
 
__반환값__ 
    
`x`의 원소들의 표준편차로 이루어진 텐서.

__NumPy 구현__
```python
def std(x, axis=None, keepdims=False):
    if isinstance(axis, list):
        axis = tuple(axis)
    return np.std(x, axis=axis, keepdims=keepdims)
```


----
### mean

```python
keras.backend.mean(x, axis=None, keepdims=False)
```

텐서 가운데 지정한 축 방향에 있는 원소들끼리 평균을 구합니다.

__인자__

- __x__: 텐서 또는 변수. 
- __axis__: 단일 `int` 값 또는 (*rank(x) = `x`의 차원 개수일 때) [-rank(x), rank(x)) 범위의 `int`로 이루어진 튜플. 평균을 구할 값들이 속한 방향을 지정합니다. 예를 들어 형태가 (3, 4)인 텐서의 평균을 구할 때 `axis=0`으로 지정할 경우 각 0번째 축 방향에 해당하는 3개의 원소끼리의 평균을 구하게 되므로 결과적으로 길이가 4인 텐서를 출력합니다. `None`인 경우 모든 원소 사이의 평균을 구합니다.
- __keepdims__: `bool`. 출력값에서 `x`차원의 개수를 유지할지 여부를 지정합니다. `keepdims`를 `True`로 지정하면 입력값과 출력값 사이에 축소된 차원을 없애는 대신 크기 1을 부여해 차원을 남깁니다. `keepdims`가 `False`인 경우 축소된 차원을 없앰에 따라 출력 텐서의 차원 개수가 1만큼 감소합니다. 예를 들어 형태가 (3, 4)인 텐서의 평균을 구할 때 `axis=0, keepdims=True`로 지정할 경우 형태가 (1, 4)인 텐서를 출력합니다. 기본값은 `False`입니다.
 
__반환값__ 
    
`x`의 원소들의 평균으로 이루어진 텐서.

__NumPy 구현__

```python
def mean(x, axis=None, keepdims=False):
    if isinstance(axis, list):
        axis = tuple(axis)
    return np.mean(x, axis=axis, keepdims=keepdims)
```


----
### any

```python
keras.backend.any(x, axis=None, keepdims=False)
```

비트 단위로 논리적 OR 조건 충족여부를 계산합니다. 대상 원소들 가운데 하나라도 True 또는 0이 아닌 숫자값을 가지는 경우 True를 반환합니다.

__인자__

- __x__: 텐서, 변수, 또는 텐서와 변수에 대한 조건식. (예: `x` 또는 `x==1`)
- __axis__:  단일 `int` 값 또는 (*rank(x) = `x`의 차원 개수일 때) [-rank(x), rank(x)) 범위의 `int`로 이루어진 튜플. or 조건을 판단할 값들이 속한 방향을 지정합니다. 예를 들어 형태가 (3, 4)인 텐서에서 논리 연산 결과를 구할 때 `axis=0`으로 지정할 경우 각 0번째 축 방향에 해당하는 3개의 원소끼리 연산하게 되므로 결과적으로 길이가 4인 텐서를 출력합니다. `None`인 경우 텐서 전체를 대상으로 논리 연산을 수행합니다.
- __keepdims__: `bool`. 출력값에서 `x`차원의 개수를 유지할지 여부를 지정합니다. `keepdims`를 `True`로 지정하면 입력값과 출력값 사이에 축소된 차원을 없애는 대신 크기 1을 부여해 차원을 남깁니다. `keepdims`가 `False`인 경우 축소된 차원을 없앰에 따라 출력 텐서의 차원 개수가 1만큼 감소합니다. 예를 들어 형태가 (3, 4)인 텐서의 논리 연산을 할 때 `axis=0, keepdims=True`로 지정할 경우 형태가 (1, 4)인 텐서를 출력합니다. 기본값은 `False`입니다.

__반환값__ 

`bool`형식의 텐서(True 또는 False).

__NumPy 구현__

```python
def any(x, axis=None, keepdims=False):
    if isinstance(axis, list):
        axis = tuple(axis)
    return np.any(x, axis=axis, keepdims=keepdims)
```


----
### all

```python
keras.backend.all(x, axis=None, keepdims=False)
```

비트 단위로 논리적 AND 조건 충족여부를 계산합니다. 모든 대상 원소들이 True 또는 0이 아닌 숫자값을 가지는 경우 True를 반환합니다.

__인자__

- __x__: 텐서, 변수, 또는 텐서와 변수에 대한 조건식. (예: `x` 또는 `x==1`)
- __axis__:  단일 `int` 값 또는 (*rank(x) = `x`의 차원 개수일 때) [-rank(x), rank(x)) 범위의 `int`로 이루어진 튜플. or 조건을 판단할 값들이 속한 방향을 지정합니다. 예를 들어 형태가 (3, 4)인 텐서에서 논리 연산 결과를 구할 때 `axis=0`으로 지정할 경우 각 0번째 축 방향에 해당하는 3개의 원소끼리 연산하게 되므로 결과적으로 길이가 4인 텐서를 출력합니다. `None`인 경우 텐서 전체를 대상으로 논리 연산을 수행합니다.
- __keepdims__: `bool`. 출력값에서 `x`차원의 개수를 유지할지 여부를 지정합니다. `keepdims`를 `True`로 지정하면 입력값과 출력값 사이에 축소된 차원을 없애는 대신 크기 1을 부여해 차원을 남깁니다. `keepdims`가 `False`인 경우 축소된 차원을 없앰에 따라 출력 텐서의 차원 개수가 1만큼 감소합니다. 예를 들어 형태가 (3, 4)인 텐서의 논리 연산을 할 때 `axis=0, keepdims=True`로 지정할 경우 형태가 (1, 4)인 텐서를 출력합니다. 기본값은 `False`입니다.

__반환값__ 

`bool` 자료형의 텐서(True 또는 False).

__NumPy 구현__

```python
def all(x, axis=None, keepdims=False):
    if isinstance(axis, list):
        axis = tuple(axis)
    return np.all(x, axis=axis, keepdims=keepdims)
```


----
### argmax

```python
keras.backend.argmax(x, axis=-1)
```

지정한 축 방향의 원소 가운데 최댓값인 원소의 인덱스를 반환합니다.

__인자__

- __x__: 텐서 또는 변수. 
- __axis__: 최댓값 인덱스를 탐색할 방향의 축을 지정합니다. 예를 들어 형태가 (3, 4)인 텐서에서 최댓값을 탐색할 때 `axis=0`으로 지정할 경우 각 0번째 축 방향에 해당하는 3개의 원소끼리 연산하게 되므로 결과적으로 길이가 4인 텐서를 출력합니다. `None`인 경우 텐서 전체를 기준으로 한 인덱스 값을 반환합니다. 

__반환값__     

텐서.  

__NumPy 구현__

```python
def argmax(x, axis=-1):
    return np.argmax(x, axis=axis)
```


----
### argmin


```python
keras.backend.argmin(x, axis=-1)
```
지정한 축 방향의 원소 가운데 최솟값인 원소의 인덱스를 반환합니다.

__인자__

- __x__: 텐서 또는 변수. 
- __axis__: 최솟값 인덱스를 탐색할 방향의 축을 지정합니다. 예를 들어 형태가 (3, 4)인 텐서에서 최솟값을 탐색할 때 `axis=0`으로 지정할 경우 각 0번째 축 방향에 해당하는 3개의 원소끼리 연산하게 되므로 결과적으로 길이가 4인 텐서를 출력합니다. `None`인 경우 텐서 전체를 기준으로 한 인덱스 값을 반환합니다. 

__반환값__     

텐서.  

__NumPy 구현__

```python
def argmin(x, axis=-1):
    return np.argmin(x, axis=axis)
```


----
### square

```python
keras.backend.square(x)
```

텐서의 각 원소를 제곱합니다.

__인자__  

- __x__: 텐서 또는 변수. 

__반환값__   

텐서.
    
    
----
### abs

```python
keras.backend.abs(x)
```

텐서의 각 원소별 절대값을 구합니다.  

__인자__  

- __x__: 텐서 또는 변수. 

__반환값__   
    
텐서.
    
    
----
### sqrt

```python
keras.backend.sqrt(x)
```

텐서의 각 원소별 제곱근을 구합니다.

__인자__  

- __x__: 텐서 또는 변수.

__반환값__   
    
텐서.

__NumPy 구현__

```python
def sqrt(x):
    y = np.sqrt(x)
    y[np.isnan(y)] = 0.
    return y
```


----
### exp

```python
keras.backend.exp(x)
```

텐서의 각 원소를 지수로 하여 자연상수의 지수제곱을 구합니다.  

__인자__  

- __x__: 텐서 또는 변수.  

__반환값__   
    
텐서.
    
    
----
### log

```python
keras.backend.log(x)
```

텐서의 각 원소별 자연로그 값을 구합니다.  

__인자__  

- __x__: 텐서 또는 변수.  

__반환값__  

텐서.
    
    
----
### logsumexp

```python
keras.backend.logsumexp(x, axis=None, keepdims=False)
```
텐서의 지정 차원에 있는 원소들의 `log(sum(exp()))`값을 계산합니다.

이 함수는 직접 `log(sum(exp(x)))` 수식을 계산하는 것에 비해 수치적으로 안정된 함수입니다. `exp` 계산에 매우 큰 값을 입력받아서 발생하는 오버플로우나 `log` 계산에 매우 작은 값을 입력받아서 발생하는 언더플로우 문제를 예방하도록 만들어져 있습니다.  
  
__인자__

- __x__: 텐서 또는 변수. 
- __axis__: 단일 `int` 값 또는 (*rank(x) = `x`의 차원 개수일 때) [-rank(x), rank(x)) 범위의 `int`로 이루어진 튜플로서 `logsumexp`를 계산할 값들이 속한 방향을 지정합니다. `None`인 경우 모든 차원에 대해 `logsumexp`를 계산합니다.  
- __keepdims__: `bool`. 출력값에서 `x`차원의 개수를 유지할지 여부를 지정합니다. `keepdims`를 `True`로 지정하면 입력값과 출력값 사이에 축소된 차원을 없애는 대신 크기 1을 부여해 차원을 남깁니다. `keepdims`가 `False`인 경우 축소된 차원을 없앰에 따라 출력 텐서의 차원 개수가 1만큼 감소합니다. 예를 들어 형태가 (3, 4)인 텐서 연산을 할 때 `axis=0, keepdims=True`로 지정할 경우 형태가 (1, 4)인 텐서를 출력합니다. 기본값은 `False`입니다.  
  
__반환값__  
    
텐서.  

__NumPy 구현__

```python
def logsumexp(x, axis=None, keepdims=False):
    if isinstance(axis, list):
        axis = tuple(axis)
    return sp.misc.logsumexp(x, axis=axis, keepdims=keepdims)
```


----
### round

```python
keras.backend.round(x)
```

텐서의 모든 원소를 반올림합니다.
0.5를 반올림하는 경우 가장 가까운 짝수를 반환합니다. (예: `round(11.5) = 12`, `round(12.5) = 12`)

__인자__  

- __x__: 텐서 또는 변수.  

__반환값__  
    
텐서.  


----
### sign

```python
keras.backend.sign(x)
```

텐서 내 각 원소의 부호값을 구합니다. 음수의 경우 `-1`, `0`의 경우 `0`, 양수의 경우 `1`을 반환합니다. 복소수의 경우 `x`가 `0`이 아닐 때는 `x/|x|`를, `0`일 때는 `0`을 반환합니다.  
  
__인자__  
  
- __x__: 텐서 또는 변수.  
  
__반환값__  
    
텐서.  
    
        
----
### pow

```python
keras.backend.pow(x, a)
```

각 원소별 a 제곱을 구합니다.

__인자__  

- __x__: 텐서 또는 변수.
- __a__: `int` 
  
__반환값__  
  
텐서.

__NumPy 구현__

```python
def pow(x, a=1.):
    return np.power(x, a)
```


----
### clip

```python
keras.backend.clip(x, min_value, max_value)
```

입력한 텐서의 원소 가운데 지정한 최솟값 또는 최댓값을 벗어나는 값을 지정한 최솟값, 최댓값으로 바꿉니다. 최댓값, 최솟값을 텐서로 입력하는 경우 `x`와 자료형이 같아야 합니다.  
  
__인자__

- __x__: 텐서 또는 변수. 
- __min_value__: `float`, `int` 또는 텐서.
- __max_value__: `float`, `int` 또는 텐서.

__반환값__ 
    
텐서.

__NumPy 구현__

```python
def clip(x, min_value, max_value):
    return np.clip(x, min_value, max_value)
```


----
### equal


```python
keras.backend.equal(x, y)
```

두 텐서의 값이 같은지 여부를 원소별로 비교합니다.

__인자__  
  
- __x__: 텐서 또는 변수.
- __y__: 텐서 또는 변수.
  
__반환값__  
  
`bool`형식의 텐서.  

__NumPy 구현__  

```python
def equal(x, y):
    return x == y
```


----

### not_equal

```python
keras.backend.not_equal(x, y)
```

두 텐서의 값이 다른지 여부를 원소별로 비교합니다.

__인자__

- __x__: 텐서 또는 변수.
- __y__: 텐서 또는 변수.

__반환값__  
  
`bool`형식의 텐서.  
  
__NumPy 구현__

```python
def not_equal(x, y):
    return x != y
```


----
### greater

```python
keras.backend.greater(x, y)
```

(x > y)의 진리값을 원소별로 판단합니다.

__인자__

- __x__: 텐서 또는 변수.
- __y__: 텐서 또는 변수.

__반환값__ 
    
 `bool`형식의 텐서.

__NumPy 구현__

```python
def greater(x, y):
    return x > y
```


----
### greater_equal

```python
keras.backend.greater_equal(x, y)
```

(x >= y)의 진리값을 원소별로 판단합니다.

__인자__

- __x__: 텐서 또는 변수.
- __y__: 텐서 또는 변수.

__반환값__ 
    
`bool`형식의 텐서.

__NumPy 구현__

```python
def greater_equal(x, y):
    return x >= y
```


----
### less

```python
keras.backend.less(x, y)
```

(x < y)의 진리값을 원소별로 판단합니다.

__인자__

- __x__: 텐서 또는 변수.
- __y__: 텐서 또는 변수.

__반환값__ 
    
`bool`형식의 텐서.

__NumPy 구현__

```python
def less(x, y):
    return x < y
```


----
### less_equal

```python
keras.backend.less_equal(x, y)
```

(x <= y)의 진리값을 원소별로 판단합니다.

__인자__

- __x__: 텐서 또는 변수.
- __y__: 텐서 또는 변수.

__반환값__ 
    
`bool`형식의 텐서.

__NumPy 구현__

```python
def less_equal(x, y):
    return x <= y
```


----
### maximum

```python
keras.backend.maximum(x, y)
```

두 텐서 가운데 각 위치별 최댓값으로 이루어진 텐서를 반환합니다.

__인자__

- __x__: 텐서 또는 변수.
- __y__: 텐서 또는 변수.

__반환값__  
      
텐서.

__NumPy 구현__

```python
def maximum(x, y):
    return np.maximum(x, y)
```


----
### minimum

```python
keras.backend.minimum(x, y)
```

두 텐서 가운데 각 위치별 최솟값으로 이루어진 텐서를 반환합니다.

__인자__

- __x__: 텐서 또는 변수.
- __y__: 텐서 또는 변수.

__반환값__ 
    
텐서.

__NumPy 구현__

```python
def minimum(x, y):
    return np.minimum(x, y)
```


----
### sin

```python
keras.backend.sin(x)
```

x의 각 원소별 사인값을 구합니다.  
  
__인자__  
  
- __x__: 텐서 또는 변수.  
  
__반환값__  
    
텐서.


----
### cos

```python
keras.backend.cos(x)
```

x의 각 원소별 코사인값을 구합니다.  
  
__인자__  
  
- __x__: 텐서 또는 변수.  
  
__반환값__  
    
텐서.  


----
### normalize_batch_in_training

```python
keras.backend.normalize_batch_in_training(x, gamma, beta, reduction_axes, epsilon=0.001)
```

배치의 대한 평균과 표준을 계산한 다음 배치 정규화를 적용합니다. 

__인자__

- __x__: Input 텐서 또는 변수.
- __gamma__: 입력 스케일링에 사용되는 텐서.
- __beta__: 입력을 중앙에 위치시키는 텐서.
- __reduction_axes__: `int` 또는 `int`로 이루어진 리스트나 튜플. 정규화를 수행할 축을 지정합니다.
- __epsilon__: 0으로 나누는 오류를 방지하기 위해 더할 작은 상수값.  
  
__반환값__  
      
`(normalized_tensor, mean, variance)` 인자로 이루어진 길이 3의 튜플.
    
----


### batch_normalization


```python
keras.backend.batch_normalization(x, mean, var, beta, gamma, axis=-1, epsilon=0.001)
```

평균과 분산, 베타와 감마가 주어진 x에 대해 배치 정규화를 수행합니다.  
  
출력 예시: `output = (x - mean) / sqrt(var + epsilon) * gamma + beta`  
  
__인자__  

- __x__: 입력 텐서 또는 변수.
- __mean__: 배치의 평균 
- __var__: 배치의 분산
- __gamma__: 입력 스케일링에 사용되는 텐서.
- __beta__: 입력을 중앙에 위치시키는 텐서.
- __axis__: `int`, 정규화 시킬 축. 일반적으로 변인<sub>feature</sub> 차원을 지정합니다.
- __epsilon__: 0으로 나누는 오류를 방지하기 위해 더할 작은 상수값.

__반환값__ 
    
텐서.
    
__NumPy 구현__

```python
def batch_normalization(x, mean, var, beta, gamma, axis=-1, epsilon=0.001):
    return ((x - mean) / sqrt(var + epsilon)) * gamma + beta
```


----
### concatenate

```python
keras.backend.concatenate(tensors, axis=-1)
```
  
리스트로 묶인 텐서들을 지정한 축을 따라 연결합니다.  
  
__인자__  

- __tensors__: 연결할 텐서 리스트.
- __axis__: 연결할 축.

__반환값__ 
    
텐서.
 
 
----
### reshape

```python
keras.backend.reshape(x, shape)
```

텐서의 형태를 지정한 모양으로 바꿉니다. 

__인자__

- __x__: 텐서 또는 변수.
- __shape__: 바꿀 형태를 나타내는 튜플.


__반환값__ 

텐서.


----
### permute_dimensions

```python
keras.backend.permute_dimensions(x, pattern)
```

텐서의 축을 지정한 순서대로 재배치합니다.  
  
__인자__  
  
- __x__: 텐서 또는 변수.  
- __pattern__: 텐서의 각 차원 인덱스로 이루어진 튜플. (예: `(0, 2, 1)`)  

__반환값__ 
    
텐서.
    
    
----
### resize_images

```python
keras.backend.resize_images(x, height_factor, width_factor, data_format, interpolation='nearest')
```

4D 텐서에 들어있는 이미지들의 크기를 조정합니다. 4D텐서는 `batch, height, width, channels` 4개의 차원으로 이루어진 텐서입니다.

__인자__  

- __x__: 크기를 조정할 텐서 또는 변수.
- __height_factor__: 양의 정수.
- __width_factor__: 양의 정수.
- __data_format__: `str`. `"channels_last"` 또는 `"channels_first"`. 입력할 4D 텐서의 차원 순서가 `(batch, height, width, channels)`가 될 것인지 `(batch, channels, height, width)`가 될 것인지를 지정합니다.
- __interpolation__: `str`. `"nearest"` 또는 `"bilinear"` 중 하나. 크기 조정에 따라 추가될 값을 결정할 방식을 지정합니다.

__반환값__   
    
텐서.

__오류__

- __ValueError__: `data_format`이 `"channels_last"` 또는 `"channels_first"`가 아닌 경우에 발생합니다.


----
### resize_volumes

```python
keras.backend.resize_volumes(x, depth_factor, height_factor, width_factor, data_format)
```

5D 텐서에 포함된 입체의 크기를 조정합니다. 5D 텐서는 `batch, depth, height, width, channels` 5개의 차원으로 이루어진 텐서입니다.

__인자__

- __x__: 크기를 조정할 텐서 또는 변수.
- __depth_factor__: 양의 정수.
- __height_factor__: 양의 정수.
- __width_factor__: 양의 정수.
- __data_format__: `str`. `"channels_last"` 또는 `"channels_first"`. 입력할 5D 텐서의 차원 순서가 `(batch, depth, height, width, channels)`가 될 것인지 `(batch, channels, depth, height, width)`가 될 것인지를 지정합니다.

__반환값__     
  
텐서.  
  
__오류__

- __ValueError__: `data_format`이 `"channels_last"` 또는 `"channels_first"`가 아닌 경우에 발생합니다.
    
    
----
### repeat_elements

```python
keras.backend.repeat_elements(x, rep, axis)
```
지정한 축을 따라 텐서의 원소를 지정한 횟수만큼 반복합니다. `np.repeat`와 같은 방식으로 작동합니다. `x`의 형식이 `(s1, s2, s3)` 이고, `axis`가 `1`이면, 출력형식은 `(s1, s2 * rep, s3)`이 됩니다. 예를 들어 `[a, b]`로 이루어진 텐서를 `rep=2`, `axis=1`로 지정하여 입력하면 결과는 `[a, a, b, b]`의 텐서가 됩니다.  
  
__인자__  

- __x__: 텐서 또는 변수.
- __rep__: `int`. 각 원소를 반복할 횟수.
- __axis__: 반복할 축

__반환값__  
    
텐서.


----
### repeat

```python
keras.backend.repeat(x, n)
```

2D 텐서를 반복합니다. 만약 `x`가 `(samples, dim)`형태이고 `n`이 `2`라면, 출력값의 형태는 `(samples, 2, dim)`이 됩니다.


__인자__  

- __x__: 텐서 또는 변수.
- __n__: `int`. 반복횟수.

__반환값__  
    
텐서.
    
    
----
### arange

```python
keras.backend.arange(start, stop=None, step=1, dtype='int32')
```

연속된 정수로 이루어진 1D 텐서를 생성합니다. 이 함수로 입력되는 인자는 Python의 range 함수와 같은 방식으로 처리됩니다. 예를 들어 `arange(5)`와 같이 하나의 인자만 입력하는 경우 `start=0`, `stop=5`로 해석되어 `[0, 1, 2, 3, 4]`로 이루어진 `(5, )`형태의 텐서를 생성합니다. 반환되는 텐서의 기본 자료형은 `'int32'`로 TensorFlow의 기본값과 일치합니다.

__인자__  

- __start__: 시작 값.
- __stop__: 정지 값.
- __step__: 이어지는 두 값 사이의 간격.
- __dtype__: `int`. 반환할 텐서의 자료형.

__반환값__  
    
`int`형 텐서.


----
### tile

```python
keras.backend.tile(x, n)
```

텐서 또는 NumPy 배열 `x`를 `n`으로 지정한 형태만큼 나열한 텐서를 생성합니다.

__인자__  

- __x__: 텐서 또는 배열.
- __n__: `int` 또는 `int`로 이루어진 리스트나 튜플. `x`를 각 축의 방향으로 몇 번 나열할 것인지를 지정합니다. 따라서 `n`의 차원 개수는 `x`의 차원 개수와 같아야 합니다. 예를 들어 `(2, )`형태의 `[a, a]`배열을 3회 나열할 경우 `n=3`이면 되지만, `(1, 2)`형태의 `[a, a]`를 1번째 축 방향으로 3회 나열할 경우 `n=(1, 3)`을 입력해야 합니다.

__반환값__  
    
텐서.
    
    
----
### flatten

```python
keras.backend.flatten(x)
```

텐서의 원소를 단일 차원으로 나열합니다. 입력값의 전체 원소 개수가 `n`일 경우 출력값은 `(n, )`의 형태를 갖게 됩니다.  

__인자__  

- __x__: 텐서 또는 변수.

__반환값__  

1D로 재배열된 텐서.


----
### batch_flatten

```python
keras.backend.batch_flatten(x)
```

각 배치별 텐서의 원소를 단일 차원으로 나열합니다. 배치 차원은 유지되기 때문에 출력값은 `(batch_size, n)`의 2차원 형태를 갖게 됩니다.

__인자__  

- __x__: 텐서 또는 변수.

__반환값__   
    
2D로 재배열된 텐서.


----
### expand_dims


```python
keras.backend.expand_dims(x, axis=-1)
```

입력한 배열에 1개의 차원을 추가합니다. 예를 들어 `(m, n)` 형태의 텐서를 입력하는 경우 출력값은 `(m, n, 1)`의 형태가 됩니다.  

__인자__  

- __x__: 텐서 또는 변수.  
- __axis__: 새로운 축을 추가할 위치.  

__반환값__  

텐서.
    
    
----
### squeeze

```python
keras.backend.squeeze(x, axis)
```

차원의 크기가 1인 축을 제거합니다. 예를 들어 형태가 `(3, 1, 4)`인 텐서의 경우 `axis=1` 인자를 통해 1크기를 갖는 1번째 차원을 제거할 수 있습니다.   

__인자__  

- __x__: 텐서 또는 변수.  
- __axis__: 제거할 축.  

__반환값__ 
    
`x`에서 지정한 차원을 줄인 것과 동일한 텐서.


----
### temporal_padding

```python
keras.backend.temporal_padding(x, padding=(1, 1))
```

3D 텐서의 중간차원에 패딩을 추가합니다.

__인자__  

- __x__: 텐서 또는 변수.  
- __padding__: 두 개의 `int`로 이루어진 튜플. 차원 1의 시작과 끝에 얼마나 많은 0을 추가할 지에 대한 수치.  

__반환값__  
    
0으로 패딩된 3D 텐서.  
    
    
----
### spatial_2d_padding

```python
keras.backend.spatial_2d_padding(x, padding=((1, 1), (1, 1)), data_format=None)
```

4D 텐서의 2번째 및 3번째 차원(= 1번 및 2번 축)에 패딩을 추가합니다.

__인자__  

- __x__: 텐서 또는 변수. 
- __padding__: 길이가 2인 튜플 두 개로 이루어진 튜플. 패딩 형태를 지정합니다. (예: `((1, 1), (1, 1))`) 하위 튜플의 각 값은 해당 차원의 앞과 뒤에 입력할 패딩의 개수입다.
- __data_format__: `str`. `"channels_last"` 또는 `"channels_first"` 가운데 하나를 지정합니다. `"channels_last"`의 경우 4개의 차원 중 가운데 두 개의 차원에, `"channels_first"`의 경우 마지막 두 개의 차원에 패딩을 추가합니다. `"None"`인 경우 `keras.backend.image_data_format()`에 지정된 기본값을 따릅니다.  

__반환값__  
0으로 패딩된 4D 텐서.

__오류__

- __ValueError__: `data_format`이 `"channels_last"` 또는 `"channels_first"`가 아닌 경우에 발생합니다. 


----
### spatial_3d_padding

```python
keras.backend.spatial_3d_padding(x, padding=((1, 1), (1, 1), (1, 1)), data_format=None)
```

5D 텐서의 깊이, 높이, 너비 차원에 0으로 패딩을 추가합니다. `padding`인자를 구성하는 세 개의 튜플 각각이 순서대로 깊이, 높이, 너비 차원에 대응됩니다. `data_format`이 `"channels_last"`인 경우 총 다섯 차원 중 가운데 세 개의 차원에, `"channels_first"`인 경우 마지막 세 개의 차원에 패딩을 추가합니다.  

__인자__  

- __x__: 텐서 또는 변수.  
- __padding__: 길이가 2인 튜플 세 개로 이루어진 이루어진 튜플. 패딩 형태를 지정합니다. (예: `((1, 1), (1, 1), (1, 1))`) 하위 튜플의 각 값은 해당 차원의 앞과 뒤에 입력할 패딩의 개수입니다. 
- __data_format__: `str`. `"channels_last"` 또는 `"channels_first"` 가운데 하나를 지정합니다. `"None"`인 경우 `keras.backend.image_data_format()`에 지정된 기본값을 따릅니다.

__반환값__ 
    
0으로 패딩된 5D 텐서.

__오류__

- __ValueError__: `data_format`이 `"channels_last"` 또는 `"channels_first"`가 아닌 경우에 발생합니다.


----
### stack


```python
keras.backend.stack(x, axis=0)
```

형태가 같은 텐서의 리스트를 지정한 축의 방향으로 쌓아 텐서를 만듭니다. 이때 지정할 수 있는 축은 최대 입력 텐서의 랭크+1 까지입니다. 예를 들어 형태가 `(m, n)`인 텐서로 이루어진 리스트를 입력할 경우 `axis`는 `0`부터 `2`까지 지정 가능합니다. 

__인자__

- __x__: 형태가 같은 텐서들로 이루어진 리스트.
- __axis__: 텐서를 쌓을 축의 방향.

__반환값__ 
    
텐서.

__NumPy 구현__

```python
def stack(x, axis=0):
    return np.stack(x, axis=axis)
```


----
### one_hot

```python
keras.backend.one_hot(indices, num_classes)
```

정수로 이루어진 텐서를 입력받아 각 클래스마다 원-핫 인코딩된 텐서를 생성합니다.

__인자__

- __indices__: `(batch_size, dim1, dim2, ... dim(n-1))`형식의 n차원 정수형 텐서.
- __num_classes__: `int`. 클래스의 개수.

__반환값__ 

인덱스 순서에서 각 클래스에 해당되는 값이 1이고 나머지가 0인 원-핫 인코딩 텐서. 입력 텐서가 `n`차원일 때 출력 텐서는 `n+1` 개의 차원을 가지며 `(batch_size, dim1, dim2, ... dim(n-1), num_classes)`의 형태로 이루어집니다.
    
    
----
### reverse

```python
keras.backend.reverse(x, axes)
```

지정된 축을 따라 텐서를 반전시킵니다.  

__인자__  

- __x__: 값을 반전시킬 텐서.  
- __axes__: `int`. 반전시킬 방향의 축.  

__반환값__  
    
텐서.

__NumPy 구현__

```python
def reverse(x, axes):
    if isinstance(axes, list):
        axes = tuple(axes)
    return np.flip(x, axes)
```


----
### slice

```python
keras.backend.slice(x, start, size)
```

텐서의 일부를 추출합니다.

__인자__  
  
- __x__: 입력 텐서.
- __start__: `int` 또는 `int`로 이루어진 리스트나 튜플, 텐서. 입력한 텐서의 각 축에서 추출을 시작할 위치 인덱스를 지정합니다.   
- __size__: `int` 또는 `int`로 이루어진 리스트나 튜플, 텐서. 각 축을 따라 추출할 배열의 길이를 나타냅니다.  
  
__반환값__  
    
`x`로부터 추출한 텐서. 형태는 아래와 같습니다.  

```python
new_x = x[start[0]: start[0] + size[0], ..., start[-1]: start[-1] + size[-1]]
```

__NumPy 구현__  

```python
def slice(x, start, size):
    slices = [py_slice(i, i + j) for i, j in zip(start, size)]
    return x[tuple(slices)]
```


----
### get_value

```python
keras.backend.get_value(x)
```

지정한 변수의 값을 NumPy 배열의 형태로 가져옵니다.   

__인자__  

- __x__: 입력 변수.  

__반환값__   
    
NumPy 배열.


----
### batch_get_value

```python
keras.backend.batch_get_value(ops)
```
텐서들의 리스트를 입력받아 그 값을 NumPy 배열의 리스트로 반환합니다.

__인자__ 

- __ops__: 적용할 텐서의 리스트.

__반환값__  

NumPy 배열의 리스트.


----
### set_value

```python
keras.backend.set_value(x, value)
```

변수의 값을 NumPy 배열로부터 가져옵니다. 

__인자__  

- __x__: 새 값을 부여할 텐서.
- __value__: 텐서에 값을 입력할 NumPy 배열.


----
### batch_set_value

```python
keras.backend.batch_set_value(tuples)
```

여러 텐서 변수들의 값을 한꺼번에 NumPy 배열로부터 가져옵니다. 각 변수와 그에 대응하는 NumPy 배열을 하나의 튜플로 지정하여 입력합니다.

__인자__  

- __tuples__: 원소가 `(tensor, value)`인 튜플로 이루어진 리스트. `value`인자는 NumPy 배열이어야 합니다.  


----
### print_tensor

```python
keras.backend.print_tensor(x, message='')
```

입력한 `message`와 함께 텐서의 값을 출력합니다. 이때 `print_tensor`는 `message`에 지정한 내용을 화면에 출력되는 print 연산 이외에도 별도로 `x`와 값이 동일한 새로운 텐서를 반환합니다. 별도로 저장하거나 연산에 입력하지 않는 경우 이 텐서는 사라지게 됩니다.  

__예시__  

```python
>>> x = K.print_tensor(x, message="x is: ")
```

__인자__  

- __x__: 출력할 텐서.
- __message__: 텐서와 함께 출력 할 메시지.  

__반환값__  
    
`x`과 값이 동일한 텐서.


----
### function


```python
keras.backend.function(inputs, outputs, updates=None)
```

케라스 함수 인스턴스를 생성합니다.

__인자__

- __inputs__: 플레이스홀더 텐서의 리스트.
- __outputs__: 출력 텐서의 리스트. 
- __updates__: 업데이트 연산의 리스트.
- __**kwargs__: `tf.Session.run`에 전달되는 값.  

__반환값__  
    
NumPy 배열 형태의 출력값.

__오류__

- __ValueError__: 유효하지 않은 kwargs 가 전달된 경우 또는 eager 모드로 실행한 경우에 발생합니다.
    
    
----
### gradients

```python
keras.backend.gradients(loss, variables)
```
각 변수에 대한 손실<sub>loss</sub>의 그래디언트를 반환합니다.

__인자__

- __loss__: 시키고자 하는 스칼라 값의 텐서.
- __variables__: 변수들의 리스트.

__반환값__    

그래디언트 텐서.
    
    
----
### stop_gradient

```python
keras.backend.stop_gradient(variables)
```

입력한 변수로부터 도출되는 모든 그래디언트를 0이 되게끔 만듭니다. 다시 말해 해당 변수가 그래디언트 계산 과정에서 상수처럼 취급되게끔 합니다. 예를 들어 `b = keras.backend.stop_gradient(a)`라고 지정할 경우 `b`로부터 이어지는 이후의 연산 단계는 정상적인 그래디언트가 계산되고 역전파 과정에서 가중치가 업데이트 되지만, `a`의 그래디언트는 0으로 고정되어 `a` 및 `a`를 도출해낸 이전 단계들은 역전파 과정에서 가중치가 동결됩니다.  

__인자__  

- __variables__: 연산 과정에서 다른 변수들에 대해 상수로 취급할 텐서 또는 텐서들의 리스트.

__반환값__  

다른 변수들에 의해 그래디언트 값이 바뀌지 않는 텐서 또는 텐서들의 리스트.  


----
### rnn

```python
keras.backend.rnn(step_function, inputs, initial_states, go_backwards=False, mask=None, constants=None, unroll=False, input_length=None)
```
텐서의 시간 차원을 따라 연산을 반복합니다.

__인자__  

- __step_function__: RNN에서 각 시간 단계마다 반복 계산할 함수.  
    - 매개변수:  
        - inputs: 시간 차원 없이 `(samples, ...)`의 형태를 가진 텐서로 특정한 시간 단계의 배치 샘플.   
        - states: 텐서의 리스트.  
    - 반환값:  
        - outputs: 시간 차원 없이 `(samples, ...)`의 형태를 가진 텐서.  
        - new_states: 'states'와 형태와 길이가 같은 텐서의 리스트. 리스트 내의 첫번째 텐서는 반드시 이전 단계의 출력 텐서여야 합니다.  
- __inputs__: `(samples, time, ...)`의 형태로 이루어진, 최소 3D 이상의 차원을 갖는 시계열 데이터의 텐서.
- __initial_states__: 시간 차원 없이 `(samples, ...)`의 형태를 가진 텐서. `step_function`에 입력할 `states`의 초기값.
- __go_backwards__: `bool`. `True`인 경우 시간 차원 순서의 역순으로 연산을 반복하여 역순으로 된 결과값을 반환합니다. 
- __mask__: `(samples, time)`의 형식을 가진 이진 텐서. 마스킹될 모든 원소마다 0값을 가집니다.
- __constants__:  각 시간 단계마다 전달할 상수 값의 리스트. 
- __unroll__: 반복 연산시 순환구조를 펼쳐서 연산 일부를 동시에 처리할 것인지, 혹은 백엔드의 심볼릭 루프를 (`while_loop` 또는 `scan`) 사용할 것인지를 지정합니다.
- __input_length__: 입력값의 시간 단계를 나타내는 고정된 숫자값.


__반환값__  
    
`(last_output, outputs, new_states)`로 구성된 튜플.  

- last_output: `rnn`의 마지막 단계 출력값으로 `(samples, ...)`의 형태를 갖고 있습니다.  
- outputs: `(samples, time, ...)`형태를 가진 텐서. 각 `outputs[s, t]`값은 `s`샘플로부터 `t`시간 단계에서 생성된 `step_function`의 출력값입니다.  
- new_states: `(samples, ...)`형태를 가진 텐서의 리스트. `step_function`이 가장 마지막 단계에서 반환한 `states`값입니다.

__오류__

- __ValueError__: 입력 차원이 3보다 작은 경우 발생합니다.
- __ValueError__: 입력의 시간 단계 길이가 유동적인데 `unroll`을 `True`로 지정한 경우 발생합니다.
- __ValueError__: `mask`를 설정했는데 (`None`이 아님) `states`가 없는 경우(`len(states)` == 0) 발생합니다.

__NumPy 구현__

<details>
<summary>Show the Numpy implementation</summary>

```python
def rnn(step_function, inputs, initial_states,
        go_backwards=False, mask=None, constants=None,
        unroll=False, input_length=None):

    if constants is None:
        constants = []

    output_sample, _ = step_function(inputs[:, 0], initial_states + constants)
    if mask is not None:
        if mask.dtype != np.bool:
            mask = mask.astype(np.bool)
        if mask.shape != inputs.shape[:2]:
            raise ValueError(
                'mask should have `shape=(samples, time)`, '
                'got {}'.format(mask.shape))

        def expand_mask(mask_, x):
            # `mask[:, t].ndim == x.ndim`이 되도록 mask를 확장합니다.
            while mask_.ndim < x.ndim + 1:
                mask_ = np.expand_dims(mask_, axis=-1)
            return mask_
        output_mask = expand_mask(mask, output_sample)
        states_masks = [expand_mask(mask, state) for state in initial_states]

    if input_length is None:
        input_length = inputs.shape[1]
    assert input_length == inputs.shape[1]
    time_index = range(input_length)
    if go_backwards:
        time_index = time_index[::-1]

    outputs = []
    states_tm1 = initial_states  # 여기서 tm1은 "이전 시간 단계"를 나타내는 "t minus one"을 뜻합니다.
    output_tm1 = np.zeros(output_sample.shape)
    for t in time_index:
        output_t, states_t = step_function(inputs[:, t], states_tm1 + constants)
        if mask is not None:
            output_t = np.where(output_mask[:, t], output_t, output_tm1)
            states_t = [np.where(state_mask[:, t], state_t, state_tm1)
                        for state_mask, state_t, state_tm1
                        in zip(states_masks, states_t, states_tm1)]
        outputs.append(output_t)
        states_tm1 = states_t
        output_tm1 = output_t

    return outputs[-1], np.stack(outputs, axis=1), states_tm1
```

</details>


----
### switch

```python
keras.backend.switch(condition, then_expression, else_expression)
```

조건에 따라 두 연산 가운데 하나의 결과를 반환합니다. 참인 경우 실행되는 `then_expression`과 거짓인 경우 실행되는 `else_expression`의 결과값은 형태가 같은 심볼릭 텐서여야 합니다. 

__인자__

- __condition__: `int (0/1)` 또는 `bool (Ture/False)` 형식의 텐서 또는 이를 반환하는 조건식.
- __then_expression__: 텐서 또는 텐서를 반환하는 호출가능한 연산. `condition`이 `1` 또는 `True`일 때 반환될 값입니다.
- __else_expression__: 텐서 또는 텐서를 반환하는 호출가능한 연산. `condition`이 `0` 또는 `False`일 때 반환될 값입니다.

__반환값__ 

조건에 따라 선택된 텐서.

__오류__

- __ValueError__: `expressions`보다 `condition`의 차원이 클 경우 오류가 발생합니다.

__NumPy 구현__

```python
def switch(condition, then_expression, else_expression):
    cond_float = condition.astype(floatx())
    while cond_float.ndim < then_expression.ndim:
        cond_float = cond_float[..., np.newaxis]
    return cond_float * then_expression + (1 - cond_float) * else_expression
```


----
### in_train_phase

```python
keras.backend.in_train_phase(x, alt, training=None)
```

훈련 단계일 경우 `x`를 선택하고 그렇지 않으면 `alt`를 선택합니다. `alt`는 `x`와 동일한 형태를 가져야합니다.  

__인자__  

- __x__: 훈련 단계일 경우 반환할 대상 (텐서 또는 텐서를 반환하는 호출가능한 연산).  
- __alt__: 그 밖의 경우에 반환할 대상 (텐서 또는 텐서를 반환하는 호출가능한 연산).  
- __training__: (필요한 경우) 훈련 단계 여부를 나타내기 위해 사용할 스칼라 텐서 또는 파이썬 `bool` 혹은 `int`값.  


__반환값__  

`training`플래그에 따라서 `x` 또는 `alt`를 반환. `training` 플래그 판단에는 기본값으로 `K.learning_phase()`를 참조하게 되어 있습니다.  


----
### in_test_phase

```python
keras.backend.in_test_phase(x, alt, training=None)
```

시험 단계일 경우 `x`를 선택하고 그렇지 않으면 `alt`를 선택합니다. `alt`는 `x`와 동일한 형태를 가져야합니다.  

__인자__  

- __x__: 시험 단계일 경우 반환할 대상 (텐서 또는 텐서를 반환하는 호출가능한 연산).  
- __alt__: 그 밖의 경우에 반환할 대상 (텐서 또는 텐서를 반환하는 호출가능한 연산).  
- __training__: (필요한 경우) 시험 단계 여부를 나타내기 위해 사용할 스칼라 텐서 또는 파이썬 `bool`이나 `int`값.  


__반환값__  

`training`플래그에 따라서 `x` 또는 `alt`를 반환. `training` 플래그 판단에는 기본값으로 `K.learning_phase()`를 참조하게 되어 있습니다.  

    
----
### relu

```python
keras.backend.relu(x, alpha=0.0, max_value=None, threshold=0.0)
```

ReLU(Rectified Linear Unit) 연산입니다. 기본값을 기준으로, 입력값의 원소별로 `max(x, 0)`의 결과를 반환합니다. 만약 `max_value` 또는 `threshold`를 지정하는 경우 `x >= max_value`인 경우 `f(x) = max_value`를, `threshold <= x < max_value`인 `x`에 대해서는 `f(x) = x`를, 그리고 나머지에 대해서는 `f(x) = alpha * (x - threshold)`를 반환합니다.

__인자__  

- __x__: 텐서 또는 변수.  
- __alpha__: 입력값 가운데 음수 영역에 곱해질 스칼라 값으로 기본값은 `0.`입니다.
- __max_value__: `float`. 반환할 값의 최댓값을 지정합니다.
- __threshold__: `float`. 반환할 값의 구간을 나누는 문턱값입니다.  

__반환값__ 

텐서.

__NumPy 구현__

```python
def relu(x, alpha=0., max_value=None, threshold=0.):
    if max_value is None:
        max_value = np.inf
    above_threshold = x * (x >= threshold)
    above_threshold = np.clip(above_threshold, 0.0, max_value)
    below_threshold = alpha * (x - threshold) * (x < threshold)
    return below_threshold + above_threshold
```


----
### elu

```python
keras.backend.elu(x, alpha=1.0)
```
ELU(Exponential Linear Unit) 연산입니다.  

__인자__  

- __x__: 활성화 함수를 계산할 텐서 또는 변수입니다. 
- __alpha__: 입력값 가운데 음수 영역에 곱해질 스칼라 값으로 기본값은 `1.0`입니다.

__반환값__  

텐서.

__NumPy 구현__

```python
def elu(x, alpha=1.):
    return x * (x > 0) + alpha * (np.exp(x) - 1.) * (x < 0)
```


----
### softmax

```python
keras.backend.softmax(x, axis=-1)
```

텐서의 소프트맥스<sub>softmax</sub> 연산입니다.  

__인자__  

- __x__: 텐서 또는 변수. 
- __axis__: softmax 연산을 적용할 차원을 지정합니다. 기본값은 텐서의 마지막 차원을 나타내는 `-1`입니다.  

__반환값__   

텐서.  

__NumPy 구현__  

```python
def softmax(x, axis=-1):
    y = np.exp(x - np.max(x, axis, keepdims=True))
    return y / np.sum(y, axis, keepdims=True)
```


----
### softplus

```python
keras.backend.softplus(x)
```

텐서의 소프트플러스<sub>softplus</sub> 연산입니다.  

__인자__  

- __x__: 텐서 또는 변수.  

__반환값__  

텐서.   

__NumPy 구현__

```python
def softplus(x):
    return np.log(1. + np.exp(x))
```


----
### softsign

```python
keras.backend.softsign(x)
```

텐서의 소프트사인<sub>softsign</sub> 연산입니다.

__인자__  

- __x__: 텐서 또는 변수.  

__반환값__  

텐서.  

__NumPy 구현__  

```python
def softsign(x):
    return x / (1 + np.abs(x))
```


----
### categorical_crossentropy

```python
keras.backend.categorical_crossentropy(target, output, from_logits=False, axis=-1)
```

출력 텐서와 목표 텐서 사이의 오차를 구하는 범주형<sub>categorical</sub> 크로스엔트로피 연산입니다.  

__인자__  

- __target__: `output`과 같은 형태의 텐서. 오차 계산의 기준입니다.  
- __output__: softmax의 결과 텐서(`from_logits`인자가 `True`인 경우는 제외. 이 경우 `output`은 로짓<sub>logit</sub> 함수의 출력값이어야 합니다). 오차를 줄일 대상입니다.  
- __from_logits__: `bool`. `output`이 softmax로부터 도출된 것인지 로짓으로부터 도출된 것인지를 지정합니다. 
- __axis__: 채널 축을 지정합니다. `axis=-1`은 `channels_last`형식 데이터에 해당됩니다. 반대로 `axis=1`은 `channels_first`형식 데이터에 해당됩니다.  

__반환값__   

출력 텐서.  

__오류__  

- __ValueError__: `axis`의 값이 `-1` 또는 `output`의 축 가운데 하나가 아닐 경우 오류가 발생합니다.

    
----
### sparse_categorical_crossentropy

```python
keras.backend.sparse_categorical_crossentropy(target, output, from_logits=False, axis=-1)
```

목표 값이 정수인 범주형 크로스엔트로피 연산입니다.

__인자__

- __target__: `int` 형식의 텐서. 오차 계산의 기준입니다.
- __output__: softmax의 결과 텐서(`from_logits`인자가 `True`인 경우는 제외. 이 경우 `output`은 로짓 함수의 출력값이어야 합니다). 오차를 줄일 대상입니다. 
- __from_logits__: `bool`. `output`이 소프트맥스로부터 도출된 것인지 로짓으로부터 도출된 것인지를 지정합니다.  
- __axis__: 채널 축을 지정합니다. `axis=-1`은 `channels_last`형식 데이터에 해당됩니다. 반대로 `axis=1`은 `channels_first`형식 데이터에 해당됩니다.  

__반환값__  
 
출력 텐서.

__오류__

- __ValueError__: `axis`의 값이 `-1` 또는 `output`의 축 가운데 하나가 아닐 경우 오류가 발생합니다.

    
----
### binary_crossentropy

```python
keras.backend.binary_crossentropy(target, output, from_logits=False)
```

출력 텐서와 목표 텐서 사이의 오차를 구하는 이진<sub>binary</sub> 크로스엔트로피 연산입니다.  

__인자__  

- __target__: `output`과 같은 형태의 텐서. 오차 계산의 기준입니다.  
- __output__: 텐서. 오차를 줄일 대상입니다.  
- __from_logits__: `output`이 로짓 함수로부터 도출된 값이면 `True`로 설정합니다. 기본값은 `False`로 이 경우 `output`은 확률분포를 나타내는 것으로 간주됩니다.  

__반환값__  
    
텐서.


----
### sigmoid

```python
keras.backend.sigmoid(x)
```

입력 텐서의 각 원소에 시그모이드<sub>sigmoid</sub> 연산을 적용합니다.

__인자__  

- __x__: 텐서 또는 변수.  

__반환값__  

텐서.

__NumPy 구현__

```python
def sigmoid(x):
    return 1. / (1. + np.exp(-x))
```


----
### hard_sigmoid

```python
keras.backend.hard_sigmoid(x)
```
각 구역별로 시그모이드 연산의 선형근사값을 반환합니다. 보통의 시그모이드보다 연산이 빠릅니다. `x < -2.5`인 경우 `0.`을, `x > 2.5`인 경우 `1.`을 반환합니다. `-2.5 <= x <= 2.5`인 경우 `0.2 * x + 0.5`를 반환합니다.  

__인자__  

- __x__: 텐서 또는 변수.  

__반환값__  

텐서.  

__NumPy 구현__

```python
def hard_sigmoid(x):
    y = 0.2 * x + 0.5
    return np.clip(y, 0, 1)
```


----
### tanh

```python
keras.backend.tanh(x)
```

각 원소에 하이퍼볼릭탄젠트<sub>tanh</sub> 연산을 적용합니다.

__인자__  

- __x__: 텐서 또는 변수.  

__반환값__   
    
텐서.  

__NumPy 구현__

```python
def tanh(x):
    return np.tanh(x)
```


----
### dropout

```python
keras.backend.dropout(x, level, noise_shape=None, seed=None)
```

`x`의 원소값을 지정한 비율에 따른 무작위 선택에 의해 `0`으로 바꿉니다. `0`으로 바뀌지 않은 원소들은 전체 텐서의 원소 수가 감소된 만큼 큰 값으로 스케일링됩니다.


__인자__  

- __x__: 텐서.  
- __level__: 텐서의 원소 가운데 값을 0으로 바꿀 비율을 정합니다.  
- __noise_shape__: 무작위로 생성할 유지/삭제 플래그의 형태를 정합니다. `x`의 형태로 브로드캐스팅될 수 있는 형태여야 합니다.  
- __seed__: 무작위 연산을 재현 가능하도록 난수 생성의 시드값을 정합니다.  

__반환값__  

텐서.

__NumPy 구현__

<details>
<summary>NumPy 구현 보기</summary>

```python
def dropout(x, level, noise_shape=None, seed=None):
    if noise_shape is None:
        noise_shape = x.shape
    if learning_phase():
        noise = np.random.choice([0, 1],
                                 noise_shape,
                                 replace=True,
                                 p=[level, 1 - level])
        return x * noise / (1 - level)
    else:
        return x
```
</details>


----
### l2_normalize


```python
keras.backend.l2_normalize(x, axis=None)
```

지정한 축을 따라 L2 노름<sub>norm</sub>으로 텐서를 정규화합니다. 

__인자__  

- __x__: 텐서 또는 변수.  
- __axis__: 정규화를 수행할 방향의 축.  

__반환값__  
    
텐서.  

__NumPy 구현__

```python
def l2_normalize(x, axis=-1):
    y = np.max(np.sum(x ** 2, axis, keepdims=True), axis, keepdims=True)
    return x / np.sqrt(y)
```


----
### in_top_k

```python
keras.backend.in_top_k(predictions, targets, k)
```

`targets`이 최상위 `k` `predictions`에 있는지를 반환합니다.  

__인자__  

- __predictions__: `float32`. `(batch_size, classes)`형식의 텐서.  
- __targets__: `int32` 또는 `int64`. 길이가 `batch_size`인 1D 텐서.  
- __k__: `int`. 고려해야 할 최상위 값의 수.  

__반환값__  
    
`batch_size`길이의 `bool`타입 텐서. `predictions[i, targets[i]]`가 `predictions[i]`의 top-`k`범위 안에 있으면 `output[i]`값은 `True`가 됩니다.
 

----
### conv1d

```python
keras.backend.conv1d(x, kernel, strides=1, padding='valid', data_format=None, dilation_rate=1)
```

1D 합성곱 연산을 수행합니다.  

__인자__  

- __x__: 텐서 또는 변수.  
- __kernel__: 커널 텐서.  
- __strides__: `int`. 합성곱 필터의 스트라이드 폭을 지정합니다.
- __padding__: `str`. `"same"`, `"causal"` 또는 `"valid"` 가운데 하나를 지정합니다.  
- __data_format__: `str`. `"channels_last"` 또는 `"channels_first"`. 이미지 입력 데이터의 채널 차원 위치를 지정합니다. `"None"`인 경우 `keras.backend.image_data_format()`에 지정된 기본값을 따릅니다.   
- __dilation_rate__: `int`. 팽창 합성곱 적용시 커널의 원소 사이의 간격을 결정하는 팽창비율입니다. 

__반환값__  
    
텐서. 1D 합성곱 연산 결과.

__오류__

- __ValueError__: `data_format`이 `"channels_last"` 또는 `"channels_first"`가 아닌 경우에 발생합니다.
    
    
----
### conv2d

```python
keras.backend.conv2d(x, kernel, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1))
```

2D 합성곱 연산을 수행합니다.

__인자__  

- __x__: 텐서 또는 변수.  
- __kernel__: 커널 텐서.  
- __strides__: `int`. 합성곱 필터의 스트라이드 폭을 지정합니다. 
- __padding__: `str`. `"same"` 또는 `"valid"` 가운데 하나를 지정합니다.  
- __data_format__: `str`. `"channels_last"` 또는 `"channels_first"`. 이미지 입력 데이터의 채널 차원 위치를 지정합니다. `"None"`인 경우 `keras.backend.image_data_format()`에 지정된 기본값을 따릅니다.  
- __dilation_rate__: 두 개의 `int`로 이루어진 튜플. 팽창 합성곱 적용시 커널의 원소 사이의 간격을 결정하는 팽창비율입니다. 

__반환값__  
    
텐서. 2D 합성곱 연산 결과.  

__오류__  

- __ValueError__: `data_format`이 `"channels_last"` 또는 `"channels_first"`가 아닌 경우에 발생합니다. 
    
  
----
### conv2d_transpose

```python
keras.backend.conv2d_transpose(x, kernel, output_shape, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1))
```

2D 형태의 전치된 합성곱 연산<sub>Transposed Convolution</sub>을 수행합니다. 전치된 합성곱 연산은 특정한 합성곱의 결과로부터 입력으로 형태를 거슬러 올라가는 역방향 변환에 주로 사용됩니다.  

__인자__  

- __x__: 텐서 또는 변수.
- __kernel__: 커널 텐서. 
- __output_shape__: 출력 형태를 나타내는 `int`형식의 1D 텐서.
- __strides__: 두 개의 `int`로 이루어진 튜플. 합성곱 필터의 스트라이드 폭을 지정합니다.
- __padding__: `str`. `"same"` 또는 `"valid"` 가운데 하나를 지정합니다.
- __data_format__: `str`. `"channels_last"` 또는 `"channels_first"`. 이미지 입력 데이터의 채널 차원 위치를 지정합니다. `"None"`인 경우 `keras.backend.image_data_format()`에 지정된 기본값을 따릅니다. 
- __dilation_rate__: 두 개의 `int`로 이루어진 튜플. 팽창 합성곱 적용시 커널의 원소 사이의 간격을 결정하는 팽창비율입니다. 


__반환값__  
    
텐서. 2D 전치된 합성곱 연산의 결과.

__오류__  

- __ValueError__: `data_format`이 `"channels_last"` 또는 `"channels_first"`가 아닌 경우에 발생합니다. 
    

----
### separable_conv1d

```python
keras.backend.separable_conv1d(x, depthwise_kernel, pointwise_kernel, strides=1, padding='valid', data_format=None, dilation_rate=1)
```

1D 분리 합성곱<sub>Separable Convolution</sub> 연산을 수행합니다.  

__인자__  

- __x__: 입력 텐서.
- __depthwise_kernel__: 깊이별 합성곱<sub>Depthwise Convolution</sub> 연산을 위한 합성곱 커널.
- __pointwise_kernel__:  1x1 합성곱에 사용할 커널.
- __strides__: `int`. 합성곱 필터의 스트라이드 폭.
- __padding__: `str`. `"same"` 또는 `"valid"` 가운데 하나를 지정합니다.
- __data_format__: `str`. `"channels_last"` 또는 `"channels_first"`. 이미지 입력 데이터의 채널 차원 위치를 지정합니다. `"None"`인 경우 `keras.backend.image_data_format()`에 지정된 기본값을 따릅니다. 
- __dilation_rate__: `int`. 팽창 합성곱 적용시 커널의 원소 사이의 간격을 결정하는 팽창비율입니다.  

__반환값__  
    
출력 텐서.  

__오류__

- __ValueError__: `data_format`이 `"channels_last"` 또는 `"channels_first"`가 아닌 경우에 발생합니다. 
  
  
----
### separable_conv2d  

```python
keras.backend.separable_conv2d(x, depthwise_kernel, pointwise_kernel, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1))
```

2D 분리 합성곱 연산을 수행합니다.  

__인자__  

- __x__: 입력 텐서.
- __depthwise_kernel__: 깊이별 합성곱 연산을 위한 합성곱 커널.
- __pointwise_kernel__:  1x1 합성곱에 사용할 커널.
- __strides__: 두 개의 `int`로 이루어진 튜플. 합성곱 필터의 스트라이드 폭을 지정합니다.
- __padding__: `str`. `"same"` 또는 `"valid"` 가운데 하나를 지정합니다.
- __data_format__: `str`. `"channels_last"` 또는 `"channels_first"`. 이미지 입력 데이터의 채널 차원 위치를 지정합니다. `"None"`인 경우 `keras.backend.image_data_format()`에 지정된 기본값을 따릅니다. 
- __dilation_rate__: 두 개의 `int`로 이루어진 튜플. 팽창 합성곱 적용시 커널의 원소 사이의 간격을 결정하는 팽창비율입니다. 
    
__반환값__  

출력 텐서.

__오류__  

- __ValueError__: `data_format`이 `"channels_last"` 또는 `"channels_first"`가 아닌 경우에 발생합니다. 
    
    
----
### depthwise_conv2d


```python
keras.backend.depthwise_conv2d(x, depthwise_kernel, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1))
```

2D 깊이별 합성곱<sub>Depthwise Convolution</sub> 연산을 수행합니다.  

__인자__

- __x__: 입력 텐서.  
- __depthwise_kernel__: 깊이별 합성곱 연산을 위한 컨볼루션 커널.  
- __strides__: 두 개의 `int`로 이루어진 튜플. 합성곱 필터의 스트라이드 폭을 지정합니다.  
- __padding__: `str`. `"same"` 또는 `"valid"` 가운데 하나를 지정합니다.  
- __data_format__: `str`. `"channels_last"` 또는 `"channels_first"`. 이미지 입력 데이터의 채널 차원 위치를 지정합니다. `"None"`인 경우 `keras.backend.image_data_format()`에 지정된 기본값을 따릅니다.  
- __dilation_rate__: 두 개의 `int`로 이루어진 튜플. 팽창 합성곱 적용시 커널의 원소 사이의 간격을 결정하는 팽창비율입니다.  

__반환값__  
   
출력 텐서.  

__오류__

- __ValueError__: `data_format`이 `"channels_last"` 또는 `"channels_first"`가 아닌 경우에 발생합니다. 
    
    
----
### conv3d

```python
keras.backend.conv3d(x, kernel, strides=(1, 1, 1), padding='valid', data_format=None, dilation_rate=(1, 1, 1))
```

3D 합성곱 연산을 수행합니다.  

__인자__  

- __x__: 텐서 또는 변수.  
- __kernel__: 커널 텐서. 
- __strides__: 세 개의 `int`로 이루어진 튜플. 합성곱 필터의 스트라이드 폭을 지정합니다.
- __padding__: `str`. `"same"` 또는 `"valid"` 가운데 하나를 지정합니다.
- __data_format__: `str`. `"channels_last"` 또는 `"channels_first"`. 이미지 입력 데이터의 채널 차원 위치를 지정합니다. `"None"`인 경우 `keras.backend.image_data_format()`에 지정된 기본값을 따릅니다. 
- __dilation_rate__: 세 개의 `int`로 이루어진 튜플. 팽창 합성곱 적용시 커널의 원소 사이의 간격을 결정하는 팽창비율입니다.  

__반환값__  
    
텐서. 3D 합성곱 연산 결과.  

__오류__  

- __ValueError__: `data_format`이 `"channels_last"` 또는 `"channels_first"`가 아닌 경우에 발생합니다. 
    
    
----
### conv3d_transpose

```python
keras.backend.conv3d_transpose(x, kernel, output_shape, strides=(1, 1, 1), padding='valid', data_format=None)
```

3D 형태의 전치된 합성곱 연산을 수행합니다. 전치된 합성곱 연산은 특정한 합성곱의 결과로부터 입력으로 형태를 거슬러 올라가는 역방향 변환에 주로 사용됩니다.    

__인자__   

- __x__: 텐서 또는 변수.
- __kernel__: 커널 텐서. 
- __output_shape__: 출력 형태를 나타내는 `int`형식의 1D 텐서.
- __strides__: 세 개의 `int`로 이루어진 튜플. 합성곱 필터의 스트라이드 폭을 지정합니다.
- __padding__: `str`. `"same"` 또는 `"valid"` 가운데 하나를 지정합니다.
- __data_format__: `str`. `"channels_last"` 또는 `"channels_first"`. 이미지 입력 데이터의 채널 차원 위치를 지정합니다. `"None"`인 경우 `keras.backend.image_data_format()`에 지정된 기본값을 따릅니다.  

__반환값__  
    
텐서. 3D 전치된 합성곱 연산의 결과.  

__오류__

- __ValueError__: `data_format`이 `"channels_last"` 또는 `"channels_first"`가 아닌 경우에 발생합니다. 
 
 
----
### pool2d

```python
keras.backend.pool2d(x, pool_size, strides=(1, 1), padding='valid', data_format=None, pool_mode='max')
```

2D 입력에 풀링<sub>pooling</sub> 연산을 적용합니다. 입력 데이터 가운데 `pool_size`로 지정한 가로 세로의 격자마다 `pool_mode`로 지정한 연산 결과로 얻어진 값을 반환하여 새로운 2D 결과값을 반환합니다.  
  
__인자__

- __x__: 텐서 또는 변수.
- __pool_size__: 두 개의 `int`로 이루어진 튜플. 풀링 필터의 크기를 지정합니다.
- __strides__: 두 개의 `int`로 이루어진 튜플. 풀링 필터의 스트라이드 폭을 지정합니다.
- __padding__: `str`. `"same"` 또는 `"valid"` 가운데 하나를 지정합니다.
- __data_format__: `str`. `"channels_last"` 또는 `"channels_first"`. 이미지 입력 데이터의 채널 차원 위치를 지정합니다. `"None"`인 경우 `keras.backend.image_data_format()`에 지정된 기본값을 따릅니다. 
- __pool_mode__: `str`. `"max"`로 지정하는 경우 풀링 범위 안의 최대값을 반환하고 `"avg"`로 지정하는 경우 풀링 범위 내 값들의 평균을 반환합니다. 기본값은 `"max"`입니다.


__반환값__ 
    
텐서. 2D 풀링 연산의 결과값.

__오류__

- __ValueError__: `data_format`이 `"channels_last"` 또는 `"channels_first"`가 아닌 경우에 발생합니다.  
- __ValueError__: `pool_mode`가 `"max"` 또는 `"avg"` 가 아닌 경우에 발생합니다.


----
### pool3d

```python
keras.backend.pool3d(x, pool_size, strides=(1, 1, 1), padding='valid', data_format=None, pool_mode='max')
```

3D 입력에 풀링<sub>pooling</sub> 연산을 적용합니다.  
  
__인자__  

- __x__: 텐서 또는 변수.
- __pool_size__: 세 개의 `int`로 이루어진 튜플. 풀링 필터의 크기를 지정합니다.
- __strides__: 세 개의 `int`로 이루어진 튜플. 풀링 필터의 스트라이드 폭을 지정합니다.
- __padding__: `str`. `"same"` 또는 `"valid"` 가운데 하나를 지정합니다.
- __data_format__: `str`. `"channels_last"` 또는 `"channels_first"`. 이미지 입력 데이터의 채널 차원 위치를 지정합니다. `"None"`인 경우 `keras.backend.image_data_format()`에 지정된 기본값을 따릅니다. 
- __pool_mode__: `str`. `"max"`로 지정하는 경우 풀링 범위 안의 최대값을 반환하고 `"avg"`로 지정하는 경우 풀링 범위 내 값들의 평균을 반환합니다. 기본값은 `"max"`입니다.

__반환값__  

텐서. 3D 풀링 연산의 결과값.  

__오류__  
  
- __ValueError__: `data_format`이 `"channels_last"` 또는 `"channels_first"`가 아닌 경우에 발생합니다.  
- __ValueError__: `pool_mode`가 `"max"` 또는 `"avg"` 가 아닌 경우에 발생합니다.  


----
### local_conv1d

```python
keras.backend.local_conv1d(inputs, kernel, kernel_size, strides, data_format=None)
```

일반적인 1D 합성곱과 달리 가중치를 공유하지 않는 1D 합성곱 연산을 수행합니다. 이 경우 합성곱 연산의 각 단계마다 서로 다른 가중치가 사용됩니다.  

__인자__

- __inputs__: `(batch_size, steps, input_dim)`으로 이루어진 3D 형태의 텐서.
- __kernel__: `(output_length, feature_dim, filters)`의 형태로 이루어진 공유되지 않은 합성곱 가중치. 
- __kernel_size__: 단일한 `int`로 이루어진 튜플. 1D 합성곱 필터의 길이를 지정합니다.
- __strides__: 단일한 `int`로 이루어진 튜플. 합성곱 연산의 스트라이드 크기를 지정합니다.
- __data_format__: `str`. `"channels_last"` 또는 `"channels_first"`. 이미지 입력 데이터의 채널 차원 위치를 지정합니다. `"None"`인 경우 `keras.backend.image_data_format()`에 지정된 기본값을 따릅니다. 


__반환값__    

(batch_size, output_length, filters) 형태의 공유되지 않은 가중치로 1D 합성곱을 수행한 결과 텐서.  

__오류__  

- __ValueError__: `data_format`이 `"channels_last"` 또는 `"channels_first"`가 아닌 경우에 발생합니다.  


----

### local_conv2d


```python
keras.backend.local_conv2d(inputs, kernel, kernel_size, strides, output_shape, data_format=None)
```


일반적인 2D 합성곱과 달리 가중치를 공유하지 않는 2D 합성곱 연산을 수행합니다. 이 경우 합성곱 연산의 각 단계마다 서로 다른 가중치가 사용됩니다. 


__인자__

- __inputs__: 
        `data_format='channels_first'`일 때, `(batch_size, filters, new_rows, new_cols)` 형태의 4D 텐서.  
        `data_format='channels_last'`일 때, `(batch_size, new_rows, new_cols, filters)` 형태의 4D 텐서.
- __kernel__: `(output_items, feature_dim, filters)`의 형태로 이루어진 공유되지 않은 합성곱 가중치. 
- __kernel_size__: 두 개의 `int`로 이루어진 튜플. 2D 합성곱 필터의 높이와 폭을 지정합니다.
- __strides__: 두 개의 `int`로 이루어진 튜플. 합성곱 연산의 세로와 가로 스트라이드 크기를 지정합니다. 
- __output_shape__: `(output_row, output_col)` 형태의 튜플.
- __data_format__: `str`. `"channels_last"` 또는 `"channels_first"`. 이미지 입력 데이터의 채널 차원 위치를 지정합니다. `"None"`인 경우 `keras.backend.image_data_format()`에 지정된 기본값을 따릅니다. 


__반환값__ 
`data_format='channels_first'`일 때, `(batch_size, filters, new_rows, new_cols)`형태의 4D 텐서.  
`data_format='channels_first'`일 때, `(batch_size, new_rows, new_cols, filters)`형태의 4D 텐서.

__오류__

- __ValueError__: `data_format`이 `"channels_last"` 또는 `"channels_first"`가 아닌 경우에 발생합니다.  

    
----
### bias_add

```python
keras.backend.bias_add(x, bias, data_format=None)
```

텐서에 편향<sub>bias</sub>벡터를 추가합니다. 

__인자__

- __x__: 텐서 또는 변수.
- __bias__: 입력 텐서에 추가할 편향값의 텐서. 
- __data_format__: `str`. `"channels_last"` 또는 `"channels_first"`. 이미지 입력 데이터의 채널 차원 위치를 지정합니다. `"None"`인 경우 `keras.backend.image_data_format()`에 지정된 기본값을 따릅니다.  
  
__반환값__  

편향을 추가한 텐서.

__오류__

- __ValueError__: `data_format`이 `"channels_last"` 또는 `"channels_first"`가 아닌 경우에 발생합니다.  
- __ValueError__: 편향의 형태가 잘못된 경우 발생합니다. 편향은 ndim(x)-1 (=입력값의 차원 개수-1) 크기의 벡터 또는 텐서여야 합니다.  
  
__NumPy 구현__  

<details>
<summary>NumPy로 구현한 코드 보기</summary>

```python
def bias_add(x, y, data_format):
    if data_format == 'channels_first':
        if y.ndim > 1:
            y = np.reshape(y, y.shape[::-1])
        for _ in range(x.ndim - y.ndim - 1):
            y = np.expand_dims(y, -1)
    else:
        for _ in range(x.ndim - y.ndim - 1):
            y = np.expand_dims(y, 0)
    return x + y
```

</details>


----
### random_normal

```python
keras.backend.random_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None)
```

정규분포에서 값을 추출하여 텐서를 생성하고 반환합니다.

__인자__  

- __shape__: `int`로 이루어진 튜플. 생성할 텐서의 형태를 정합니다.
- __mean__: `float`. 표본을 추출할 정규분포의 평균을 정합니다.
- __stddev__: `float`. 표본을 추출할 정규분포의 표준편차를 정합니다.
- __dtype__: `str`. 반환할 텐서의 자료형을 지정합니다.
- __seed__: `int`. 무작위 연산을 재현 가능하도록 난수 생성의 시드값을 정합니다.


__반환값__     

텐서.  


----
### random_uniform

```python
keras.backend.random_uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None)
```

균등분포에서 값을 추출하여 텐서를 생성하고 반환합니다.  

__인자__  

- __shape__: `int`로 이루어진 튜플. 생성할 텐서의 형태를 정합니다.
- __minval__: `float`. 표본을 추출할 균등 분포의 최솟값을 지정합니다.
- __maxval__: `float`. 표본을 추출할 균등 분포의 최댓값을 지정합니다.
- __dtype__: `str`. 반환할 텐서의 자료형을 지정합니다.
- __seed__: `int`. 무작위 연산을 재현 가능하도록 난수 생성의 시드값을 정합니다.


__반환값__     

텐서.  
    
    
----
### random_binomial

```python
keras.backend.random_binomial(shape, p=0.0, dtype=None, seed=None)
```

이항분포에서 값을 추출하여 텐서를 생성하고 반환합니다.  

__인자__

- __shape__: `int`로 이루어진 튜플. 생성할 텐서의 형태를 정합니다.
- __p__: `float`. `0. <= p <= 1`. 이항분포의 확률을 정합니다.
- __dtype__: `str`. 반환할 텐서의 자료형을 지정합니다.
- __seed__: `int`. 무작위 연산을 재현 가능하도록 난수 생성의 시드값을 정합니다.


__반환값__     

텐서.  
   
   
----
### truncated_normal

```python
keras.backend.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None)
```

절단 정규분포<sub>truncated normal</sub>에서 값을 추출하여 텐서를 생성하고 반환합니다. 절단 정규분포는 처음 지정한 정규분포의 평균과 표준편차에서 +-2 표준편차 영역 바깥을 잘라낸 다음, 남은 값들의 평균과 표준편차로 생성한 새로운 정규분포를 이용합니다.  

__인자__  

- __shape__: `int`로 이루어진 튜플. 생성할 텐서의 형태를 정합니다.
- __mean__: `float`. 표본을 추출할 정규분포의 평균을 정합니다.
- __stddev__: `float`. 표본을 추출할 정규분포의 표준편차를 정합니다.
- __dtype__: `str`. 반환할 텐서의 자료형을 지정합니다.
- __seed__: `int`. 무작위 연산을 재현 가능하도록 난수 생성의 시드값을 정합니다.

__반환값__  
    
텐서.  


----
### ctc_label_dense_to_sparse

```python
keras.backend.ctc_label_dense_to_sparse(labels, label_lengths)
```

CTC 레이블을 dense에서 sparse로 변경합니다. 

__인자__  

- __labels__: dense CTC 레이블.  
- __label_lengths__: 레이블의 길이.  

__반환값__  
    
레이블의 희소<sub>sparse</sub> 텐서 표현.


----
### ctc_batch_cost

```python
keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
```

배치의 각 원소들에 대해 CTC 손실을 구하는 알고리즘을 수행합니다.  

__인자__  

- __y_true__: 참인 레이블을 포함한 `(samples, max_string_length)` 형태의 텐서.  
- __y_pred__: 예측값 또는 Softmax의 출력값을 포함한 `(samples, time_steps, num_categories)` 형태의 텐서.  
- __input_length__: `(samples, 1)` 형태의 텐서로 `samples`는 `y_pred`의 각 배치 항목의 시퀀스 길이.  
- __label_length__: `(samples, 1)` 형태의 텐서로 `samples`는 `y_true`의 각 배치 항목의 시퀀스 길이.  
  

__반환값__  
    
각 원소들의 CTC 손실 값으로 이루어진 `(samples, 1)`형태의 텐서.  


----
### ctc_decode

```python
keras.backend.ctc_decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1)
```

Softmax의 결과를 해석합니다. 해석에는 greedy 방식의 탐색(최적 경로 탐색)과 제한된 사전<sub>dictionary</sub> 탐색을 사용할 수 있습니다.  

__인자__  

- __y_pred__: 예측값 또는 Softmax의 출력값을 포함한 `(samples, time_steps, num_categories)` 형태의 텐서.  
- __input_length__: `(samples, 1)` 형태의 텐서로 `samples`는 `y_pred`의 각 배치 항목 내의 순서값<sub>sequence</sub>의 길이.  
- __greedy__: `true`인 경우 최적 경로 탐색을 수행하며 속도가 빠릅니다. 이 경우 사전을 사용하지 않습니다. 
- __beam_width__: `greedy`가 `false`일 때 사용할 beam 탐색 방식 디코더의 폭을 지정합니다. 
- __top_paths__: `greedy`가 `false`일 때 확률이 있는 경로 가운데 상위 몇 개의 경로를 반환할 것인지를 지정합니다. 
   

__반환값__  

- __튜플__:
   - 리스트: `greedy`가 `true`인 경우 해석된 순서값(예: 문장)들을 원소로 갖는 리스트를 반환합니다.   
             `false`인 경우 확률이 가장 높은 탐색 결과로 이루어진 `top_paths`들을 반환합니다.  
    - 중요: 비어있는 레이블의 경우 `-1`로 반환됩니다.  
           `(top_paths, )`텐서는 해석된 순서값들의 로그 확률을 함께 반환합니다.  


----
### control_dependencies

```python
keras.backend.control_dependencies(control_inputs)
```

Control dependency들을 관리합니다. Control dependency는 지정한 연산을 실행할 때 앞서 필요한 다른 연산들이 순서에 따라 자동으로 이루어지도록 나열한 리스트입니다.

__인자__  

- __control_inputs__: 지정한 연산이 실행되기 전에 먼저 실행되어야 할 연산 또는 텐서 객체의 리스트. 비우고자 할 경우 `None`으로 지정할 수 있습니다.

__반환값__  

Control dependency를 적용시키는 파이썬 컨텍스트 매니저<sub>Context manager</sub>.


----
### map_fn

```python
keras.backend.map_fn(fn, elems, name=None, dtype=None)
```

지정한 함수를 각 요소에 매핑하고 그 출력을 반환합니다.

__인자__

- __fn__: `elems`에 들어있는 각 구성요소에 적용할 함수(또는 호출가능한 Callable).
- __elems__: 텐서.
- __name__: 해당 map 노드의 이름. 
- __dtype__: 출력 데이터 타입.

__반환값__  

지정한 `dtype`의 텐서.


----
### foldl

```python
keras.backend.foldl(fn, elems, initializer=None, name=None)
```

`fn`으로 지정한 함수를 지정한 순서형<sub>sequence</sub> 데이터에 순차적으로 반복 적용합니다. `fn`은 `elems`의 첫 번째 차원을 따라 적용되며 매번 이전 단계까지 누적된 실행 결과와 현재 단계의 대상을 각각 별도의 인자로 받아 계산하게 됩니다. `initializer`는 최초 단계 계산 시 '이전 단계' 인자로 사용할 초기값이며, `None`인 경우 `elems`의 첫 번째 요소를(`elems[0]`) 대신 사용하게 됩니다.
        
__인자__

- __fn__: `elems`의 각 요소에 적용할 함수 (예: `lambda acc, x: acc + x`). 
- __elems__: 텐서.
- __initializer__: 연산의 첫 단계에서 이전 단계 값으로 대신 사용할 초기값 (`None`인 경우 `elems[0]`을 대신 사용).
- __name__: 해당 foldl 노드의 이름.

__반환값__  

`initializer`와 형태와 데이터 타입이 같은 텐서.
    
    
----
### foldr


```python
keras.backend.foldr(fn, elems, initializer=None, name=None)
```

`fn`으로 지정한 함수를 지정한 순서형<sub>sequence</sub> 데이터에 **역순**으로 반복 적용합니다. `fn`은 `elems`의 첫 번째 차원을 따라 적용되며 매번 이전 단계까지 누적된 실행 결과와 현재 단계의 대상을 각각 별도의 인자로 받아 계산하게 됩니다. `initializer`는 최초 단계 계산 시 '이전 단계' 인자로 사용할 초기값이며, `None`인 경우 `elems`의 가장 마지막 요소를(`elems[-1]`) 대신 사용하게 됩니다.
        
__인자__

- __fn__: `elems`의 각 요소에 적용할 함수 (예: `lambda acc, x: acc + x`). 
- __elems__: 텐서.
- __initializer__: 연산의 첫 단계에서 이전 단계 값으로 대신 사용할 초기값 (`None`인 경우 `elems[-1]`을 대신 사용).
- __name__: 해당 foldl 노드의 이름.

__반환값__  

`initializer`와 형태와 데이터 타입이 같은 텐서.    


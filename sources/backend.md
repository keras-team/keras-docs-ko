# 케라스 백엔드

## "백엔드"는 무엇인가요? 

케라스는 딥러닝 모델을 블록쌓기처럼 구성요소를 쌓아 만들 수 있게끔 해주는 높은 차원의 모델 수준 라이브러리입니다. 케라스는 단독으로는 텐서들의 곱셈이나 합성곱과 같은 낮은 차원의 직접적인 연산을 지원하지 않습니다. 대신에 텐서 연산에 특화된 라이브러리들을 가져와 "백엔드 엔진"으로 사용합니다. 케라스는 하나의 텐서 연산 라이브러리를 골라 그에 의지하는 대신 모듈 방식으로 문제를 처리하여 서로 다른 백엔드 엔진을 케라스와 매끄럽게 연결하게끔 합니다.

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
- __shape__: 플레이스홀더의 형태. 정수값으로 이루어진 튜플로, `None`이 포함될 수 있습니다.
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

텐서 또는 변수의 형태를 정수 또는 None 값을 포함하는 튜플 형식으로 반환합니다.

__인자__
- __x__: 텐서 또는 변수. 

__반환값__ 
    
정수 또는 None을 포함한 튜플.

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

__NumPy 적용__

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

__NumPy 적용__

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

__NumPy 적용__

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

__NumPy 적용__
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

__NumPy 적용__
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

__NumPy 적용__

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

__NumPy 적용__

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

__NumPy 적용__

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

__NumPy 적용__

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

__NumPy 적용__

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

__NumPy 적용__

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
차원의 갯수가 서로 다른 텐서를 내적할 경우 Theano의 계산 방식을 따라 결과를 생성합니다. 
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
__NumPy 적용__

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
- __axes__: `int` 또는 튜플. `x`와 `y`의 순서대로 각각 감소 대상인 차원(내적 연산의 대상이 되는 차원)을 지정합니다. 단일한 정수를 입력하는 경우 `x`와 `y` 모두에 같은 차원이 지정됩니다.


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

__NumPy 적용__

<details>
<summary>NumPy 적용 보기</summary>

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

__NumPy 적용__

```python
def transpose(x):
    return np.transpose(x)
```


----
### gather

```python
keras.backend.gather(reference, indices)
```
`reference`로 지정한 텐서에서 `indices`로 지정한 인덱스의 요소들을 하나의 텐서로 배열하여 가져옵니다. 

__인자__

- __reference__: 대상을 추출할 텐서.
- __indices__: 추출 대상의 인덱스를 지정한 정수 텐서.

__반환값__  
  
`reference`와 자료형이 동일한 텐서.

__NumPy 적용__

```python
def gather(reference, indices):
    return reference[indices]
```


----
### max

```python
keras.backend.max(x, axis=None, keepdims=False)
```


텐서에 대한 최댓값.

__인자__



- __x__: 텐서 또는 변수. 
- __axis__: [-rank(x), rank(x)) 범위의 `int`로 이루어진 튜플 또는 `int` 
    최댓값을 찾기위한 축. 만약 <sag>None</sag>이라면 모든 차원에 대한 최댓값을 찾습니다. 
- __keepdims__: <sag>boolean</sag>, 차원이 유지되고 있는지에 대한 여부. 
    `keepdims`가`False` 인 경우 텐서의 rank가 1만큼 감소합니다
    `keepdims`가`True`이면 축소 된 치수는 길이 1로 유지됩니다.
    

__반환값__ 
    


x의 최대값에 대한 텐서. 

__NumPy 적용__



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


텐서에 대한 최솟값.

__인자__



- __x__: 텐서 또는 변수. 
- __axis__: [-rank(x), rank(x)) 범위의 `int`로 이루어진 튜플 또는 `int` 
    최솟값을 찾기위한 축. 만약 <sag>None</sag>이라면 모든 차원에 대한 최솟값을 찾습니다. 
- __keepdims__: <sag>boolean</sag>, 차원이 유지되고 있는지에 대한 여부. 
    `keepdims`가`False` 인 경우 텐서의 rank가 1만큼 감소합니다
    `keepdims`가`True`이면 축소 된 치수는 길이 1로 유지됩니다.


__반환값__ 
    


x의 최솟값에 대한 텐서. 

__NumPy 적용__



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


지정된 축에따른 텐서의 값들의 합.

__인자__



- __x__: 텐서 또는 변수. 
- __axis__: [-rank(x), rank(x)) 범위의 `int`로 이루어진 튜플 또는 `int` 를 합산 하기위한 축.
     만약 <sag>None</sag>이라면 모든 차원에 대한 합의 값을 찾습니다. 
- __keepdims__: <sag>boolean</sag>, 차원이 유지되고 있는지에 대한 여부. 
    `keepdims`가`False` 인 경우 텐서의 rank가 1만큼 감소합니다
    `keepdims`가`True`이면 축소 된 치수는 길이 1로 유지됩니다.
    

__반환값__ 
    


'x'의 합을 가진 텐서. 

__NumPy 적용__



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



지정된 축을 따라, 텐서의 값을 곱합니다.

__인자__

- __x__: A tensor or variable.
- __x__: 텐서 또는 변수. 
- __axis__: An integer or list of integers in [-rank(x), rank(x)) 범위 내 
    <sag>integers</sag>의 리스트 또는 <sag>integers</sag>로서, 곱을 계산한 축. 
    만약 <sag>None</sag>이라면 모든 차원에 대해 곱을 계산합니다. 
- __keepdims__: <sag>boolean</sag>, 차원이 유지되고 있는지 아닌지에 대한 진리값. 
    만약 `keepdims` 가 <sag>False</sag>라면, 텐서의 랭크가 1만큼 감소합니다. 
    만약 `keepdims` 가 <sag>True</sag>라면, 줄어든 차원이 길이 1만큼 유지됩니다. 


__반환값__ 
    


'x'의 요소들의 곱에대한 텐서.

__NumPy 적용__



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


지정된 축에 따라, 텐서 값의 누적된 합계. 


__인자__


- __x__: 텐서 또는 변수.
- __axis__: An integer, 합계를 계산하는 축.

__반환값__ 
    

x의 값에 따른 축의 누적된 합의 텐서.

__NumPy 적용__



```python
def cumsum(x, axis=0):
    return np.cumsum(x, axis=axis)
```


----

### cumprod


```python
keras.backend.cumprod(x, axis=0)
```


지정된 축에 따라, 텐서 값의 누적된 곱.


__인자__


- __x__: 텐서 또는 변수.
- __axis__: An integer, 곱 계산에 대한 축.

__반환값__ 
    

x의 값에 따른 축의 누적된 곱의 텐서.

__NumPy 적용__



```python
def cumprod(x, axis=0):
    return np.cumprod(x, axis=axis)
```


----

### var


```python
keras.backend.var(x, axis=None, keepdims=False)
```


지정된 축에 따라, 텐서의 분산.

__인자__

- __x__: 텐서 또는 변수. 
- __axis__: [-rank(x), rank(x)) 범위의  `int` 타입 리스트 또는 `int` 으로, 분산을 계산 할 축.
    <sag>None</sag> (default)이면 계산. 모든 차원에 대한 분산을 계산합니다..
- __keepdims__: <sag>boolean</sag>, 차원을 유지 하였는지에 대한 진리값.
    `keepdims` 가 <sag>False</sag>인 경우, 텐서의 랭크가 1씩 감소합니다.   
    `keepdims` 가 <sag>True</sag>인 경우, 줄어든 차원의 길이는 1로 유지됩니다. 


__반환값__ 
    

`x`의 요소의 분산을 갖는 텐서.


__NumPy 적용__



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



지정된 축과 함께 텐서의 표준 편차를 반환한다. 

__인자__

- __x__: 텐서 또는 변수. 
- __axis__:  [-rank(x), rank(x)) 범위의  `int` 타입 리스트 또는 `int` 으로, 표준편차를 계산하는 축.
    <sag>None</sag> (default)이면 계산. 모든 차원에 대한 표준편차를 계산합니다. 
- __keepdims__: <sag>boolean</sag>, 차원을 유지 하였는지에 대한 진리값.
    `keepdims` 가 <sag>False</sag>인 경우, 텐서의 랭크가 1씩 감소합니다.   
    `keepdims` 가 <sag>True</sag>인 경우, 줄어든 차원의 길이는 1로 유지됩니다. 


__반환값__ 
    

x의 요소의 표준편차에 대한 텐서. 

__NumPy 적용__



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


지정된 축에 따른 텐서의 평균.


__인자__


- __x__: 텐서 또는 변수.
- __axis__: [-rank(x), rank(x)) 범위의  `int` 타입 리스트 또는 `int` 으로, 평균을 계산하는 축.
    <sag>None</sag> (default)이면 계산. 모든 차원에 대한 평균을 계산합니다. 
- __keepdims__: <sag>boolean</sag>, 차원을 유지 하였는지에 대한 진리값.
    `keepdims` 가 <sag>False</sag>인 경우, 축의 각 항목에 대해 텐서의 랭크가 1씩 감소합니다.   
    `keepdims` 가 <sag>True</sag>인 경우, 줄어든 차원의 길이는 1로 유지됩니다. 


__반환값__ 
    


`x`의 요소의 평균을 가진 텐서. 

__NumPy 적용__



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



비트단위 감소(logical OR).

__인자__


- __x__: Tensor or variable.
- __axis__:  [-rank(x), rank(x)) 범위의  `int` 타입 리스트 또는 `int` 
    <sag>None</sag> (default)이면 계산. 모든 차원에 대한 평균을 계산합니다. 
- __keepdims__: 감소한 축을 브로드캐스트 하는지 드롭하는지에 대한 여부.


__반환값__ 
    


uint8텐서 (0s and 1s).

__NumPy 적용__



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


비트단위 감소 (logical AND).

__인자__


- __x__: 텐서 또는 변수. 
- __axis__: [-rank(x), rank(x)) 범위의  `int` 타입 리스트 또는 `int` 
    <sag>None</sag> (default)이면 계산. 모든 차원에 대한 평균을 계산합니다. 
- __keepdims__: 감소한 축을 브로드캐스트 하는지 드롭하는지에 대한 여부.


__반환값__ 
    


uint8텐서 (0s and 1s).

__NumPy 적용__



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


축에 따른 최댓값의 인덱스를 반환합니다. 

__인자__


- __x__: 텐서 또는 변수. 
- __axis__: 감소 수행에 따른 축. 

__반환값__ 
    

텐서.

__NumPy 적용__



```python
def argmax(x, axis=-1):
    return np.argmax(x, axis=axis)
```


----

### argmin


```python
keras.backend.argmin(x, axis=-1)
```


축에 따른 최솟값의 인덱스를 반환합니다.

__인자__


- __x__: 텐서 또는 변수. 
- __axis__: 축소를 수행에 따른 축.

__반환값__ 
    

텐서

__NumPy 적용__



```python
def argmin(x, axis=-1):
    return np.argmin(x, axis=axis)
```


----

### square


```python
keras.backend.square(x)
```


요소별로 제곱계산.

__인자__

- __x__: 텐서 또는 변수. 

__반환값__ 
    

텐서.
    
----

### abs


```python
keras.backend.abs(x)
```


절대값 계산.


__인자__

- __x__: 텐서 또는 변수. 

__반환값__ 
    

텐서.
    
----

### sqrt


```python
keras.backend.sqrt(x)
```


요소별 제곱근 계산.

__인자__

- __x__: 텐서 또는 변수.

__반환값__ 
    

텐서

__NumPy 적용__



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


Element-wise exponential.

__인자__

- __x__: Tensor or variable.

__반환값__ 
    

A tensor.
    
----

### log


```python
keras.backend.log(x)
```


log 취하기.

__인자__

- __x__: 텐서 또는 변수.

__반환값__ 
    

텐서.
    
----

### logsumexp


```python
keras.backend.logsumexp(x, axis=None, keepdims=False)
```


log(sum(exp(elements across dimensions of a tensor)))를 계산합니다. 
log(sum(exp(x))) 보다 수치적으로 안정된 함수입니다.
큰 입력값의 exp를 취해서 오버플로가 발생하고
작은 입력값의 log를 가져와서 언더플로가 발생하는 것을 방지합니다.


__인자__


- __x__: 텐서 또는 변수. 
- __axis__: An integer or list of integers in [-rank(x), rank(x)) 범위 내 
    <sag>integers</sag>의 리스트 또는 <sag>integers</sag>로서, <sag>logsunexp</sag>을 계산한 축. 
    만약 <sag>None</sag>이라면 모든 차원에 대해 <sag>logsunexp</sag>을 계산합니다. 
- __keepdims__: <sag>boolean</sag>, 차원이 유지되고 있는지 아닌지에 대한 진리값. 
    만약 `keepdims` 가 <sag>False</sag>라면, 텐서의 랭크가 1만큼 감소합니다. 
    만약 `keepdims` 가 <sag>True</sag>라면, 줄어든 차원이 길이 1만큼 유지됩니다. 


__반환값__ 
    

감소된 텐서. 

__NumPy 적용__



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


요소별로 가장 가까운 수로 반올림.
0.5.의 경우, 가장 가까운 짝수로 반올림 보내는 방식을 사용합니다.


__인자__

- __x__: 텐서 또는 변수. 

__반환값__ 
    

텐서.
    
----

### sign


```python
keras.backend.sign(x)
```


요소별로 sign 취하기.

__인자__

- __x__: 텐서 또는 변수.

__반환값__ 
    

텐서.
    
----

### pow


```python
keras.backend.pow(x, a)
```


요소별로 지수화.


__인자__

- __x__: 텐서 또는 변수.
- __a__: `int` 

__반환값__ 
    

텐서.

__NumPy 적용__



```python
def pow(x, a=1.):
    return np.power(x, a)
```


----

### clip


```python
keras.backend.clip(x, min_value, max_value)
```


간격이 주어지면 간격 가장자리에서 값이 잘립니다. (클리핑)

__인자__

- __x__: 텐서 또는 변수. 
- __min_value__: `float`. `int`  or tensor.
- __max_value__: `float`. `int`  or tensor.

__반환값__ 
    

텐서

__NumPy 적용__



```python
def clip(x, min_value, max_value):
    return np.clip(x, min_value, max_value)
```


----

### equal


```python
keras.backend.equal(x, y)
```


두 텐서 사이의 대등함을 비교.

__인자__

- __x__: 텐서 또는 변수.
- __y__: 텐서 또는 변수.


__반환값__ 
    

불리언 텐서.

__NumPy 적용__



```python
def equal(x, y):
    return x == y
```


----

### not_equal


```python
keras.backend.not_equal(x, y)
```

두 텐서사이 동등하지 않음을 판정.

__인자__

- __x__: 텐서 또는 변수.
- __y__: 텐서 또는 변수.

__반환값__ 
    

불리언 텐서.

__NumPy 적용__



```python
def not_equal(x, y):
    return x != y
```


----

### greater


```python
keras.backend.greater(x, y)
```


(x > y)의 진리값.

__인자__

- __x__: 텐서 또는 변수.
- __y__: 텐서 또는 변수.

__반환값__ 
    

불리언 텐서.

__NumPy 적용__



```python
def greater(x, y):
    return x > y
```


----

### greater_equal


```python
keras.backend.greater_equal(x, y)
```


(x >= y)의 진리값.

__인자__

- __x__: 텐서 또는 변수.
- __y__: 텐서 또는 변수.


__반환값__ 
    

불리언 텐서.

__NumPy 적용__



```python
def greater_equal(x, y):
    return x >= y
```


----

### less


```python
keras.backend.less(x, y)
```


(x < y)의 진리값.

__인자__

- __x__: Tensor or variable.
- __y__: Tensor or variable.

__반환값__ 
    

불리언 텐서.

__NumPy 적용__



```python
def less(x, y):
    return x < y
```


----

### less_equal


```python
keras.backend.less_equal(x, y)
```


(x <= y)의 진리값.

__인자__

- __x__: 텐서 또는 변수.
- __y__: 텐서 또는 변수.

__반환값__ 
    

불리언 텐서.

__NumPy 적용__



```python
def less_equal(x, y):
    return x <= y
```


----

### maximum


```python
keras.backend.maximum(x, y)
```


두 텐서사이 최댓값.

__인자__


- __x__: 텐서 또는 변수.
- __y__: 텐서 또는 변수.

__반환값__ 
    

한 개의 텐서.

__NumPy 적용__



```python
def maximum(x, y):
    return np.maximum(x, y)
```


----

### minimum


```python
keras.backend.minimum(x, y)
```


두 텐서 사이 최솟값.

__인자__

- __x__: 텐서 또는 변수.
- __y__: 텐서 또는 변수.

__반환값__ 
    

텐서.

__NumPy 적용__



```python
def minimum(x, y):
    return np.minimum(x, y)
```


----

### sin


```python
keras.backend.sin(x)
```


x의 sin 계산.

__인자__

- __x__: 텐서 또는 변수.

__반환값__ 
    

한 개의 텐서.
    
----

### cos


```python
keras.backend.cos(x)
```


x의 cos 계산.

__인자__

- __x__: 텐서 또는 변수.

__반환값__ 
    

텐서.
    
----

### normalize_batch_in_training


```python
keras.backend.normalize_batch_in_training(x, gamma, beta, reduction_axes, epsilon=0.001)
```


배치에 대한 평균과 표준을 계산 한 다음 배치에 배치 정규화를 적용합니다. 

__인자__

- __x__: Input 텐서 또는 변수.
- __gamma__: 입력 스케일링에 사용되는 텐서.
- __beta__: 입력을 중앙에 위치시키는 텐서.
- __reduction_axes__: 정수 반복가능, 정규화 할 축.
- __epsilon__: 퍼지 상수.


__반환값__ 
    

 `(normalized_tensor, mean, variance)` 인자의 튜플 길이.
    
----

### batch_normalization


```python
keras.backend.batch_normalization(x, mean, var, beta, gamma, axis=-1, epsilon=0.001)
```


평균, 분산, 베타와 감마가 주어진 x에 대해 배치 정규화를 수행합니다. 

I.e. returns:
`output = (x - mean) / sqrt(var + epsilon) * gamma + beta`

__인자__


- __x__: 입력 텐서 또는 변수.
- __mean__: 배치의 평균 
- __var__: 배치의 분산
- __beta__: 입력을 중앙에 위치시키는 텐서. 
- __gamma__: 입력 스케일링에 의한 텐서.
- __axis__: Integer, 정규화 시켜야 하는 축.
    (typically the features axis).
- __epsilon__: 퍼지 상수.


__반환값__ 
    

텐서.
    
----

### concatenate


```python
keras.backend.concatenate(tensors, axis=-1)
```


지정된 축에따른 텐서의 리스트 연결.


__인자__

- __tensors__: 연결 할 텐서의 목록.
- __axis__: 연결 축.

__반환값__ 
    

텐서.
    
----

### reshape


```python
keras.backend.reshape(x, shape)
```


텐서를 지정한 형식으로 다시 재정의 합니다. 

__인자__

- __x__: 텐서 또는 변수.
- __shape__: 대상이 되는 형식튜플.


__반환값__ 
    

텐서.
    
----

### permute_dimensions


```python
keras.backend.permute_dimensions(x, pattern)
```


텐서의 축을 치환합니다.

__인자__

- __x__: 텐서 또는 변수.
- __pattern__: `(0, 2, 1)`처럼 인덱스들의 차원의 튜플.


__반환값__ 
    

텐서.
    
----

### resize_images


```python
keras.backend.resize_images(x, height_factor, width_factor, data_format, interpolation='nearest')
```


4차원 텐서에 포함된 이미지들을 재조정합니다.

__인자__

- __x__: 텐서 또는 변수. to resize.
- __height_factor__: 양의 정수.
- __width_factor__: 양의 정수.
- __data_format__: `str`. `"channels_last"` 또는 `"channels_first"`.
- __interpolation__: `str`. `nearest` 또는 `bilinear` 중 하나.


__반환값__ 
    

텐서.

__오류__

- __ValueError__: `data_format`이면 'channels_last' 또는 'channels_first' 모두 아니다.
    
----

### resize_volumes


```python
keras.backend.resize_volumes(x, depth_factor, height_factor, width_factor, data_format)
```


5차원 텐서에 포함된 볼륨 크기 조정.

__인자__

- __x__: 텐서 또는 변수. to resize.
- __depth_factor__: 양의 정수.
- __height_factor__: 양의 정수.
- __width_factor__: 양의 정수.
- __data_format__: `str`. `"channels_last"` 또는 `"channels_first"`.


__반환값__ 
    

한 개의 텐서.

__오류__

- __ValueError__: `data_format`이면 'channels_last' 또는 'channels_first' 모두 아니다.
    
----

### repeat_elements


```python
keras.backend.repeat_elements(x, rep, axis)
```


`np.repeat`처럼 축을따라 텐서의 요소를 반복.

`x` 형식이 `(s1, s2, s3)` 이고, `axis` 이 `1`이면, 출력형식은 (s1, s2 * rep, s3)`입니다.

__인자__

- __x__: 텐서 또는 변수.
- __rep__: `int`. 반복횟수.
- __axis__: 반복 할 축

__반환값__ 
    

한 개의 텐서.
    
----

### repeat


```python
keras.backend.repeat(x, n)
```


2차원 텐서를 반복합니다.
만약 x가 (samples, dim)형식이고 'n'이 2라면, 출력값은 형식이 (samples, 2, dim)가 됩니다.


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



정수 시퀀스를 포함하는 1D 텐서를 생성합니다.
함수 인자는 "Theano 's arange (단 하나의 인수 만 제공되면 실제로 "stop"인수이고 "start"는 0입니다.)"와 같은 규칙을 사용합니다.


반환 된 텐서의 기본 타입은` 'int32'`입니다.
TensorFlow의 기본값과 일치합니다.

__인자__

- __start__: 시작 값.
- __stop__: 정지 값.
- __step__: 두 개의 연속적인 값의 차이.
- __dtype__: `int`  dtype

__반환값__ 
    

정수형 텐서.


----

### tile


```python
keras.backend.tile(x, n)
```


x를 n으로 나열하여 생성합니다. 


__인자__

- __x__: 텐서 또는 배열.
- __n__: `int` 의 리스트. x의 차원의 개수와 그 길이가 같다.

__반환값__ 
    

나열된 텐서.
    
----

### flatten


```python
keras.backend.flatten(x)
```


텐서를 합쳐서 나열합니다.

__인자__

- __x__: 텐서 또는 변수.

__반환값__ 
    

텐서를 1차원으로 형식을 재구성하여 나열합니다.
    
----

### batch_flatten


```python
keras.backend.batch_flatten(x)
```



n차원 텐서를 같은 0차원의 2차원 텐서로 변형합니다.
즉, 배치의 각 데이터 샘플을 위 차원의 변형에 맞게 변환합니다.


__인자__

- __x__: 텐서 또는 변수.

__반환값__ 
    

텐서.
    
----

### expand_dims


```python
keras.backend.expand_dims(x, axis=-1)
```


축의 인덱스값에 1만큼의 차원을 더한다.

__인자__

- __x__: 텐서 또는 변수.
- __axis__: 새로운 축을 추가한 위치.

__반환값__ 
    


확장한 차원들의 텐서.
    
----

### squeeze


```python
keras.backend.squeeze(x, axis)
```



축의 인덱스 값에 해당하는 텐서를 1차원의 크기만큼 제거합니다.


__인자__

- __x__: 텐서 또는 변수.
- __axis__: 없앨 축.

__반환값__ 
    



줄어든 차원의 x와 동일한 데이터를 가지는 텐서.
    
----

### temporal_padding


```python
keras.backend.temporal_padding(x, padding=(1, 1))
```


3차원 텐서의 중간차원을 채웁니다.

__인자__

- __x__: 텐서 또는 변수.
- __padding__: 2개의 `int`로 이루어진 튜플. 차원 1의 시작과 끝에 얼마나 많은 0을 추가할 지에 대한 수치.

__반환값__ 
    

3차원 텐서를 채워 넣습니다.
    
----

### spatial_2d_padding


```python
keras.backend.spatial_2d_padding(x, padding=((1, 1), (1, 1)), data_format=None)
```



4차원 텐서에서 2차원과 3차원을 채워 넣습니다.


__인자__

- __x__: 텐서 또는 변수.
- __padding__: 2 튜플들의 튜플. 채워진 패턴.
- __data_format__: `str`. `"channels_last"` 또는 `"channels_first"`.

__반환값__ 
    


채워진 4차원 텐서.

__오류__

- __ValueError__: `data_format`이면 'channels_last' 또는 'channels_first' 모두 아니다.

    
----

### spatial_3d_padding


```python
keras.backend.spatial_3d_padding(x, padding=((1, 1), (1, 1), (1, 1)), data_format=None)
```


깊이, 높이, 너비 치수를 따라 0으로 채워진 5차원 텐서를 채웁니다.

"padding [0]", "padding [1]"및 "padding [2]"는 왼쪽과 오른쪽으로 0인 치수를 각각 채 웁니다.

'channels_last'data_format의 경우 2 차원, 3 차원 및 4 차원이 채워집니다.
'channels_first'data_format의 경우 3 차원, 4 차원 및 5 차원이 채워집니다.


__인자__

- __x__: 텐서 또는 변수.
- __padding__: 3 튜플들의 튜플. 채워진 패턴.
- __data_format__: `str`. `"channels_last"` 또는 `"channels_first"`.


__반환값__ 
    


채워진 5차원 텐서.

__오류__

- __ValueError__: `data_format`이면 'channels_last' 또는 'channels_first' 모두 아니다.


----

### stack


```python
keras.backend.stack(x, axis=0)
```


랭크`R` 텐서의 <sag>list</sag>를 랭크`R + 1` 텐서에 쌓습니다.

__인자__

- __x__: 텐서들의 <sag>list</sag>
- __axis__: 텐서를 쌓을 축.

__반환값__ 
    

텐서.

__NumPy 적용__



```python
def stack(x, axis=0):
    return np.stack(x, axis=axis)
```


----

### one_hot


```python
keras.backend.one_hot(indices, num_classes)
```


정수형 텐서의 원핫 표기를 계산합니다.

__인자__

- __indices__: `(batch_size, dim1, dim2, ... dim(n-1))`형식의 n차원 정수형 텐서.
- __num_classes__: `int`. 클래스들의 개수.

__반환값__ 
    


`(batch_size, dim1, dim2, ... dim(n-1), num_classes)`형식의 입력값의 (n+1)차원의 원핫 표현형식.
    
----

### reverse


```python
keras.backend.reverse(x, axes)
```


지정된 축을 따라 텐서를 반전시킵니다.

__인자__

- __x__: 텐서를 반전시킨다.
- __axes__: 축이 반전된, 정수형 또는 반복 가능한 정수.

__반환값__ 
    

텐서.

__NumPy 적용__



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


텐서에서 슬라이스를 추출합니다.

__인자__

- __x__: 입력 텐서.
- __start__: 각 축에 따라 슬라이스의 시작 인덱스를 나타내는 텐서 또는 `int` 리스트/튜플 자료형.
- __size__: 각 축을 따라 슬라이스 할 차원의 수를 나타내는 텐서 또는 `int` 리스트/튜플 자료형.


__반환값__ 
    

A sliced tensor:
```python
new_x = x[start[0]: start[0] + size[0], ..., start[-1]: start[-1] + size[-1]]
```

__NumPy 적용__



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


Returns the value of a variable.

__인자__

- __x__: 입력 변수. 

__반환값__ 
    

넘파이 배열.
    
----

### batch_get_value


```python
keras.backend.batch_get_value(ops)
```


한 가지 이상의 텐서 변수의 값을 반환합니다.

__인자__

- __ops__: 실행할 ops 목록.

__반환값__ 
    

넘파이 배열 리스트.
    
----

### set_value


```python
keras.backend.set_value(x, value)
```


넘파이 배열에서 변수의 값을 설정합니다. 

__인자__

- __x__: 새로운 값으로 설정하는 텐서.
- __value__: 넘파이 배열로 텐서를 설정하는 값.


----

### batch_set_value


```python
keras.backend.batch_set_value(tuples)
```

한번에 밚은 텐서 변수들의 값을 설정합니다.


__인자__

- __tuples__: `(tensor, value)` 튜플 리스트, <sag>value</sag>인자는 넘파이 배열이어야 합니다.
    
----

### print_tensor


```python
keras.backend.print_tensor(x, message='')
```



평가시 <sag>message</sag>와 텐서 값을 출력합니다. 

`print_tensor`는 `x`와 동일한 새로운 텐서를 반환합니다. 이 코드는 반드시 다음코드에 사용해야 합니다. 
그렇지 않으면 평가 중 프린트 연산이 고려되지 않습니다. 



__예시__

```python
>>> x = K.print_tensor(x, message="x is: ")
```

__인자__

- __x__: 출력 할 텐서.
- __message__: 텐서와 함께 출력 할 메시지.


__반환값__ 
    

변경되지 않은 같은 텐서  `x`.
    
----

### function


```python
keras.backend.function(inputs, outputs, updates=None)
```


케라스 함수 인스턴스화하기.

__인자__

- __inputs__: 플레이스홀더 텐서의 리스트.
- __outputs__: 출력 텐서의 리스트. 
- __updates__: 업데이트 연산의 리스트.
- __**kwargs__: `tf.Session.run`에 전달되는 값.


__반환값__ 
    


넘파이 배열의 값 출력.

__오류__

- __ValueError__: 유효하지 않은 kwargs 가 전달된 경우.
    
----

### gradients


```python
keras.backend.gradients(loss, variables)
```


변수에 대한 손실의 그라디언트를 반환합니다.

__인자__

- __loss__: 최소화시킨 스칼라값 텐서.
- __variables__: 변수들의 리스트.

__반환값__ 
    

그라디언트 텐서.
    
----

### stop_gradient


```python
keras.backend.stop_gradient(variables)
```


모든 다른 변수에 대한 0 그라디언트 'variables'를 반환합니다. 

__인자__

- __variables__: 또 다른 변수에 대한 상수를 고려한 텐서 또는 텐서의 리스트.

__반환값__ 
    

전달받은 인자에 따른 또 다른 변수에 대한 상수 그라디언트를 가진 텐서 또는 텐서의 리스트.



----

### rnn


```python
keras.backend.rnn(step_function, inputs, initial_states, go_backwards=False, mask=None, constants=None, unroll=False, input_length=None)
```


텐서의 시간 차원에 대한 반복.

__인자__

- __step_function__:
    매개변수:
        inputs: 시간 차원이 없고 형식이 있는 텐서. 어떤 시간 단계의 배치에 관한 입력값을 나타냅니다.
        state: 텐서의 리스트.
    반환값:
        outputs: 시간 차원이 없고 형식이 있는 텐서. 
        new_states: 'states'의 형식과 같은 길이의 텐서 리스트. 
- __inputs__: 적어도 3차원인 형식의 일시적인 데이터의 텐서  (samples, time, ...)
- __initial_states__: 단계함수에서 사용된 상태의 초기 값을 포함한 시간 차원이 없고 형식이 있는 텐서.
- __go_backwards__: <sag>boolean</sag> 만약 True라면 그 시간동안 반복한다. 
        뒤집힌 순서를 반환하며 뒤집힌 순서의 차원이다. 
- __mask__: (samples, time)형식을 가진 이진 텐서. 마스크의 모든 요소에 0 포함.
- __constants__:  각 단계에 전달된 상수 값 리스트. 
- __unroll__:  RNN을 사용하거나 기호 루프를 사용할지에 대한 여부. (백엔드에 따라 `while_loop` 또는 `scan`)
- __input_length__: 입력 시, 시간단계의  <sag>static</sag>숫자.


__반환값__ 
    

A tuple, `(last_output, outputs, new_states)`.

last_output: `(samples, ...)` 형식의, rnn의 최근 출력값. 
outputs: `(samples, time, ...)` 형식이 있는 텐서 의 각 `outputs[s, t]`요소는 's'샘플에 대한 't'시간에 대한 단계 함수의 출력요소 입니다. 
new_states: `(samples, ...)`형식의 단계함수로 반환된 최근 상태의 텐서 리스트.


__오류__

- __ValueError__: 입력 차원이 3보다 작은 경우.
- __ValueError__: `unroll`이  `True`인 경우. 
    입력 시간 단계는 고정이 아님.
- __ValueError__: `mask` 가 존재하면 (not `None`)
    상태는 (`len(states)` == 0).

__NumPy 적용__



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
            # expand mask so that `mask[:, t].ndim == x.ndim`
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
    states_tm1 = initial_states  # tm1 means "t minus one" as in "previous timestep"
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


스칼라 값에 따라 두 연산사이를 전환합니다. 

`then_expression` 와 `else_expression` 모두 동일 모양의 기호 텐서. 


__인자__


- __condition__: 텐서 (<sag>int</sag> or <sag>bool</sag>).
- __then_expression__: 텐서 또는 텐서를 반환하는 호출가능한 값.
- __else_expression__: 텐서 또는 텐서를 반환하는 호출가능한 값.

__반환값__ 
    

지정한 텐서.

__오류__

- __ValueError__: 표현된 랭크보다 더 나은 'condition'의 랭크일 경우, 에러.

__NumPy 적용__



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


열차 단계에서 'x'를 선택하고 그렇지 않으면 'alt'를 선택합니다.

`alt`는`x`와 동일한 모양 *을 가져야합니다.

__인자__


- __x__: 훈련 단계에서 반환하는 것.
    (텐서 또는 호출가능한 텐서).
- __alt__: 그 밖의 것을 반환.
    (텐서 또는 호출가능한 텐서).
- __training__: 학습 단계를 지정한 선택적 스칼라 텐서. 
    (<sag>Python boolean</sag> 또는 <sag>Python integer</sag>)


__반환값__ 
    

플래그에 기반한 `x` 또는 `alt`.
`training` 플래그는 기본적으로 `K.learning_phase()`입니다. 
    
----

### in_test_phase


```python
keras.backend.in_test_phase(x, alt, training=None)
```


열차 단계에서 'x'를 선택하고 그렇지 않으면 'alt'를 선택합니다.

`alt`는`x`와 동일한 모양 *을 가져야합니다.

__인자__

- __x__: 테스트 단계에서 반환 할 내용. 
    (tensor or callable that returns a tensor).
- __alt__: 다른 경우 반환 할 내용.
    (tensor or callable that returns a tensor).
- __training__: 학습 단계를 지정한 선택적 스칼라 텐서. 
    (<sag>Python boolean</sag> 또는 <sag>Python integer</sag>)


__반환값__ 
    

'learning_phase()'에 기반한 `x` 또는 `alt'.
    
----

### relu


```python
keras.backend.relu(x, alpha=0.0, max_value=None, threshold=0.0)
```


정제된 선형 단위.

기본값으로, 요소별로`max(x, 0)를 반환합니다.

그 외,
`f(x) = max_value` for `x >= max_value`,
`f(x) = x` for `threshold <= x < max_value`,
`f(x) = alpha * (x - threshold)` otherwise.

__인자__

- __x__: 텐서 또는 변수. 
- __alpha__: 음수 섹션의 스칼라, 기울기 (default=`0.`).
- __max_value__: `float`. 포화상태의 임계값.
- __threshold__: `float`. 임계값 활성화에 대한 임계값.

__반환값__ 
    

텐서.

__NumPy 적용__



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


지수적증가의 선형 단위.

__인자__

- __x__: 활성화 함수를 계산할 텐서 또는 변수 입니다. 
- __alpha__: 음수 섹션의 스칼라, 기울기. 

__반환값__ 
    

텐서.

__NumPy 적용__



```python
def elu(x, alpha=1.):
    return x * (x > 0) + alpha * (np.exp(x) - 1.) * (x < 0)
```


----

### softmax


```python
keras.backend.softmax(x, axis=-1)
```


텐서의 Softmax.

__인자__

- __x__: 텐서 또는 변수. 
- __axis__: 차수 softmax가 수행 됩니다. 
    기본값은 -1을 나타내며 마지막 차원을 나타냅니다. 

__반환값__ 
    

텐서.

__NumPy 적용__



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


텐서의 Softplus.

__인자__

- __x__: 텐서 또는 변수.

__반환값__ 
    

텐서.

__NumPy 적용__



```python
def softplus(x):
    return np.log(1. + np.exp(x))
```


----

### softsign


```python
keras.backend.softsign(x)
```


텐서의 Softsign.

__인자__

- __x__: 텐서 또는 변수.

__반환값__ 
    

텐서.

__NumPy 적용__



```python
def softsign(x):
    return x / (1 + np.abs(x))
```


----

### categorical_crossentropy


```python
keras.backend.categorical_crossentropy(target, output, from_logits=False, axis=-1)
```


결과 텐서와 목표 텐서 사이의 범주형의 크로스엔트로피.

__인자__

- __target__: `output`과 같은 모양의 텐서.
- __output__: softmax의 결과 텐서.
    (unless `from_logits` is True, in which
    case `output` is expected to be the logits).
- __from_logits__: <sag>boolean</sag>, <sag>logits</sag>의 텐서이거나 softmax의 결과의 'output' 입니다. 
- __axis__: 채널 축을 지정합니다. `axis=-1`
    `channels_last`형식 데이터에 해당합니다,
    `channels_first` 데이터 형식은 `axis=1`에 해당 합니다. 

__반환값__ 
    

출력 텐서. 

__오류__

- __ValueError__: `output`의 축 도 아니고 -1도 아닌 축.

    
----

### sparse_categorical_crossentropy


```python
keras.backend.sparse_categorical_crossentropy(target, output, from_logits=False, axis=-1)
```


정수 목표를 가진 범주형 크로스엔트로피.

__인자__

- __target__: An integer tensor.
- __output__: softmax의 결과로 나온 텐서. 
    (unless `from_logits` is True, in which
    case `output` is expected to be the logits).
- __from_logits__: Boolean, whether `output` is the
    result of a softmax, or is a tensor of logits.
- __axis__:
    `channels_last` 데이터 형식에 해당하는  Int 채널 축을 지정합니다. `axis=-1`
    and `axis=1` corresponds to data format `channels_first`.

__반환값__ 
    

텐서.

__오류__

- __ValueError__: `axis`가 -1 또는 `output`의 축 모두 아니다.

    
----

### binary_crossentropy


```python
keras.backend.binary_crossentropy(target, output, from_logits=False)
```


출력 텐서와 목표 텐서 사나의 이진 크로스엔트로피.

__인자__

- __target__: `output`과 같은 형식의 텐서.
- __output__: 텐서.
- __from_logits__: logits 텐서가 출력값으로 나올 것인지에 대한 값.
    기본적으로 'output'은 확률분포를 내포 합니다. 

__반환값__ 
    

텐서.
    
----

### sigmoid


```python
keras.backend.sigmoid(x)
```


요소별로 sigmoid.

__인자__

- __x__: 텐서 또는 변수.

__반환값__ 
    

텐서.

__NumPy 적용__



```python
def sigmoid(x):
    return 1. / (1. + np.exp(-x))
```


----

### hard_sigmoid


```python
keras.backend.hard_sigmoid(x)
```



각 세그먼트의 sigmoid 선형 근사.

sigmoid보다 더 빠르다.
Returns `0.` if `x < -2.5`, `1.` if `x > 2.5`.
In `-2.5 <= x <= 2.5`, returns `0.2 * x + 0.5`.


__인자__

- __x__: 텐서 또는 변수.

__반환값__ 
    

텐서.

__NumPy 적용__



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


요소별로 tanh.

__인자__

- __x__: 텐서 또는 변수.

__반환값__ 
    

텐서.

__NumPy 적용__



```python
def tanh(x):
    return np.tanh(x)
```


----

### dropout


```python
keras.backend.dropout(x, level, noise_shape=None, seed=None)
```

전체 텐서를 스케일링하는 동안 'x'의 항목을 임의로 설정합니다. 


__인자__

- __x__: 텐서.
- __level__: 텐서 항목의 일부가 0으로 설정됩니다.
- __noise_shape__: `x`의 형식을 확장해야 하므로 유지/삭제 플래그를 랜덤으로 생성하는 형식.
- __seed__: 결정성을 보장하기 위한 난수생성.


__반환값__ 
    


텐서.

__NumPy 적용__



<details>
<summary>Show the Numpy implementation</summary>

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


지정된 축을 따라 L2 norm으로 텐서를 정규화 시킨다. 

__인자__

- __x__: 텐서 또는 변수.
- __axis__: axis along which to perform normalization. 정규화를 수행하는 축.


__반환값__ 
    

텐서.

__NumPy 적용__



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


`targets`이 최상위`k` `predictions`에 있는지를 반환합니다.

__인자__

- __predictions__: `float32`타입과  `(batch_size, classes)`형식의 텐서.
- __targets__: `batch_size` and type `int32` or `int64`의 길이의 1차원 텐서. 
- __k__: An `int`, 고려해야 할 최상위 요소의 수. 

__반환값__ 
    

A 1D tensor of length `batch_size` and type `bool`.
만약 `predictions[i, targets[i]]` 이 top-`k`내에 있다면, `output[i]` 이 `True`.
`predictions[i]'의 값. 
    
----

### conv1d


```python
keras.backend.conv1d(x, kernel, strides=1, padding='valid', data_format=None, dilation_rate=1)
```


1D convolution.

__인자__

- __x__: 텐서 또는 변수. 
- __kernel__: 커널 텐서.
- __strides__: 정수형 스트라이드. 
- __padding__: `str`. `"same"`, `"causal"` or `"valid"`.
- __data_format__: `str`. `"channels_last"` or `"channels_first"`.
- __dilation_rate__: 정수 확장 비율.



__반환값__ 
    

1차원 컨볼루션 연산 결과, 텐서 값.

__오류__

- __ValueError__:`data_format`이 모두 `"channels_last"` ,`"channels_first"`이 아닐 때.
    
    
----

### conv2d


```python
keras.backend.conv2d(x, kernel, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1))
```


2차원 컨볼루션.

__인자__

- __x__: 텐서 또는 변수.
- __kernel__: 커널 텐서.
- __strides__: 스트라이드 튜플.
- __padding__: `str`. `"same"` or `"valid"`.
- __data_format__: `str`. `"channels_last"` or `"channels_first"`.
     inputs/kernels/outputs에 대한 Theano 또는 TensorFlow/CNTK데이터 형식을 사용할 여부.
- __dilation_rate__: 2 integers의 튜플.



__반환값__ 
    

텐서, 2차원 컨볼루션 연산 결과.

__오류__

- __ValueError__: `data_format`이 모두 `"channels_last"` ,`"channels_first"`이 아닐 때.


----

### conv2d_transpose


```python
keras.backend.conv2d_transpose(x, kernel, output_shape, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1))
```


2차원의 트렌스포즈된 컨볼루션 연산을 수행합니다.

__인자__


- __x__: 텐서 또는 변수.
- __kernel__: 커널 텐서. 
- __output_shape__: 1D int tensor 출력 형식에 대해 1차원 <sag>int</sag>텐서 
- __strides__: 스트라이드 튜플. 
- __padding__: `str`. `"same"` 또는 `"valid"`.
- __data_format__: `str`. `"channels_last"` 또는 `"channels_first"`.
    inputs/kernels/outputs에 대한 Theano 또는 TensorFlow/CNTK 데이터 형태 
- __dilation_rate__: 2개의 `int`로 이루어진 튜플.


__반환값__ 
    

2차원의 트렌스포즈된 컨볼루션 결과, 텐서.

__오류__

- __ValueError__: `data_format`이 모두 `"channels_last"` ,`"channels_first"`이 아닐 때.
    
----

### separable_conv1d


```python
keras.backend.separable_conv1d(x, depthwise_kernel, pointwise_kernel, strides=1, padding='valid', data_format=None, dilation_rate=1)
```


분리가능한 필터와 1차원 컨볼루션 연산.

__인자__

- __x__: input tensor
- __depthwise_kernel__: 깊이 컨볼루션을 위한 컨볼루션 커널.
- __pointwise_kernel__:  1x1 컨볼루션에 대한 커널.
- __strides__: 스트라이드 정수형.
- __padding__: `str`. `"same"` or `"valid"`.
- __data_format__: `str`. `"channels_last"` or `"channels_first"`.
- __dilation_rate__: integer dilation rate.



__반환값__ 
    


출력 텐서.

__오류__

- __ValueError__: `data_format`이 모두 `"channels_last"` ,`"channels_first"`이 아닐 때.
    
----

### separable_conv2d


```python
keras.backend.separable_conv2d(x, depthwise_kernel, pointwise_kernel, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1))
```


분리가능한 필터와 2차원 컨볼루션 연산.

__인자__

- __x__: input tensor
- __depthwise_kernel__: 깊이 컨볼루션을 위한 컨볼루션 커널.
- __pointwise_kernel__:  1x1 컨볼루션에 대한 커널.
- __strides__: strides tuple (length 2).
- __padding__: `str`. `"same"` or `"valid"`.
- __data_format__: `str`. `"channels_last"` or `"channels_first"`.
- __dilation_rate__: integers의 튜플.
    분리가능한 컨볼루션의 팽창률.
    

__반환값__ 
    

출력 텐서.


__오류__

- __ValueError__: `data_format`이 모두 `"channels_last"` ,`"channels_first"`이 아닐 때.
    
----

### depthwise_conv2d


```python
keras.backend.depthwise_conv2d(x, depthwise_kernel, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1))
```


분리가능한 필터로 2차원 컨볼루션 연산.

__인자__

- __x__: input tensor
- __depthwise_kernel__: 깊이 별 컨볼루션 연산을 위한 컨볼루션 커널.
- __strides__: strides tuple (length 2).
- __padding__: string, `"same"` or `"valid"`.
- __data_format__: string, `"channels_last"` or `"channels_first"`.
- __dilation_rate__: integers의 튜플.
    분리가능한 컨볼루션의 팽창률.

__반환값__ 
    

출력텐서.


__오류__

- __ValueError__: `data_format`이 모두 `"channels_last"` ,`"channels_first"`이 아닐 때.
    
----

### conv3d


```python
keras.backend.conv3d(x, kernel, strides=(1, 1, 1), padding='valid', data_format=None, dilation_rate=(1, 1, 1))
```


3차원 컨볼루션 연산.

__인자__

- __x__: 텐서 또는 변수.
- __kernel__: 커널 텐서. 
- __strides__: 스트라이드 튜플. 
- __padding__: `str`. `"same"` 또는 `"valid"`.
- __data_format__: `str`. `"channels_last"` 또는 `"channels_first"`.
    inputs/kernels/outputs에 대한 Theano 또는 TensorFlow/CNTK 데이터 형태 
- __dilation_rate__: 2개의 `int`로 이루어진 튜플.


__반환값__ 
    

텐서, 3차원 컨볼루션 연산 결과.

__오류__

- __ValueError__: `data_format`이 모두 `"channels_last"` ,`"channels_first"`이 아닐 때.
    
----

### conv3d_transpose


```python
keras.backend.conv3d_transpose(x, kernel, output_shape, strides=(1, 1, 1), padding='valid', data_format=None)
```


3차원 트렌스포즈 컨볼루션.

__인자__

- __x__: 텐서 또는 변수.
- __kernel__: 커널 텐서. 
- __output_shape__: 결과값 형식에 대한 1차원 정수형 텐서.
- __strides__: 스트라이드 튜플. 
- __padding__: `str`. `"same"` 또는 `"valid"`.
- __data_format__: `str`. `"channels_last"` 또는 `"channels_first"`.
    inputs/kernels/outputs에 대한 Theano 또는 TensorFlow/CNTK 데이터 형태 
- __dilation_rate__: 2개의 `int`로 이루어진 튜플.


__반환값__ 
    

트렌스포즈된 3차원 컨볼루션 연산결과 텐서.

__오류__

- __ValueError__: `data_format`이 모두 `"channels_last"` ,`"channels_first"`이 아닐 때.
    
----

### pool2d


```python
keras.backend.pool2d(x, pool_size, strides=(1, 1), padding='valid', data_format=None, pool_mode='max')
```


2차원 풀링연산.

__인자__

- __x__: 텐서 또는 변수.
- __pool_size__: 2개의 `int`로 이루어진 튜플.
- __strides__: 2개의 `int`로 이루어진 튜플.
- __padding__: `str`. `"same"` 또는 `"valid"`.
- __data_format__: `str`. `"channels_last"` 또는 `"channels_first"`.
- __pool_mode__: `str`. `"max"`  `"avg"`.


__반환값__ 
    

2차원 풀링 연산 결과값의 텐서.

__오류__

- __ValueError__: `data_format`이 모두 `"channels_last"` ,`"channels_first"`이 아닐 때.

- __ValueError__: 만약 `pool_mode` 라면 `"max"` 또는 `"avg"` 둘 다 아니다.
    
----

### pool3d


```python
keras.backend.pool3d(x, pool_size, strides=(1, 1, 1), padding='valid', data_format=None, pool_mode='max')
```


3D Pooling.

__인자__ <sag>

- __x__: 텐서 또는 변수.
- __pool_size__: 3개의 `int`로 이루어진 튜플.
- __strides__: 3개의 `int`로 이루어진 튜플.
- __padding__: `str`. `"same"` 또는 `"valid"`.
- __data_format__: `str`. `"channels_last"` 또는 `"channels_first"`.
- __pool_mode__:  `str`. `"max"` 또는 `"avg"`.

__반환값__ 
    

텐서, 3차원 풀링 결과.

__오류__

- __ValueError__: `data_format`이 모두 `"channels_last"` ,`"channels_first"`이 아닐 때.

- __ValueError__: 만약 `pool_mode` 라면 `"max"` 또는 `"avg"` 둘 다 아니다.
    
----

### bias_add


```python
keras.backend.bias_add(x, bias, data_format=None)
```


텐서에 대한 바이어스 벡터 추가. 

__인자__

- __x__: 텐서 또는 변수.
- __bias__: 추가 할 바이어스 텐서. 
- __data_format__: `str`. `"channels_last"` 또는 `"channels_first"`.


__반환값__ 
    

결과 텐서.

__오류__

ValueError : 아래 두 경우 중 하나에서 :
1. 유효하지 않은`data_format` 인수.
2. 잘못된 편향 모양.
편향은 벡터이거나 ndim (x)-1 차원의 텐서.


__NumPy 적용__



<details>
<summary>Show the Numpy implementation</summary>

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


값의 정규분포를 포함한 텐서를 반환 합니다.

__인자__

- __shape__: `int`로 이루어진 튜플. 생성할 텐서의 형식.
- __mean__: `float`. 정규 분포의 평균 그리기.
- __stddev__: `float`. 정규 분포의 표준편차 그리기.
- __dtype__: `str`. 반환된 텐서의 dtype.
- __seed__: `int`. random seed.


__반환값__ 
    

텐서.
    
----

### random_uniform


```python
keras.backend.random_uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None)
```


값의 균등분포를 포함한 텐서를 반환 합니다. 

__인자__


- __shape__: `int`로 이루어진 튜플. 생성할 텐서의 형식.
- __minval__: `float`. 균등 분포의 하한 샘플 그리기.
- __maxval__: `float`. 균등 분포의 상한 샘플 그리기.
- __dtype__: `str`. 반환된 텐서의 dtype.
- __seed__: `int`. random seed.

__반환값__ 
    

텐서.
    
----

### random_binomial


```python
keras.backend.random_binomial(shape, p=0.0, dtype=None, seed=None)
```


값의 임의의 이항 분포의 텐서를 반환합니다. 

__인자__

- __shape__: `int`로 이루어진 튜플. 생성할 텐서의 형식.
- __p__: `float`. `0. <= p <= 1`범위의 이항 분포의 확률
- __dtype__: `str`. 반환된 텐서의 dtype.
- __seed__: `int`. random seed.

__반환값__ 
    


텐서.
    
----

### truncated_normal


```python
keras.backend.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None)
```


값의 임의의 정규분포가 잘린 텐서를 반환합니다. 


평균에 대한 두 표준편차가 제거되고 다시 지정되어 크기가 더 큰 값을 제외한 뒤
지정된 평균과 표준편차로 정규푼보에 따라 생성된 값.

__인자__

- __shape__: `int`로 이루어진 튜플. 생성할 텐서의 형식.
- __mean__: 값들의 평균.
- __stddev__: 값들의 표준편차. 
- __dtype__: `str`. 반환된 텐서의 dtype.
- __seed__: `int`. 난수생성.

__반환값__ 
    

텐서.
    
----

### ctc_label_dense_to_sparse


```python
keras.backend.ctc_label_dense_to_sparse(labels, label_lengths)
```

<sag>dense</sag>에서 <sag>sparse</sag>로 CTC레이블을 변환합니다.



__인자__

- __labels__: <sag>dense</sag> CTC 레이블.
- __label_lengths__: 레이블의 길이.

__반환값__ 
    

레이블의 희소 텐서 표현.
    
----

### ctc_batch_cost


```python
keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
```


각 배치에서 CTC손실 알고리즘을 수행합니다.  

__인자__


- __y_true__: truth 레이블을 포함한 `(samples, max_string_length)` 텐서.
- __y_pred__: softmax의 출력 또는 예측값을 포함한 `(samples, time_steps, num_categories)` 텐서.
- __input_length__: `y_pred`의 각 배치 항목의 시퀀스 길이를 포함하는 `(samples, 1)`텐서.
- __label_length__:  `y_true`의 각 배치 항목의 시퀀스 길이를 포함하는 `(samples, 1)`텐서.


__반환값__ 
    

각 요소의 CTC 손실값을 포함한 텐서의 (samples,1)형식.
    
----

### ctc_decode


```python
keras.backend.ctc_decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1)
```


소프트맥스의 결과를 해석.


그리디 탐색(최적화)이나 제한적인 딕셔너리 탐색이 가능합니다.

__인자__


- __y_pred__: 예측을 포함한  `(samples, time_steps, num_categories)` 텐서 또는 소프트맥스의 출력.
- __input_length__: `y_pred`의 각 배치 항목에 대한 시퀀스 길이를 포함한 `(samples, )`텐서. 
- __greedy__: 만약 `true`라면 훨씬 더 빠르고 좋은 탐색을 수행합니다. 딕셔너리 자료형을 사용하지 않습니다. 
- __beam_width__: `greedy`가 `false`일 때, beam 탐색 디코더가 너비의 beam으로 사용됩니다. 
- __top_paths__: `greedy`가 `false`일 때, 가장 가능할만한 경로 중에 얼마나 많은 경로가 있는지 반환합니다. 
   

__반환값__ 
    

- __Tuple__:
    List:  `greedy`가 `true`일 때, 디코딩 된 시퀀스를 포함한 요소의 리스트를 반환합니다.
        `false`일 때, 가장 높은 가능성이 있는 `top_paths`을 반환합니다.
        Important: `-1`로 비어있는 레이블을 반환합니다.
        디코딩 된 각 시퀀스의 로그확률을 포함한  `(top_paths, )`텐서.
    
----

### map_fn


```python
keras.backend.map_fn(fn, elems, name=None, dtype=None)
```

fn 함수를 요소 위에 맵핑하고 출력을 반환합니다. 

__인자__

- __fn__: <sag>elems</sag>에 있는 각 요소에 대해 호출가능.
- __elems__: 텐서
- __name__: 그래프에서 맵 노드에 대한 문자열 이름. 
- __dtype__: 출력 데이터 타입.

__반환값__ 
    

`dtype`의 텐서.
    
----

### foldl


```python
keras.backend.foldl(fn, elems, initializer=None, name=None)
```


왼쪽에서 오른쪽으로 결합하기위해 <sag>fn</sag>을 사용해 요소를 감소시킵니다.  

__인자__

- __fn__: <sag>elems</sag>에서 각 요소에 호출 될 연산기, 예를 들어, `lambda acc, x: acc + x` 
- __elems__: 텐서
- __initializer__: 사용된 첫 번째 값. (`elems[0]` in case of None)
- __name__: 그래프 fodl 노드에 대한 문자열 이름.

__반환값__ 
    

`initializer` 모양과 같은 타입의 텐서.
    
----

### foldr


```python
keras.backend.foldr(fn, elems, initializer=None, name=None)
```


<sag>fn</sag>인자를 사용하여 오른쪽에서 왼쪽으로 텐서 요소들을 줄인다.

__인자__

- __fn__: <sag>elems</sag>에서 호출가능한 각 요소와 누산기. 
    예를들어, `lambda acc, x: acc + x`
- __elems__: 텐서
- __initializer__: 사용된 첫번 째 값 (`elems[-1]` in case of None)
- __name__: 그래프에서 <sag>foldr node</sag>의 문자열 이름

__반환값__ 
    

`initializer` 모양과 같은 타입의 텐서.
    
----



### local_conv1d


```python
keras.backend.local_conv1d(inputs, kernel, kernel_size, strides, data_format=None)
```


공유되지 않은 가중치를 1D 컨볼루션에 적용합니다.

__인자__

- __inputs__: 3D 텐서의 형식: (batch_size, steps, input_dim)
- __kernel__: (output_length, feature_dim, filters)형식의 컨볼루션의 공유되지 않은 가중치.
- __kernel_size__: 1d 컨볼루션 윈도우의 길이를 지정한 단일 `int`  튜플.
- __strides__: 컨볼루션의 스타라이드 길이를 지정한 단일 `int`  튜플.
- __data_format__: 데이터 형식, channels_first 또는 channels_last


__반환값__ 
    

(batch_size, output_length, filters)형식: 공유되지 않은 가중치로 1d 컨볼루션 연산 후의 텐서.

__오류__

- __ValueError__: If `data_format`가 
    <sag>channels_last</sag> 또는 <sag>channels_first"`이 아닐 때, 오류.
       
----

### local_conv2d


```python
keras.backend.local_conv2d(inputs, kernel, kernel_size, strides, output_shape, data_format=None)
```


2D 컨볼루션에 공유되지 않은 가중치를 적용합니다.


__인자__

- __inputs__: 
        data_format='channels_first'일 때, 4D 텐서 형식:
        (batch_size, filters, new_rows, new_cols)
        data_format='channels_last'일 때, 4D 텐서 형식:
        (batch_size, new_rows, new_cols, filters)
- __kernel__: (output_items, feature_dim, filters) 형식의 컨볼루션 연산을 위한 공유되지 않은 가중치
- __kernel_size__: 2차원 컨볼루션 윈도우의 너비와 높이를 지정한 2개의 `int`로 이루어진 튜플.
- __strides__: 2<sag>integers</sag>인 튜플. 너비와 높이에 따른 컨볼루션의 스트라이드를 지정합니다. 
- __output_shape__: (output_row, output_col)형태의 튜플
- __data_format__: 데이터 형식, 'channels_first' 또는 'channels_last'.


__반환값__ 
    

4d 텐서의 형식:
data_format='channels_first'일 때,
(batch_size, filters, new_rows, new_cols)

 4d 텐서의 형식:
data_format='channels_last'일 때,
(batch_size, new_rows, new_cols, filters)

__오류__

- __ValueError__: <sag>data_format</sag>가 
            <sag>channels_last</sag> 또는 <sag>channels_first</sag>이 아니었을 때, 오류.
    

    






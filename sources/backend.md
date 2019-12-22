# Keras 백엔드

## "백앤드"는 무엇인가요? 

Keras는 딥러닝 모델을 개발하기 위한 고수준의 구성요성 요소를 제공하는 모델 레벨의 라이브러리입니다. Keras는 텐서 곱셈, 합성곱 등의 저수준의 연산을 제공하지 않습니다. 대신 Keras의 "백엔드 엔진" 역할을 하는 특수하고 잘 최적화 된 텐서 라이브러리에 의존합니다. 하나의 단일 텐서 라이브러리를 선택하고 Keras 구현을 해당 라이브러리에 묶는 대신, Keras는 모듈 방식으로 문제를 처리하여 여러 다른 백엔드 엔진들을 Keras에 매끄럽게 연결할 수 있게 합니다.

현재 Keras는 **TensorFlow**, **Theano**, 그리고 **CNTK**의 세 가지 백엔드를 지원합니다.

- [TensorFlow](http://www.tensorflow.org/) is an open-source symbolic tensor manipulation framework developed by Google.
- [Theano](http://deeplearning.net/software/theano/) is an open-source symbolic tensor manipulation framework developed by LISA Lab at   Université de Montréal.
- [CNTK](https://www.microsoft.com/en-us/cognitive-toolkit/) is an open-source toolkit for deep learning developed by Microsoft.
- [TensorFlow](http://www.tensorflow.org/)는 구글에서 만든 기호텐서 프레임워크 오픈소스입니다.
- [Theano](http://deeplearning.net/software/theano/)은 Université de Montréal LISA Lab에서 개발한 기호텐서 프레임워크 오픈소스입니다. 
- [CNTK](https://www.microsoft.com/en-us/cognitive-toolkit/)마이크로소프트에서 만든 딥러닝 개발을 위한 오픈소스 툴킷입니다.

앞으로 더 많은 백엔드 옵션을 지원할 예정입니다.

----

## 한 백엔드에서 다른 백엔드로의 전환

Keras를 한 번이라도 실행한 적이 있다면, 아래의 위치에서 Keras 구성 파일을 찾을 수 있습니다.

`$HOME/.keras/keras.json`

만약 파일이 없다면, 해당 위치에 구성 파일을 만들 수 있습니다.

**Windows(윈도우) 사용자를 위한 노트:** `$HOME`을 `%USERPROFILE%`로 바꾸십시오.

기본 구성 파일의 내용은 다음과 같습니다.

```
{
    "image_data_format": "channels_last",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}
```

단순히 `backend` 필드의 값을 `"theano"`, `"tensorflow"` 또는 `"cntk"`로 바꿔주는 것 만으로
새로운 백엔드를 사용해 Keras 코드를 실행할 수 있습니다. 

또는 아래와 같이 환경 변수 `KERAS_BACKEND`를 정의해 설정 파일에 정의된 것을 대체할 수도 있습니다. 

```bash
KERAS_BACKEND=tensorflow python -c "from keras import backend"
Using TensorFlow backend.
```

Keras에서는 `"tensorflow"`, `"theano"` 그리고 `"cntk"`외에도 사용자가 지정한 임의의 백엔드를 로드하는 것이 가능합니다.
만약 `my_module`이라는 이름의 Python 모듈을 백엔드로 사용하고자 한다면,
`keras.json` 파일의 `"backend"` 변수 값을 아래와 같이 바꿔주어야 합니다.  

```
{
    "image_data_format": "channels_last",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "my_package.my_module"
}
```

사용하고자 하는 외부 백엔드는 반드시 검증된 것이어야 하며,
`placeholder`, `variable` 그리고 `function` 세 함수들을 지원해야 합니다.

만약, 외부 백엔드가 필수 항목이 누락되어 유효하지 않은 경우라면, 누락된 항목/항목들에 대한 오류가 기록됩니다.

----

## keras.json 상세

`keras.json` 구성 파일은 아래의 설정들을 포함합니다.

```
{
    "image_data_format": "channels_last",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}
```

`$HOME/.keras/keras.json` 파일을 편집하여 설정을 변경할 수 있습니다.  

* `image_data_format`: String, either `"channels_last"` or `"channels_first"`. It specifies which data format convention Keras will follow. (`keras.backend.image_data_format()` returns it.)
  - For 2D data (e.g. image), `"channels_last"` assumes `(rows, cols, channels)` while `"channels_first"` assumes `(channels, rows, cols)`. 
  - For 3D data, `"channels_last"` assumes `(conv_dim1, conv_dim2, conv_dim3, channels)` while `"channels_first"` assumes `(channels, conv_dim1, conv_dim2, conv_dim3)`.
* `epsilon`: Float, a numeric fuzzing constant used to avoid dividing by zero in some operations.
* `floatx`: String, `"float16"`, `"float32"`, or `"float64"`. Default float precision.
* `backend`: String, `"tensorflow"`, `"theano"`, or `"cntk"`.

----

## 추상화된 Keras 백엔드를 사용하여 새로운 코드 작성하기

만약 Theano(`th`)와 Tensorflow(`tf`) 모두와 호환이 되는 Keras 모듈을 작성하고자 한다면,
아래와 같이 추상화된 Keras 백엔드 API를 사용해야 합니다. 

다음과 같이 백엔드 모듈을 사용할 수 있습니다.

```python
from keras import backend as K
```

아래는 입력 `placeholder`를 인스턴스화하는 코드입니다. 
이는 `tf.placeholder()`, `th.tensor.matrix()` 또는 `th.tensor.tensor()` 등을 실행하는 것과 같습니다.

```python
inputs = K.placeholder(shape=(2, 4, 5))
# also works:
inputs = K.placeholder(shape=(None, 4, 5))
# also works:
inputs = K.placeholder(ndim=3)
```

아래의 코드는 변수를 인스턴스화합니다. `tf.Variable()` 또는 `th.shared()`를 실행하는 것과 같습니다.

```python
import numpy as np
val = np.random.random((3, 4, 5))
var = K.variable(value=val)

# all-zeros variable:
var = K.zeros(shape=(3, 4, 5))
# all-ones:
var = K.ones(shape=(3, 4, 5))
```

구현에 필요한 대부분의 텐서 연산들은 사용법이 TensorFlow나 Theano와 크게 다르지 않습니다. 

```python
# Initializing Tensors with Random Numbers
b = K.random_uniform_variable(shape=(3, 4), low=0, high=1) # Uniform distribution
c = K.random_normal_variable(shape=(3, 4), mean=0, scale=1) # Gaussian distribution
d = K.random_normal_variable(shape=(3, 4), mean=0, scale=1)

# Tensor Arithmetic
a = b + c * K.abs(d)
c = K.dot(a, K.transpose(b))
a = K.sum(b, axis=1)
a = K.softmax(b)
a = K.concatenate([b, c], axis=-1)
# etc...
```

----

## 백엔드 함수들


### epsilon


```python
keras.backend.epsilon()
```


수치 식에 사용되는 fuzz factor(엡실론의<sag>float</sag>값)을 반환합니다.

__Returns__

A float.

__Example__

```python
>>> keras.backend.epsilon()
1e-07
```
    
----

### set_epsilon


```python
keras.backend.set_epsilon(e)
```


수치 식에 사용되는 fuzz factor의 값을 설정합니다.

__Arguments__

- __e__: <sag>float</sag>, 엡실론의 새로운 값.

__Example__

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


Returns the default float type, as a string.
(e.g. 'float16', 'float32', 'float64').


__Returns__

String, the current default float type.

__Example__

```python
>>> keras.backend.floatx()
'float32'
```
    
----

### set_floatx


```python
keras.backend.set_floatx(floatx)
```


기본 실수형 타입을 설정합니다.

__Arguments__

- __floatx__: <sag>String</sag>, 'float16', 'float32', or 'float64'.

__Example__

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


NumPy 배열을 Keras의 기본 실수형 타입으로 변환합니다.

__Arguments__

- __x__: NumPy 배열.

__Returns__

변환된 NumPy 배열

__Example__

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


Returns the default image data format convention.

__Returns__

A string, either `'channels_first'` or `'channels_last'`

__Example__

```python
>>> keras.backend.image_data_format()
'channels_first'
```
    
----

### set_image_data_format


```python
keras.backend.set_image_data_format(data_format)
```


Sets the value of the data format convention.

__Arguments__

- __data_format__: string. `'channels_first'` 또는 `'channels_last'`.

__Example__

```python
>>> from keras import backend as K
>>> K.image_data_format()
'channels_first'
>>> K.set_image_data_format('channels_last')
>>> K.image_data_format()
'channels_last'
```
    
----

### get_uid


```python
keras.backend.get_uid(prefix='')
```


디폴트 그래프의 uid 값을 가져옵니다.

__Arguments__

- __prefix__: An optional prefix of the graph.

__Returns__

그래프의 고유 식별자(uid)
    
----

### reset_uids


```python
keras.backend.reset_uids()
```


그래프의 식별자를 재설정합니다.

----

### clear_session


```python
keras.backend.clear_session()
```


현재 TF 그래프를 없애고, 새로운 TF 그래프를 만듭니다.

오래된 모델 혹은 층과의 혼란을 피할 때 유용합니다.

----

### manual_variable_initialization


```python
keras.backend.manual_variable_initialization(value)
```


수동 변수 초기화 플래그를 설정합니다.

이 boolean 플래그는 변수가 인스턴스화 될 때 초기화 되어야 하는지(기본값),
혹은 사용자가 직접 초기화를 처리해야 하는지 여부를 결정합니다. 
(e.g. via `tf.initialize_all_variables()`).

__Arguments__

- __value__: Python boolean.
    
----

### learning_phase


```python
keras.backend.learning_phase()
```


학습 단계를 나타내는 플래그를 반환합니다.


해당 플래그 변수는 학습과 테스트시에 다른 행동을 취하는 
Keras 함수에 입력으로 전달되는 bool형 텐서 (0 = 테스트, 1 = 학습)입니다.

__Returns__


학습 단계 ( 스칼라 정수 텐서 또는 파이썬 정수형 ).

----

### set_learning_phase


```python
keras.backend.set_learning_phase(value)
```


학습 단계 변수를 주어진 값으로 고정합니다.

__Arguments__

- __value__: 학습 단계 값, 0 또는 1(정수).

__Raises__

- __ValueError__: `value` 가 `0` 또는 `1`이 아닌 경우.
    
----

### is_sparse


```python
keras.backend.is_sparse(tensor)
```


희소 텐서인지 아닌지를 반환합니다. 

__Arguments__

- __tensor__: 한 개의 텐서 인스턴스.

__Returns__

A boolean.

__Example__

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


<sag>sparse</sag> 텐서에서 <sag>dense</sag>텐서로 바꿔준다.

__Arguments__

- __tensor__: <sag>sparse</sag> 텐서일 수도 있는 인스턴스.

__Returns__


한 개의 <sag>dense</sag>텐서.

__Examples__

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


변수를 인스턴스화한 후 반환합니다.

__Arguments__

- __value__: NumPy 배열, 텐서의 초기 값.
- __dtype__: 텐서 타입.
- __name__: 텐서의 이름(선택사항).
- __constraint__: 옵티마이저 업데이트 후 변수에 적용되는 투영 함수입니다(선택사항).

__Returns__

변수 인스턴스(Keras 메타 데이터 포함).

__Examples__

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

__Arguments__

- __value__: 상수 값(또는 리스트)
- __dtype__: 결과의 텐서의 요소의 형태.
- __shape__: 결과 텐서의 크기(선택사항).
- __name__: 텐서의 이름(선택사항).

__Returns__

상수 텐서
    
----

### is_keras_tensor


```python
keras.backend.is_keras_tensor(x)
```


`x`가 Keras 텐서인지 아닌지를 반환합니다.

"Keras 텐서"란 Keras 층(`Layer` 클래스) 또는 `Input`에 의해 반환된 텐서입니다.

__Arguments__

- __x__: 후보 텐서.

__Returns__

A boolean: 주어진 인자가 Keras 텐서인지의 여부.

__Raises__

- __ValueError__: `x`가 심볼릭 텐서가 아닌 경우.

__Examples__

```python
>>> from keras import backend as K
>>> from keras.layers import Input, Dense
>>> np_var = numpy.array([1, 2])
>>> K.is_keras_tensor(np_var) # A numpy array is not a symbolic tensor.
ValueError
>>> k_var = tf.placeholder('float32', shape=(1,1))
>>> # A variable indirectly created outside of keras is not a Keras tensor.
>>> K.is_keras_tensor(k_var)
False
>>> keras_var = K.variable(np_var)
>>> # A variable created with the keras backend is not a Keras tensor.
>>> K.is_keras_tensor(keras_var)
False
>>> keras_placeholder = K.placeholder(shape=(2, 4, 5))
>>> # A placeholder is not a Keras tensor.
>>> K.is_keras_tensor(keras_placeholder)
False
>>> keras_input = Input([10])
>>> K.is_keras_tensor(keras_input) # An Input is a Keras tensor.
True
>>> keras_layer_output = Dense(10)(keras_input)
>>> # Any Keras layer output is a Keras tensor.
>>> K.is_keras_tensor(keras_layer_output)
True
```
    
----

### is_tensor


```python
keras.backend.is_tensor(x)
```

----

### placeholder


```python
keras.backend.placeholder(shape=None, ndim=None, dtype=None, sparse=False, name=None)
```


플레이스홀더 텐서를 인스턴스화 한 후 반환합니다.

__Arguments__

- __shape__: 플레이스홀더의 형식
    (<sag>integer</sag> 튜플은 <sag>None</sag>요소가 없을수도 있습니다.)
- __ndim__: 텐서 축의 갯수.
    적어도 {'shape`, `ndim`} 중 하나는 반드시 명시되어야 합니다.
    만약 두 요소 모두 명시되었다면, <sag>shape</sag>가 사용됩니다.
- __dtype__: 플레이스홀더 타입.
- __sparse__: 불리언 타입<sag>Boolean</sag>,플레이스홀더가 <sag>sparse</sag>타입이어야 하는지에 대한 진리값.
- __name__: 문자열 플레이스홀더에 대한 선택적인 이름.

__Returns__

케라스의 메타데이터가 포함된 텐서 인스턴스. 

__Examples__

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


'x'가 플레이스홀더인지 아닌지를 반환한다.

__Arguments__

- __x__: 한 개의 후보 플레이스홀더.

__Returns__


불리언 값.
    
----

### shape


```python
keras.backend.shape(x)
```


텐서 또는 변수의 기호 형식을 반환합니다.

__Arguments__

- __x__: 한 개의 텐서 또는 변수. 

__Returns__

텐서 그 자체의 기호형식.

__Examples__

```python
# TensorFlow example
>>> from keras import backend as K
>>> tf_session = K.get_session()
>>> val = np.array([[1, 2], [3, 4]])
>>> kvar = K.variable(value=val)
>>> inputs = keras.backend.placeholder(shape=(2, 4, 5))
>>> K.shape(kvar)
<tf.Tensor 'Shape_8:0' shape=(2,) dtype=int32>
>>> K.shape(inputs)
<tf.Tensor 'Shape_9:0' shape=(3,) dtype=int32>
# To get integer shape (Instead, you can use K.int_shape(x))
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


<sag>int</sag> 또는 <sag>None</sag>요소의 튜플로서 변수 또는 텐서의 형식을 반환합니다. 

__Arguments__

- __x__: 텐서 또는 변수. 

__Returns__

<sag>integers</sag>(또는 <sag>None</sag>)의 튜플.

__Examples__

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

__Numpy implementation__


```python
def int_shape(x):
    return x.shape
```


----

### ndim


```python
keras.backend.ndim(x)
```


<sag>integer</sag>타입으로, 텐서의 축의 갯수를 반환합니다. 


__Arguments__

- __x__: 텐서 또는 변수.

__Returns__


축의 갯 수, 정수형(스칼라값)으로 반환합니다.

__Examples__

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

__Numpy implementation__


```python
def ndim(x):
    return x.ndim
```


----

### dtype


```python
keras.backend.dtype(x)
```


<sag>string</sag>타입으로 케라스 변수 또는 텐서의 <sag>dtype</sag>을 반환한다.

__Arguments__

- __x__: 텐서 또는 변수. 

__Returns__

'x'의 dtype<sag>string</sag>

__Examples__

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
__Numpy implementation__


```python
def dtype(x):
    return x.dtype.name
```


----

### eval


```python
keras.backend.eval(x)
```


변수의 값을 평가한다. 

__Arguments__

- __x__: 한 개의 변수. 

__Returns__

넘파이 배열.

__Examples__

```python
>>> from keras import backend as K
>>> kvar = K.variable(np.array([[1, 2], [3, 4]]), dtype='float32')
>>> K.eval(kvar)
array([[ 1.,  2.],
       [ 3.,  4.]], dtype=float32)
```
__Numpy implementation__


```python
def eval(x):
    return x
```


----

### zeros


```python
keras.backend.zeros(shape, dtype=None, name=None)
```


모두 0인 변수로 인스턴스화 하고 반환한다.


__Arguments__

- __shape__: <sag>integers</sag>의 튜플, 반환된 케라스 변수의 형식
- __dtype__: <sag>string</sag>, 반환된 케라스 변수의 데이터 타입
- __name__: <sag>string</sag>, 반환된 케라스 변수의 이름

__Returns__

Keras 메타 데이터를 포함한 `0.0`으로 채워진 변수.
`shape`가 기호 인 경우 변수를 반환 할 수 없습니다.
대신 동적 모양의 텐서를 반환합니다.

__Example__

```python
>>> from keras import backend as K
>>> kvar = K.zeros((3,4))
>>> K.eval(kvar)
array([[ 0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.]], dtype=float32)
```
__Numpy implementation__


```python
def zeros(shape, dtype=floatx(), name=None):
    return np.zeros(shape, dtype=dtype)
```


----

### ones


```python
keras.backend.ones(shape, dtype=None, name=None)
```


모든 변수를 인스턴스화하고 반환합니다.

__Arguments__

- __shape__: <sag>integers</sag>의 튜플, 반환된 케라스 변수 형식.
- __dtype__: <sag>string</sag>, 반환된 케라스 데이터 타입. 
- __name__: <sag>string</sag>, 반환된 케라스 변수 이름.

__Returns__



`1.0`으로 채워진 Keras 변수.
`shape`가 기호 인 경우 변수를 반환 할 수 없습니다.
대신 동적 모양의 텐서를 반환합니다.

__Example__

```python
>>> from keras import backend as K
>>> kvar = K.ones((3,4))
>>> K.eval(kvar)
array([[ 1.,  1.,  1.,  1.],
       [ 1.,  1.,  1.,  1.],
       [ 1.,  1.,  1.,  1.]], dtype=float32)
```
__Numpy implementation__


```python
def ones(shape, dtype=floatx(), name=None):
    return np.ones(shape, dtype=dtype)
```


----

### eye


```python
keras.backend.eye(size, dtype=None, name=None)
```


단위행렬을 인스턴스화 하고 반환합니다. 


__Arguments__


- __size__: <sag>integer</sag>, 행과 열의 수. 
- __dtype__: <sag>string</sag>, 반환된 케라스 변수의 데이터 타입. 
- __name__: <sag>string</sag>,  반환된 케라스 변수의 이름. 

__Returns__

단위행렬, 케라스 변수. 

__Example__

```python
>>> from keras import backend as K
>>> kvar = K.eye(3)
>>> K.eval(kvar)
array([[ 1.,  0.,  0.],
       [ 0.,  1.,  0.],
       [ 0.,  0.,  1.]], dtype=float32)
```
__Numpy implementation__


```python
def eye(size, dtype=None, name=None):
    return np.eye(size, dtype=dtype)
```


----

### zeros_like


```python
keras.backend.zeros_like(x, dtype=None, name=None)
```


또 다른 텐서이면서 같은 형식의 모두 0값인 변수가 인스턴스화 됩니다.

__Arguments__


- __x__: 케라스 변수 또는 케라스 텐서. 
- __dtype__: <sag>string</sag>, 반환된 케라스 변수의 dtype.
     x의 dtype을 사용하지 않습니다.
- __name__: <sag>string</sag>, 생성할 변수의 이름. 


__Returns__


0으로 채워진 x 형식의 케라스 변수.

__Example__

```python
>>> from keras import backend as K
>>> kvar = K.variable(np.random.random((2,3)))
>>> kvar_zeros = K.zeros_like(kvar)
>>> K.eval(kvar_zeros)
array([[ 0.,  0.,  0.],
       [ 0.,  0.,  0.]], dtype=float32)
```
__Numpy implementation__


```python
def zeros_like(x, dtype=floatx(), name=None):
    return np.zeros_like(x, dtype=dtype)
```


----

### ones_like


```python
keras.backend.ones_like(x, dtype=None, name=None)
```


또 다른 텐서와 동일한 모양의 <sag>all-ones</sag> 변수를 인스턴스화 합니다.


__Arguments__



- __x__: 케라스 변수 또는 케라스 텐서. 
- __dtype__: <sag>string</sag>, 반환된 케라스 변수의 dtype.
     x의 dtype을 사용하지 않습니다.
- __name__: <sag>string</sag>, 생성할 변수의 이름. 

__Returns__

ones로 전달된 형식에 대한 케라스 변수.

__Example__

```python
>>> from keras import backend as K
>>> kvar = K.variable(np.random.random((2,3)))
>>> kvar_ones = K.ones_like(kvar)
>>> K.eval(kvar_ones)
array([[ 1.,  1.,  1.],
       [ 1.,  1.,  1.]], dtype=float32)
```
__Numpy implementation__


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

__Arguments__

- __x__: 입력텐서.
- __name__: <sag>string</sag>, 생성 할 변수의 이름.

__Returns__


형식 및 내용이 같은 텐서.
    
----

### random_uniform_variable


```python
keras.backend.random_uniform_variable(shape, low, high, dtype=None, name=None, seed=None)
```


균등 분포에서 가져온 값의 변수를 인스턴스화 합니다. 

__Arguments__


- __shape__: <sag>integers</sag>의 튜플, 반환된 케라스 변수의 형식.
- __low__: <sag>float</sag>, 출력 범위의 하한.
- __high__: <sag>float</sag>, 출력 번위의 상한. 
- __dtype__: <sag>string</sag>, 반환된 케라스 변수의 dtype.
- __name__: <sag>string</sag>, 반환된 케라스 변수의 이름.
- __seed__: <sag>integer</sag>, 난수생성.

__Returns__


샘플에서 가져온 케라스 변수.

__Example__

```python
# TensorFlow example
>>> kvar = K.random_uniform_variable((2,3), 0, 1)
>>> kvar
<tensorflow.python.ops.variables.Variable object at 0x10ab40b10>
>>> K.eval(kvar)
array([[ 0.10940075,  0.10047495,  0.476143  ],
       [ 0.66137183,  0.00869417,  0.89220798]], dtype=float32)
```
__Numpy implementation__


```python
def random_uniform_variable(shape, low, high, dtype=None, name=None, seed=None):
    return (high - low) * np.random.random(shape).astype(dtype) + low
```


----

### random_normal_variable


```python
keras.backend.random_normal_variable(shape, mean, scale, dtype=None, name=None, seed=None)
```


정규 분포에서 가져온 값의 변수를 인스턴스화 합니다. 

__Arguments__


- __shape__: <sag>integers</sag>의 튜플, 반환된 케라스 변수의 형식.
- __mean__: <sag>float</sag>, 정규분포의 평균.
- __scale__: <sag>float</sag>, 정규분포의 표준편차.
- __dtype__: <sag>string</sag>, 반환된 케라스 변수의 dtype.
- __name__: <sag>string</sag>, 반환된 케라스 변수의 이름.
- __seed__: <sag>integer</sag>, 난수생성.

__Returns__


샘플에서 가져온 케라스 변수.

__Example__

```python
# TensorFlow example
>>> kvar = K.random_normal_variable((2,3), 0, 1)
>>> kvar
<tensorflow.python.ops.variables.Variable object at 0x10ab12dd0>
>>> K.eval(kvar)
array([[ 1.19591331,  0.68685907, -0.63814116],
       [ 0.92629528,  0.28055015,  1.70484698]], dtype=float32)
```
__Numpy implementation__


```python
def random_normal_variable(shape, mean, scale, dtype=None, name=None, seed=None):
    return scale * np.random.randn(*shape).astype(dtype) + mean
```


----

### count_params


```python
keras.backend.count_params(x)
```


케라스 변수 또는 텐서에서 요소들의 <sag>static</sag> 숫자를 반환합니다. 

__Arguments__

- __x__: 케라스 텐서 또는 변수.

__Returns__


<sag>integer</sag>,`x`요소의 갯수,
즉, 배열의 정적차원<sag>static dimensions</sag>의 곱 연산.

__Example__

```python
>>> kvar = K.zeros((2,3))
>>> K.count_params(kvar)
6
>>> K.eval(kvar)
array([[ 0.,  0.,  0.],
       [ 0.,  0.,  0.]], dtype=float32)
```
__Numpy implementation__


```python
def count_params(x):
    return x.size
```


----

### cast


```python
keras.backend.cast(x, dtype)
```


텐서를 다른 dtype으로 타입을 바꿔주고 반환합니다.

케라스 변수 타입을 바꿔줄 수 있으나 여전히 텐서를 반환합니다. 

__Arguments__



- __x__: 케라스 텐서 또는 변수.
- __dtype__: <sag>string</sag>, 'float16', 'float32', 또는 'float64'


__Returns__

<sag>dtype</sag>의 케라스 텐서.

__Example__

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


x값을 new_x로 갱신합니다.

__Arguments__


- __x__: 한개의 변수.
- __new_x__: x의 같은 형식의 텐서. 

__Returns__


x변수를 갱신합니다.
    
----

### update_add


```python
keras.backend.update_add(x, increment)
```


<sag>increment</sag>를 x에 더한 값을 갱신합니다. 

__Arguments__


- __x__: 변수.
- __increment__: x와 같은 형식의 텐서.


__Returns__


변수 x 갱신.
    
----

### update_sub


```python
keras.backend.update_sub(x, decrement)
```


<sag>decrement</sag>를 뺀 후  x의 값 갱신.

__Arguments__


- __x__: A `Variable`.
- __decrement__: x와 같은 형식의 텐서.

__Returns__


변수 x 갱신.
    
----

### moving_average_update


```python
keras.backend.moving_average_update(x, value, momentum)
```


변수의 이동평균을 계산합니다. 

__Arguments__


- __x__: `Variable`.
- __value__:같은`x`형식의 텐서.
- __momentum__: 이동 평균 운동량.

__Returns__

변수를 업데이트하는 연산.
    
----

### dot


```python
keras.backend.dot(x, y)
```


2 텐서(또는 변수)를 곱하고 텐서를 반환합니다. 
N차원의 텐서를 곱하려고 시도할 때, N차원의 텐서가 Theano의 방식으로 다시 생성합니다. 
(e.g. `(2, 3) * (4, 3, 5) -> (2, 4, 5)`)

__Arguments__


- __x__: 텐서 또는 변수.
- __y__: 텐서 또는 변수.

__Returns__

`x` 과 `y`의 내적을 텐서로 반환.

__Examples__

```python
# dot product between tensors
>>> x = K.placeholder(shape=(2, 3))
>>> y = K.placeholder(shape=(3, 4))
>>> xy = K.dot(x, y)
>>> xy
<tf.Tensor 'MatMul_9:0' shape=(2, 4) dtype=float32>
```

```python
# dot product between tensors
>>> x = K.placeholder(shape=(32, 28, 3))
>>> y = K.placeholder(shape=(3, 4))
>>> xy = K.dot(x, y)
>>> xy
<tf.Tensor 'MatMul_9:0' shape=(32, 28, 4) dtype=float32>
```

```python
# Theano-like behavior example
>>> x = K.random_uniform_variable(shape=(2, 3), low=0, high=1)
>>> y = K.ones((4, 3, 5))
>>> xy = K.dot(x, y)
>>> K.int_shape(xy)
(2, 4, 5)
```
__Numpy implementation__


```python
def dot(x, y):
    return np.dot(x, y)
```


----

### batch_dot


```python
keras.backend.batch_dot(x, y, axes=None)
```


배치방식의 내적.

x와 y가 배치 데이터일 때, x와 y의 내적을 계산하여 batch_dot을 사용한다. 즉, (batch_size, :)형식.
batch_dot은 입력값보다 차수가 작은 텐서 또는 변수를 반환합니다. 
차원의 수가 1로 줄어들면 적어도 2차원이상인지 확인하기 위해 expand_dims를 사용합니다.

__Arguments__


- __x__: `ndim >= 2` 조건의 케라스 텐서 또는 변수.
- __y__: `ndim >= 2` 조건의 케라스 텐서 또는 변수. 
- __axes__: 목적 차원이 감소된 (int,int)튜플 또는 <sag>int</sag>

__Returns__

x 형식의 연쇄와 같은 형식의 텐서와 y형식. y형식은 배치차원과 합산된 차원보다 더 적습니다. 
rank가 1이면, (batch_size,1)로 재설정합니다. 

__Examples__

`x = [[1, 2], [3, 4]]` and `y = [[5, 6], [7, 8]]`일 때,
`batch_dot(x, y, axes=1) = [[17], [53]]` 
비대각선을 계산할 필요가 없을 때도, `x.dot(y.T)` 주대각선 계산.

Pseudocode:
```
inner_products = []
for xi, yi in zip(x, y):
    inner_products.append(xi.dot(yi))
result = stack(inner_products)
```

형식 추론하기:
`x`의 모양은`(100, 20)`이되고`y`의 모양은`(100, 30, 20)`이됩니다.
'축'이 (1, 2) 인 경우, 결과 텐서의 출력값 형식을 찾으려면,
`x` 형식과`y`형식으로 각 차원을 반복합니다.

*`x.shape [0]`: 100 : 출력 형식에 추가
*`x.shape [1]`: 20 : 출력 형식에 추가하지 않습니다, `x`의 차원 1이 됩니다. (`dot_axes [0]`= 1)
*`y.shape [0]`: 100 : 출력 형태에 추가하지 않습니다, 항상 y의 첫 번째 차원을 배제합니다.
*`y.shape [1]`: 30 : 출력 형식에 추가
*`y.shape [2]`: 20 : 출력 형식에 추가하지 않습니다, `y`의 차원 2이 됩니다. (`dot_axes [1]`= 2)
`output_shape` =`(100, 30)`


```python
>>> x_batch = K.ones(shape=(32, 20, 1))
>>> y_batch = K.ones(shape=(32, 30, 20))
>>> xy_batch_dot = K.batch_dot(x_batch, y_batch, axes=(1, 2))
>>> K.int_shape(xy_batch_dot)
(32, 1, 30)
```

__Numpy implementation__


<details>
<summary>Show the Numpy implementation</summary>

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


Transposes a tensor and returns it.

__Arguments__

- __x__: 텐서 또는 변수. 

__Returns__

텐서.

__Examples__

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
__Numpy implementation__


```python
def transpose(x):
    return np.transpose(x)
```


----

### gather


```python
keras.backend.gather(reference, indices)
```


텐서 `reference`에서 `indices`의 인덱스 요소를 검색합니다.


__Arguments__


- __reference__: 텐서.
- __indices__: 인덱스의 <sag>integer</sag>텐서.

__Returns__


<sag>reference</sag>와 같은 타입의 텐서.

__Numpy implementation__


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

__Arguments__



- __x__: 텐서 또는 변수. 
- __axis__: [-rank(x), rank(x)) 범위의 <sag>integers</sag>의 튜플 또는 <sag>integer</sag>
    최댓값을 찾기위한 축. 만약 <sag>None</sag>이라면 모든 차원에 대한 최댓값을 찾습니다. 
- __keepdims__: <sag>boolean</sag>, 차원이 유지되고 있는지에 대한 여부. 
    `keepdims`가`False` 인 경우 텐서의 rank가 1만큼 감소합니다
    `keepdims`가`True`이면 축소 된 치수는 길이 1로 유지됩니다.
    

__Returns__


x의 최대값에 대한 텐서. 

__Numpy implementation__


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

__Arguments__



- __x__: 텐서 또는 변수. 
- __axis__: [-rank(x), rank(x)) 범위의 <sag>integers</sag>의 튜플 또는 <sag>integer</sag>
    최솟값을 찾기위한 축. 만약 <sag>None</sag>이라면 모든 차원에 대한 최솟값을 찾습니다. 
- __keepdims__: <sag>boolean</sag>, 차원이 유지되고 있는지에 대한 여부. 
    `keepdims`가`False` 인 경우 텐서의 rank가 1만큼 감소합니다
    `keepdims`가`True`이면 축소 된 치수는 길이 1로 유지됩니다.


__Returns__


x의 최솟값에 대한 텐서. 

__Numpy implementation__


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

__Arguments__



- __x__: 텐서 또는 변수. 
- __axis__: [-rank(x), rank(x)) 범위의 <sag>integers</sag>의 튜플 또는 <sag>integer</sag>를 합산 하기위한 축.
     만약 <sag>None</sag>이라면 모든 차원에 대한 합의 값을 찾습니다. 
- __keepdims__: <sag>boolean</sag>, 차원이 유지되고 있는지에 대한 여부. 
    `keepdims`가`False` 인 경우 텐서의 rank가 1만큼 감소합니다
    `keepdims`가`True`이면 축소 된 치수는 길이 1로 유지됩니다.
    

__Returns__


'x'의 합을 가진 텐서. 

__Numpy implementation__


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

__Arguments__

- __x__: A tensor or variable.
- __x__: 텐서 또는 변수. 
- __axis__: An integer or list of integers in [-rank(x), rank(x)) 범위 내 
    <sag>integers</sag>의 리스트 또는 <sag>integers</sag>로서, 곱을 계산한 축. 
    만약 <sag>None</sag>이라면 모든 차원에 대해 곱을 계산합니다. 
- __keepdims__: <sag>boolean</sag>, 차원이 유지되고 있는지 아닌지에 대한 진리값. 
    만약 `keepdims` 가 <sag>False</sag>라면, 텐서의 랭크가 1만큼 감소합니다. 
    만약 `keepdims` 가 <sag>True</sag>라면, 줄어든 차원이 길이 1만큼 유지됩니다. 


__Returns__


'x'의 요소들의 곱에대한 텐서.

__Numpy implementation__


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


__Arguments__


- __x__: 텐서 또는 변수.
- __axis__: An integer, 합계를 계산하는 축.

__Returns__

x의 값에 따른 축의 누적된 합의 텐서.

__Numpy implementation__


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


__Arguments__


- __x__: 텐서 또는 변수.
- __axis__: An integer, 곱 계산에 대한 축.

__Returns__

x의 값에 따른 축의 누적된 곱의 텐서.

__Numpy implementation__


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

__Arguments__

- __x__: 텐서 또는 변수. 
- __axis__: [-rank(x), rank(x)) 범위의  <sag>integer</sag>타입 리스트 또는 <sag>integer</sag>으로, 분산을 계산 할 축.
    <sag>None</sag> (default)이면 계산. 모든 차원에 대한 분산을 계산합니다..
- __keepdims__: <sag>boolean</sag>, 차원을 유지 하였는지에 대한 진리값.
    `keepdims` 가 <sag>False</sag>인 경우, 텐서의 랭크가 1씩 감소합니다.   
    `keepdims` 가 <sag>True</sag>인 경우, 줄어든 차원의 길이는 1로 유지됩니다. 


__Returns__

`x`의 요소의 분산을 갖는 텐서.


__Numpy implementation__


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

__Arguments__

- __x__: 텐서 또는 변수. 
- __axis__:  [-rank(x), rank(x)) 범위의  <sag>integer</sag>타입 리스트 또는 <sag>integer</sag>으로, 표준편차를 계산하는 축.
    <sag>None</sag> (default)이면 계산. 모든 차원에 대한 표준편차를 계산합니다. 
- __keepdims__: <sag>boolean</sag>, 차원을 유지 하였는지에 대한 진리값.
    `keepdims` 가 <sag>False</sag>인 경우, 텐서의 랭크가 1씩 감소합니다.   
    `keepdims` 가 <sag>True</sag>인 경우, 줄어든 차원의 길이는 1로 유지됩니다. 


__Returns__

x의 요소의 표준편차에 대한 텐서. 

__Numpy implementation__


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


__Arguments__


- __x__: 텐서 또는 변수.
- __axis__: [-rank(x), rank(x)) 범위의  <sag>integer</sag>타입 리스트 또는 <sag>integer</sag>으로, 평균을 계산하는 축.
    <sag>None</sag> (default)이면 계산. 모든 차원에 대한 평균을 계산합니다. 
- __keepdims__: <sag>boolean</sag>, 차원을 유지 하였는지에 대한 진리값.
    `keepdims` 가 <sag>False</sag>인 경우, 축의 각 항목에 대해 텐서의 랭크가 1씩 감소합니다.   
    `keepdims` 가 <sag>True</sag>인 경우, 줄어든 차원의 길이는 1로 유지됩니다. 


__Returns__


`x`의 요소의 평균을 가진 텐서. 

__Numpy implementation__


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

__Arguments__


- __x__: Tensor or variable.
- __axis__:  [-rank(x), rank(x)) 범위의  <sag>integer</sag>타입 리스트 또는 <sag>integer</sag>
    <sag>None</sag> (default)이면 계산. 모든 차원에 대한 평균을 계산합니다. 
- __keepdims__: 감소한 축을 브로드캐스트 하는지 드롭하는지에 대한 여부.


__Returns__


uint8텐서 (0s and 1s).

__Numpy implementation__


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

__Arguments__


- __x__: 텐서 또는 변수. 
- __axis__: [-rank(x), rank(x)) 범위의  <sag>integer</sag>타입 리스트 또는 <sag>integer</sag>
    <sag>None</sag> (default)이면 계산. 모든 차원에 대한 평균을 계산합니다. 
- __keepdims__: 감소한 축을 브로드캐스트 하는지 드롭하는지에 대한 여부.


__Returns__


uint8텐서 (0s and 1s).

__Numpy implementation__


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

__Arguments__


- __x__: 텐서 또는 변수. 
- __axis__: 감소 수행에 따른 축. 

__Returns__

텐서.

__Numpy implementation__


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

__Arguments__


- __x__: 텐서 또는 변수. 
- __axis__: 축소를 수행에 따른 축.

__Returns__

텐서

__Numpy implementation__


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

__Arguments__

- __x__: 텐서 또는 변수. 

__Returns__

텐서.
    
----

### abs


```python
keras.backend.abs(x)
```


절대값 계산.


__Arguments__

- __x__: 텐서 또는 변수. 

__Returns__

텐서.
    
----

### sqrt


```python
keras.backend.sqrt(x)
```


요소별 제곱근 계산.

__Arguments__

- __x__: 텐서 또는 변수.

__Returns__

텐서

__Numpy implementation__


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

__Arguments__

- __x__: Tensor or variable.

__Returns__

A tensor.
    
----

### log


```python
keras.backend.log(x)
```


log 취하기.

__Arguments__

- __x__: 텐서 또는 변수.

__Returns__

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


__Arguments__


- __x__: 텐서 또는 변수. 
- __axis__: An integer or list of integers in [-rank(x), rank(x)) 범위 내 
    <sag>integers</sag>의 리스트 또는 <sag>integers</sag>로서, <sag>logsunexp</sag>을 계산한 축. 
    만약 <sag>None</sag>이라면 모든 차원에 대해 <sag>logsunexp</sag>을 계산합니다. 
- __keepdims__: <sag>boolean</sag>, 차원이 유지되고 있는지 아닌지에 대한 진리값. 
    만약 `keepdims` 가 <sag>False</sag>라면, 텐서의 랭크가 1만큼 감소합니다. 
    만약 `keepdims` 가 <sag>True</sag>라면, 줄어든 차원이 길이 1만큼 유지됩니다. 


__Returns__

감소된 텐서. 

__Numpy implementation__


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


__Arguments__

- __x__: 텐서 또는 변수. 

__Returns__

텐서.
    
----

### sign


```python
keras.backend.sign(x)
```


요소별로 sign 취하기.

__Arguments__

- __x__: 텐서 또는 변수.

__Returns__

텐서.
    
----

### pow


```python
keras.backend.pow(x, a)
```


요소별로 지수화.


__Arguments__

- __x__: 텐서 또는 변수.
- __a__: <sag>integer</sag>

__Returns__

텐서.

__Numpy implementation__


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

__Arguments__

- __x__: 텐서 또는 변수. 
- __min_value__: <sag>float</sag>, <sag>integer</sag> or tensor.
- __max_value__: <sag>float</sag>, <sag>integer</sag> or tensor.

__Returns__

텐서

__Numpy implementation__


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

__Arguments__

- __x__: 텐서 또는 변수.
- __y__: 텐서 또는 변수.


__Returns__

불리언 텐서.

__Numpy implementation__


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

__Arguments__

- __x__: 텐서 또는 변수.
- __y__: 텐서 또는 변수.

__Returns__

불리언 텐서.

__Numpy implementation__


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

__Arguments__

- __x__: 텐서 또는 변수.
- __y__: 텐서 또는 변수.

__Returns__

불리언 텐서.

__Numpy implementation__


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

__Arguments__

- __x__: 텐서 또는 변수.
- __y__: 텐서 또는 변수.


__Returns__

불리언 텐서.

__Numpy implementation__


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

__Arguments__

- __x__: Tensor or variable.
- __y__: Tensor or variable.

__Returns__

불리언 텐서.

__Numpy implementation__


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

__Arguments__

- __x__: 텐서 또는 변수.
- __y__: 텐서 또는 변수.

__Returns__

불리언 텐서.

__Numpy implementation__


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

__Arguments__


- __x__: 텐서 또는 변수.
- __y__: 텐서 또는 변수.

__Returns__

한 개의 텐서.

__Numpy implementation__


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

__Arguments__

- __x__: 텐서 또는 변수.
- __y__: 텐서 또는 변수.

__Returns__

텐서.

__Numpy implementation__


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

__Arguments__

- __x__: 텐서 또는 변수.

__Returns__

한 개의 텐서.
    
----

### cos


```python
keras.backend.cos(x)
```


x의 cos 계산.

__Arguments__

- __x__: 텐서 또는 변수.

__Returns__

텐서.
    
----

### normalize_batch_in_training


```python
keras.backend.normalize_batch_in_training(x, gamma, beta, reduction_axes, epsilon=0.001)
```


배치에 대한 평균과 표준을 계산 한 다음 배치에 배치 정규화를 적용합니다. 

__Arguments__

- __x__: Input 텐서 또는 변수.
- __gamma__: 입력 스케일링에 사용되는 텐서.
- __beta__: 입력을 중앙에 위치시키는 텐서.
- __reduction_axes__: 정수 반복가능, 정규화 할 축.
- __epsilon__: 퍼지 상수.


__Returns__

 `(normalized_tensor, mean, variance)` 인자의 튜플 길이.
    
----

### batch_normalization


```python
keras.backend.batch_normalization(x, mean, var, beta, gamma, axis=-1, epsilon=0.001)
```


평균, 분산, 베타와 감마가 주어진 x에 대해 배치 정규화를 수행합니다. 

I.e. returns:
`output = (x - mean) / sqrt(var + epsilon) * gamma + beta`

__Arguments__


- __x__: 입력 텐서 또는 변수.
- __mean__: 배치의 평균 
- __var__: 배치의 분산
- __beta__: 입력을 중앙에 위치시키는 텐서. 
- __gamma__: 입력 스케일링에 의한 텐서.
- __axis__: Integer, 정규화 시켜야 하는 축.
    (typically the features axis).
- __epsilon__: 퍼지 상수.


__Returns__

텐서.
    
----

### concatenate


```python
keras.backend.concatenate(tensors, axis=-1)
```


지정된 축에따른 텐서의 리스트 연결.


__Arguments__

- __tensors__: 연결 할 텐서의 목록.
- __axis__: 연결 축.

__Returns__

텐서.
    
----

### reshape


```python
keras.backend.reshape(x, shape)
```


텐서를 지정한 형식으로 다시 재정의 합니다. 

__Arguments__

- __x__: 텐서 또는 변수.
- __shape__: 대상이 되는 형식튜플.


__Returns__

텐서.
    
----

### permute_dimensions


```python
keras.backend.permute_dimensions(x, pattern)
```


텐서의 축을 치환합니다.

__Arguments__

- __x__: 텐서 또는 변수.
- __pattern__: `(0, 2, 1)`처럼 인덱스들의 차원의 튜플.


__Returns__

텐서.
    
----

### resize_images


```python
keras.backend.resize_images(x, height_factor, width_factor, data_format, interpolation='nearest')
```


4차원 텐서에 포함된 이미지들을 재조정합니다.

__Arguments__

- __x__: 텐서 또는 변수. to resize.
- __height_factor__: 양의 정수.
- __width_factor__: 양의 정수.
- __data_format__: <sag>string</sag>, `"channels_last"` 또는 `"channels_first"`.
- __interpolation__: <sag>string</sag>, `nearest` 또는 `bilinear` 중 하나.


__Returns__

텐서.

__Raises__

- __ValueError__: `data_format`이면 'channels_last' 또는 'channels_first' 모두 아니다.
    
----

### resize_volumes


```python
keras.backend.resize_volumes(x, depth_factor, height_factor, width_factor, data_format)
```


5차원 텐서에 포함된 볼륨 크기 조정.

__Arguments__

- __x__: 텐서 또는 변수. to resize.
- __depth_factor__: 양의 정수.
- __height_factor__: 양의 정수.
- __width_factor__: 양의 정수.
- __data_format__: <sag>string</sag>, `"channels_last"` 또는 `"channels_first"`.


__Returns__

한 개의 텐서.

__Raises__

- __ValueError__: `data_format`이면 'channels_last' 또는 'channels_first' 모두 아니다.
    
----

### repeat_elements


```python
keras.backend.repeat_elements(x, rep, axis)
```


`np.repeat`처럼 축을따라 텐서의 요소를 반복.

`x` 형식이 `(s1, s2, s3)` 이고, `axis` 이 `1`이면, 출력형식은 (s1, s2 * rep, s3)`입니다.

__Arguments__

- __x__: 텐서 또는 변수.
- __rep__: <sag>integer</sag>, 반복횟수.
- __axis__: 반복 할 축

__Returns__

한 개의 텐서.
    
----

### repeat


```python
keras.backend.repeat(x, n)
```


2차원 텐서를 반복합니다.
만약 x가 (samples, dim)형식이고 'n'이 2라면, 출력값은 형식이 (samples, 2, dim)가 됩니다.


__Arguments__

- __x__: 텐서 또는 변수.
- __n__: <sag>integer</sag>, 반복횟수.

__Returns__

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

__Arguments__

- __start__: 시작 값.
- __stop__: 정지 값.
- __step__: 두 개의 연속적인 값의 차이.
- __dtype__: <sag>Integer</sag> dtype

__Returns__

정수형 텐서.


----

### tile


```python
keras.backend.tile(x, n)
```


x를 n으로 나열하여 생성합니다. 


__Arguments__

- __x__: 텐서 또는 배열.
- __n__: <sag>integer</sag>의 리스트. x의 차원의 갯수와 그 길이가 같다.

__Returns__

나열된 텐서.
    
----

### flatten


```python
keras.backend.flatten(x)
```


텐서를 합쳐서 나열합니다.

__Arguments__

- __x__: 텐서 또는 변수.

__Returns__

텐서를 1차원으로 형식을 재구성하여 나열합니다.
    
----

### batch_flatten


```python
keras.backend.batch_flatten(x)
```



n차원 텐서를 같은 0차원의 2차원 텐서로 변형합니다.
즉, 배치의 각 데이터 샘플을 위 차원의 변형에 맞게 변환합니다.


__Arguments__

- __x__: 텐서 또는 변수.

__Returns__

텐서.
    
----

### expand_dims


```python
keras.backend.expand_dims(x, axis=-1)
```


축의 인덱스값에 1만큼의 차원을 더한다.

__Arguments__

- __x__: 텐서 또는 변수.
- __axis__: 새로운 축을 추가한 위치.

__Returns__


확장한 차원들의 텐서.
    
----

### squeeze


```python
keras.backend.squeeze(x, axis)
```



축의 인덱스 값에 해당하는 텐서를 1차원의 크기만큼 제거합니다.


__Arguments__

- __x__: 텐서 또는 변수.
- __axis__: 없앨 축.

__Returns__



줄어든 차원의 x와 동일한 데이터를 가지는 텐서.
    
----

### temporal_padding


```python
keras.backend.temporal_padding(x, padding=(1, 1))
```


3차원 텐서의 중간차원을 채웁니다.

__Arguments__

- __x__: 텐서 또는 변수.
- __padding__: 2 <sag>integers</sag>의 튜플, 차원 1의 시작과 끝에 얼마나 많은 0을 추가할 지에 대한 수치.

__Returns__

3차원 텐서를 채워 넣습니다.
    
----

### spatial_2d_padding


```python
keras.backend.spatial_2d_padding(x, padding=((1, 1), (1, 1)), data_format=None)
```



4차원 텐서에서 2차원과 3차원을 채워 넣습니다.


__Arguments__

- __x__: 텐서 또는 변수.
- __padding__: 2 튜플들의 튜플, 채워진 패턴.
- __data_format__: <sag>string</sag>, `"channels_last"` 또는 `"channels_first"`.

__Returns__


채워진 4차원 텐서.

__Raises__

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


__Arguments__

- __x__: 텐서 또는 변수.
- __padding__: 3 튜플들의 튜플, 채워진 패턴.
- __data_format__: <sag>string</sag>, `"channels_last"` 또는 `"channels_first"`.


__Returns__


채워진 5차원 텐서.

__Raises__

- __ValueError__: `data_format`이면 'channels_last' 또는 'channels_first' 모두 아니다.


----

### stack


```python
keras.backend.stack(x, axis=0)
```


랭크`R` 텐서의 <sag>list</sag>를 랭크`R + 1` 텐서에 쌓습니다.

__Arguments__

- __x__: 텐서들의 <sag>list</sag>
- __axis__: 텐서를 쌓을 축.

__Returns__

텐서.

__Numpy implementation__


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

__Arguments__

- __indices__: `(batch_size, dim1, dim2, ... dim(n-1))`형식의 n차원 정수형 텐서.
- __num_classes__: <sag>integer</sag>, 클래스들의 갯수.

__Returns__


`(batch_size, dim1, dim2, ... dim(n-1), num_classes)`형식의 입력값의 (n+1)차원의 원핫 표현형식.
    
----

### reverse


```python
keras.backend.reverse(x, axes)
```


지정된 축을 따라 텐서를 반전시킵니다.

__Arguments__

- __x__: 텐서를 반전시킨다.
- __axes__: 축이 반전된, 정수형 또는 반복 가능한 정수.

__Returns__

텐서.

__Numpy implementation__


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

__Arguments__

- __x__: 입력 텐서.
- __start__: 각 축에 따라 슬라이스의 시작 인덱스를 나타내는 텐서 또는 <sag>integer</sag>리스트/튜플 자료형.
- __size__: 각 축을 따라 슬라이스 할 차원의 수를 나타내는 텐서 또는 <sag>integer</sag>리스트/튜플 자료형.


__Returns__

A sliced tensor:
```python
new_x = x[start[0]: start[0] + size[0], ..., start[-1]: start[-1] + size[-1]]
```

__Numpy implementation__


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

__Arguments__

- __x__: 입력 변수. 

__Returns__

넘파이 배열.
    
----

### batch_get_value


```python
keras.backend.batch_get_value(ops)
```


한 가지 이상의 텐서 변수의 값을 반환합니다.

__Arguments__

- __ops__: 실행할 ops 목록.

__Returns__

넘파이 배열 리스트.
    
----

### set_value


```python
keras.backend.set_value(x, value)
```


넘파이 배열에서 변수의 값을 설정합니다. 

__Arguments__

- __x__: 새로운 값으로 설정하는 텐서.
- __value__: 넘파이 배열로 텐서를 설정하는 값.


----

### batch_set_value


```python
keras.backend.batch_set_value(tuples)
```

한번에 밚은 텐서 변수들의 값을 설정합니다.


__Arguments__

- __tuples__: `(tensor, value)` 튜플 리스트, <sag>value</sag>인자는 넘파이 배열이어야 합니다.
    
----

### print_tensor


```python
keras.backend.print_tensor(x, message='')
```



평가시 <sag>message</sag>와 텐서 값을 출력합니다. 

`print_tensor`는 `x`와 동일한 새로운 텐서를 반환합니다. 이 코드는 반드시 다음코드에 사용해야 합니다. 
그렇지 않으면 평가 중 프린트 연산이 고려되지 않습니다. 



__Example__

```python
>>> x = K.print_tensor(x, message="x is: ")
```

__Arguments__

- __x__: 출력 할 텐서.
- __message__: 텐서와 함께 출력 할 메시지.


__Returns__

변경되지 않은 같은 텐서  `x`.
    
----

### function


```python
keras.backend.function(inputs, outputs, updates=None)
```


케라스 함수 인스턴스화하기.

__Arguments__

- __inputs__: 플레이스홀더 텐서의 리스트.
- __outputs__: 출력 텐서의 리스트. 
- __updates__: 업데이트 연산의 리스트.
- __**kwargs__: `tf.Session.run`에 전달되는 값.


__Returns__


넘파이 배열의 값 출력.

__Raises__

- __ValueError__: 유효하지 않은 kwargs 가 전달된 경우.
    
----

### gradients


```python
keras.backend.gradients(loss, variables)
```


변수에 대한 손실의 그라디언트를 반환합니다.

__Arguments__

- __loss__: 최소화시킨 스칼라값 텐서.
- __variables__: 변수들의 리스트.

__Returns__

그라디언트 텐서.
    
----

### stop_gradient


```python
keras.backend.stop_gradient(variables)
```


모든 다른 변수에 대한 0 그라디언트 'variables'를 반환합니다. 

__Arguments__

- __variables__: 또 다른 변수에 대한 상수를 고려한 텐서 또는 텐서의 리스트.

__Returns__

전달받은 인자에 따른 또 다른 변수에 대한 상수 그라디언트를 가진 텐서 또는 텐서의 리스트.



----

### rnn


```python
keras.backend.rnn(step_function, inputs, initial_states, go_backwards=False, mask=None, constants=None, unroll=False, input_length=None)
```


텐서의 시간 차원에 대한 반복.

__Arguments__

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


__Returns__

A tuple, `(last_output, outputs, new_states)`.

last_output: `(samples, ...)` 형식의, rnn의 최근 출력값. 
outputs: `(samples, time, ...)` 형식이 있는 텐서 의 각 `outputs[s, t]`요소는 's'샘플에 대한 't'시간에 대한 단계 함수의 출력요소 입니다. 
new_states: `(samples, ...)`형식의 단계함수로 반환된 최근 상태의 텐서 리스트.


__Raises__

- __ValueError__: 입력 차원이 3보다 작은 경우.
- __ValueError__: `unroll`이  `True`인 경우. 
    입력 시간 단계는 고정이 아님.
- __ValueError__: `mask` 가 존재하면 (not `None`)
    상태는 (`len(states)` == 0).

__Numpy implementation__


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


__Arguments__


- __condition__: 텐서 (<sag>int</sag> or <sag>bool</sag>).
- __then_expression__: 텐서 또는 텐서를 반환하는 호출가능한 값.
- __else_expression__: 텐서 또는 텐서를 반환하는 호출가능한 값.

__Returns__

지정한 텐서.

__Raises__

- __ValueError__: 표현된 랭크보다 더 나은 'condition'의 랭크일 경우, 에러.

__Numpy implementation__


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

__Arguments__


- __x__: 훈련 단계에서 반환하는 것.
    (텐서 또는 호출가능한 텐서).
- __alt__: 그 밖의 것을 반환.
    (텐서 또는 호출가능한 텐서).
- __training__: 학습 단계를 지정한 선택적 스칼라 텐서. 
    (<sag>Python boolean</sag> 또는 <sag>Python integer</sag>)


__Returns__

플래그에 기반한 `x` 또는 `alt`.
`training` 플래그는 기본적으로 `K.learning_phase()`입니다. 
    
----

### in_test_phase


```python
keras.backend.in_test_phase(x, alt, training=None)
```


Selects `x` in test phase, and `alt` otherwise.

Note that `alt` should have the *same shape* as `x`.

__Arguments__

- __x__: What to return in test phase
    (tensor or callable that returns a tensor).
- __alt__: What to return otherwise
    (tensor or callable that returns a tensor).
- __training__: Optional scalar tensor
    (or Python boolean, or Python integer)
    specifying the learning phase.

__Returns__

Either `x` or `alt` based on `K.learning_phase`.
    
----

### relu


```python
keras.backend.relu(x, alpha=0.0, max_value=None, threshold=0.0)
```


Rectified linear unit.

With default values, it returns element-wise `max(x, 0)`.

Otherwise, it follows:
`f(x) = max_value` for `x >= max_value`,
`f(x) = x` for `threshold <= x < max_value`,
`f(x) = alpha * (x - threshold)` otherwise.

__Arguments__

- __x__: A tensor or variable.
- __alpha__: A scalar, slope of negative section (default=`0.`).
- __max_value__: float. Saturation threshold.
- __threshold__: float. Threshold value for thresholded activation.

__Returns__

A tensor.

__Numpy implementation__


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


Exponential linear unit.

__Arguments__

- __x__: A tensor or variable to compute the activation function for.
- __alpha__: A scalar, slope of negative section.

__Returns__

A tensor.

__Numpy implementation__


```python
def elu(x, alpha=1.):
    return x * (x > 0) + alpha * (np.exp(x) - 1.) * (x < 0)
```


----

### softmax


```python
keras.backend.softmax(x, axis=-1)
```


Softmax of a tensor.

__Arguments__

- __x__: A tensor or variable.
- __axis__: The dimension softmax would be performed on.
    The default is -1 which indicates the last dimension.

__Returns__

A tensor.

__Numpy implementation__


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


Softplus of a tensor.

__Arguments__

- __x__: A tensor or variable.

__Returns__

A tensor.

__Numpy implementation__


```python
def softplus(x):
    return np.log(1. + np.exp(x))
```


----

### softsign


```python
keras.backend.softsign(x)
```


Softsign of a tensor.

__Arguments__

- __x__: A tensor or variable.

__Returns__

A tensor.

__Numpy implementation__


```python
def softsign(x):
    return x / (1 + np.abs(x))
```


----

### categorical_crossentropy


```python
keras.backend.categorical_crossentropy(target, output, from_logits=False, axis=-1)
```


Categorical crossentropy between an output tensor and a target tensor.

__Arguments__

- __target__: A tensor of the same shape as `output`.
- __output__: A tensor resulting from a softmax
    (unless `from_logits` is True, in which
    case `output` is expected to be the logits).
- __from_logits__: Boolean, whether `output` is the
    result of a softmax, or is a tensor of logits.
- __axis__: Int specifying the channels axis. `axis=-1`
    corresponds to data format `channels_last`,
    and `axis=1` corresponds to data format
    `channels_first`.

__Returns__

Output tensor.

__Raises__

- __ValueError__: if `axis` is neither -1 nor one of
    the axes of `output`.
    
----

### sparse_categorical_crossentropy


```python
keras.backend.sparse_categorical_crossentropy(target, output, from_logits=False, axis=-1)
```


Categorical crossentropy with integer targets.

__Arguments__

- __target__: An integer tensor.
- __output__: A tensor resulting from a softmax
    (unless `from_logits` is True, in which
    case `output` is expected to be the logits).
- __from_logits__: Boolean, whether `output` is the
    result of a softmax, or is a tensor of logits.
- __axis__: Int specifying the channels axis. `axis=-1`
    corresponds to data format `channels_last`,
    and `axis=1` corresponds to data format
    `channels_first`.

__Returns__

Output tensor.

__Raises__

- __ValueError__: if `axis` is neither -1 nor one of
    the axes of `output`.
    
----

### binary_crossentropy


```python
keras.backend.binary_crossentropy(target, output, from_logits=False)
```


Binary crossentropy between an output tensor and a target tensor.

__Arguments__

- __target__: A tensor with the same shape as `output`.
- __output__: A tensor.
- __from_logits__: Whether `output` is expected to be a logits tensor.
    By default, we consider that `output`
    encodes a probability distribution.

__Returns__

A tensor.
    
----

### sigmoid


```python
keras.backend.sigmoid(x)
```


Element-wise sigmoid.

__Arguments__

- __x__: A tensor or variable.

__Returns__

A tensor.

__Numpy implementation__


```python
def sigmoid(x):
    return 1. / (1. + np.exp(-x))
```


----

### hard_sigmoid


```python
keras.backend.hard_sigmoid(x)
```


Segment-wise linear approximation of sigmoid.

Faster than sigmoid.
Returns `0.` if `x < -2.5`, `1.` if `x > 2.5`.
In `-2.5 <= x <= 2.5`, returns `0.2 * x + 0.5`.

__Arguments__

- __x__: A tensor or variable.

__Returns__

A tensor.

__Numpy implementation__


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


Element-wise tanh.

__Arguments__

- __x__: A tensor or variable.

__Returns__

A tensor.

__Numpy implementation__


```python
def tanh(x):
    return np.tanh(x)
```


----

### dropout


```python
keras.backend.dropout(x, level, noise_shape=None, seed=None)
```


Sets entries in `x` to zero at random, while scaling the entire tensor.

__Arguments__

- __x__: tensor
- __level__: fraction of the entries in the tensor
    that will be set to 0.
- __noise_shape__: shape for randomly generated keep/drop flags,
    must be broadcastable to the shape of `x`
- __seed__: random seed to ensure determinism.

__Returns__

A tensor.
__Numpy implementation__


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


Normalizes a tensor wrt the L2 norm alongside the specified axis.

__Arguments__

- __x__: Tensor or variable.
- __axis__: axis along which to perform normalization.

__Returns__

A tensor.

__Numpy implementation__


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


Returns whether the `targets` are in the top `k` `predictions`.

__Arguments__

- __predictions__: A tensor of shape `(batch_size, classes)` and type `float32`.
- __targets__: A 1D tensor of length `batch_size` and type `int32` or `int64`.
- __k__: An `int`, number of top elements to consider.

__Returns__

A 1D tensor of length `batch_size` and type `bool`.
`output[i]` is `True` if `predictions[i, targets[i]]` is within top-`k`
values of `predictions[i]`.
    
----

### conv1d


```python
keras.backend.conv1d(x, kernel, strides=1, padding='valid', data_format=None, dilation_rate=1)
```


1D convolution.

__Arguments__

- __x__: Tensor or variable.
- __kernel__: kernel tensor.
- __strides__: stride integer.
- __padding__: string, `"same"`, `"causal"` or `"valid"`.
- __data_format__: string, `"channels_last"` or `"channels_first"`.
- __dilation_rate__: integer dilate rate.

__Returns__

A tensor, result of 1D convolution.

__Raises__

- __ValueError__: If `data_format` is neither
    `"channels_last"` nor `"channels_first"`.
    
----

### conv2d


```python
keras.backend.conv2d(x, kernel, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1))
```


2D convolution.

__Arguments__

- __x__: Tensor or variable.
- __kernel__: kernel tensor.
- __strides__: strides tuple.
- __padding__: string, `"same"` or `"valid"`.
- __data_format__: string, `"channels_last"` or `"channels_first"`.
    Whether to use Theano or TensorFlow/CNTK data format
    for inputs/kernels/outputs.
- __dilation_rate__: tuple of 2 integers.

__Returns__

A tensor, result of 2D convolution.

__Raises__

- __ValueError__: If `data_format` is neither
    `"channels_last"` nor `"channels_first"`.
    
----

### conv2d_transpose


```python
keras.backend.conv2d_transpose(x, kernel, output_shape, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1))
```


2D deconvolution (i.e. transposed convolution).

__Arguments__

- __x__: Tensor or variable.
- __kernel__: kernel tensor.
- __output_shape__: 1D int tensor for the output shape.
- __strides__: strides tuple.
- __padding__: string, `"same"` or `"valid"`.
- __data_format__: string, `"channels_last"` or `"channels_first"`.
    Whether to use Theano or TensorFlow/CNTK data format
    for inputs/kernels/outputs.
- __dilation_rate__: tuple of 2 integers.

__Returns__

A tensor, result of transposed 2D convolution.

__Raises__

- __ValueError__: If `data_format` is neither
    `"channels_last"` nor `"channels_first"`.
    
----

### separable_conv1d


```python
keras.backend.separable_conv1d(x, depthwise_kernel, pointwise_kernel, strides=1, padding='valid', data_format=None, dilation_rate=1)
```


1D convolution with separable filters.

__Arguments__

- __x__: input tensor
- __depthwise_kernel__: convolution kernel for the depthwise convolution.
- __pointwise_kernel__: kernel for the 1x1 convolution.
- __strides__: stride integer.
- __padding__: string, `"same"` or `"valid"`.
- __data_format__: string, `"channels_last"` or `"channels_first"`.
- __dilation_rate__: integer dilation rate.

__Returns__

Output tensor.

__Raises__

- __ValueError__: If `data_format` is neither
    `"channels_last"` nor `"channels_first"`.
    
----

### separable_conv2d


```python
keras.backend.separable_conv2d(x, depthwise_kernel, pointwise_kernel, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1))
```


2D convolution with separable filters.

__Arguments__

- __x__: input tensor
- __depthwise_kernel__: convolution kernel for the depthwise convolution.
- __pointwise_kernel__: kernel for the 1x1 convolution.
- __strides__: strides tuple (length 2).
- __padding__: string, `"same"` or `"valid"`.
- __data_format__: string, `"channels_last"` or `"channels_first"`.
- __dilation_rate__: tuple of integers,
    dilation rates for the separable convolution.

__Returns__

Output tensor.

__Raises__

- __ValueError__: If `data_format` is neither
    `"channels_last"` nor `"channels_first"`.
    
----

### depthwise_conv2d


```python
keras.backend.depthwise_conv2d(x, depthwise_kernel, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1))
```


2D convolution with separable filters.

__Arguments__

- __x__: input tensor
- __depthwise_kernel__: convolution kernel for the depthwise convolution.
- __strides__: strides tuple (length 2).
- __padding__: string, `"same"` or `"valid"`.
- __data_format__: string, `"channels_last"` or `"channels_first"`.
- __dilation_rate__: tuple of integers,
    dilation rates for the separable convolution.

__Returns__

Output tensor.

__Raises__

- __ValueError__: If `data_format` is neither
    `"channels_last"` nor `"channels_first"`.
    
----

### conv3d


```python
keras.backend.conv3d(x, kernel, strides=(1, 1, 1), padding='valid', data_format=None, dilation_rate=(1, 1, 1))
```


3D convolution.

__Arguments__

- __x__: Tensor or variable.
- __kernel__: kernel tensor.
- __strides__: strides tuple.
- __padding__: string, `"same"` or `"valid"`.
- __data_format__: string, `"channels_last"` or `"channels_first"`.
    Whether to use Theano or TensorFlow/CNTK data format
    for inputs/kernels/outputs.
- __dilation_rate__: tuple of 3 integers.

__Returns__

A tensor, result of 3D convolution.

__Raises__

- __ValueError__: If `data_format` is neither
    `"channels_last"` nor `"channels_first"`.
    
----

### conv3d_transpose


```python
keras.backend.conv3d_transpose(x, kernel, output_shape, strides=(1, 1, 1), padding='valid', data_format=None)
```


3D deconvolution (i.e. transposed convolution).

__Arguments__

- __x__: input tensor.
- __kernel__: kernel tensor.
- __output_shape__: 1D int tensor for the output shape.
- __strides__: strides tuple.
- __padding__: string, "same" or "valid".
- __data_format__: string, `"channels_last"` or `"channels_first"`.
    Whether to use Theano or TensorFlow/CNTK data format
    for inputs/kernels/outputs.

__Returns__

A tensor, result of transposed 3D convolution.

__Raises__

- __ValueError__: If `data_format` is neither
    `"channels_last"` nor `"channels_first"`.
    
----

### pool2d


```python
keras.backend.pool2d(x, pool_size, strides=(1, 1), padding='valid', data_format=None, pool_mode='max')
```


2D Pooling.

__Arguments__

- __x__: Tensor or variable.
- __pool_size__: tuple of 2 integers.
- __strides__: tuple of 2 integers.
- __padding__: string, `"same"` or `"valid"`.
- __data_format__: string, `"channels_last"` or `"channels_first"`.
- __pool_mode__: string, `"max"` or `"avg"`.

__Returns__

A tensor, result of 2D pooling.

__Raises__

- __ValueError__: if `data_format` is

neither `"channels_last"` or `"channels_first"`.

- __ValueError__: if `pool_mode` is neither `"max"` or `"avg"`.
    
----

### pool3d


```python
keras.backend.pool3d(x, pool_size, strides=(1, 1, 1), padding='valid', data_format=None, pool_mode='max')
```


3D Pooling.

__Arguments__

- __x__: Tensor or variable.
- __pool_size__: tuple of 3 integers.
- __strides__: tuple of 3 integers.
- __padding__: string, `"same"` or `"valid"`.
- __data_format__: string, `"channels_last"` or `"channels_first"`.
- __pool_mode__: string, `"max"` or `"avg"`.

__Returns__

A tensor, result of 3D pooling.

__Raises__

- __ValueError__: if `data_format` is

neither `"channels_last"` or `"channels_first"`.

- __ValueError__: if `pool_mode` is neither `"max"` or `"avg"`.
    
----

### bias_add


```python
keras.backend.bias_add(x, bias, data_format=None)
```


Adds a bias vector to a tensor.

__Arguments__

- __x__: Tensor or variable.
- __bias__: Bias tensor to add.
- __data_format__: string, `"channels_last"` or `"channels_first"`.

__Returns__

Output tensor.

__Raises__

ValueError: In one of the two cases below:
1. invalid `data_format` argument.
2. invalid bias shape.
the bias should be either a vector or
a tensor with ndim(x) - 1 dimension
__Numpy implementation__


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


Returns a tensor with normal distribution of values.

__Arguments__

- __shape__: A tuple of integers, the shape of tensor to create.
- __mean__: A float, mean of the normal distribution to draw samples.
- __stddev__: A float, standard deviation of the normal distribution
    to draw samples.
- __dtype__: String, dtype of returned tensor.
- __seed__: Integer, random seed.

__Returns__

A tensor.
    
----

### random_uniform


```python
keras.backend.random_uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None)
```


Returns a tensor with uniform distribution of values.

__Arguments__

- __shape__: A tuple of integers, the shape of tensor to create.
- __minval__: A float, lower boundary of the uniform distribution
    to draw samples.
- __maxval__: A float, upper boundary of the uniform distribution
    to draw samples.
- __dtype__: String, dtype of returned tensor.
- __seed__: Integer, random seed.

__Returns__

A tensor.
    
----

### random_binomial


```python
keras.backend.random_binomial(shape, p=0.0, dtype=None, seed=None)
```


Returns a tensor with random binomial distribution of values.

__Arguments__

- __shape__: A tuple of integers, the shape of tensor to create.
- __p__: A float, `0. <= p <= 1`, probability of binomial distribution.
- __dtype__: String, dtype of returned tensor.
- __seed__: Integer, random seed.

__Returns__

A tensor.
    
----

### truncated_normal


```python
keras.backend.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None)
```


Returns a tensor with truncated random normal distribution of values.

The generated values follow a normal distribution
with specified mean and standard deviation,
except that values whose magnitude is more than
two standard deviations from the mean are dropped and re-picked.

__Arguments__

- __shape__: A tuple of integers, the shape of tensor to create.
- __mean__: Mean of the values.
- __stddev__: Standard deviation of the values.
- __dtype__: String, dtype of returned tensor.
- __seed__: Integer, random seed.

__Returns__

A tensor.
    
----

### ctc_label_dense_to_sparse


```python
keras.backend.ctc_label_dense_to_sparse(labels, label_lengths)
```


Converts CTC labels from dense to sparse.

__Arguments__

- __labels__: dense CTC labels.
- __label_lengths__: length of the labels.

__Returns__

A sparse tensor representation of the labels.
    
----

### ctc_batch_cost


```python
keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
```


Runs CTC loss algorithm on each batch element.

__Arguments__

- __y_true__: tensor `(samples, max_string_length)`
    containing the truth labels.
- __y_pred__: tensor `(samples, time_steps, num_categories)`
    containing the prediction, or output of the softmax.
- __input_length__: tensor `(samples, 1)` containing the sequence length for
    each batch item in `y_pred`.
- __label_length__: tensor `(samples, 1)` containing the sequence length for
    each batch item in `y_true`.

__Returns__

Tensor with shape (samples,1) containing the
    CTC loss of each element.
    
----

### ctc_decode


```python
keras.backend.ctc_decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1)
```


Decodes the output of a softmax.

Can use either greedy search (also known as best path)
or a constrained dictionary search.

__Arguments__

- __y_pred__: tensor `(samples, time_steps, num_categories)`
    containing the prediction, or output of the softmax.
- __input_length__: tensor `(samples, )` containing the sequence length for
    each batch item in `y_pred`.
- __greedy__: perform much faster best-path search if `true`.
    This does not use a dictionary.
- __beam_width__: if `greedy` is `false`: a beam search decoder will be used
    with a beam of this width.
- __top_paths__: if `greedy` is `false`,
    how many of the most probable paths will be returned.

__Returns__

- __Tuple__:
    List: if `greedy` is `true`, returns a list of one element that
        contains the decoded sequence.
        If `false`, returns the `top_paths` most probable
        decoded sequences.
        Important: blank labels are returned as `-1`.
    Tensor `(top_paths, )` that contains
        the log probability of each decoded sequence.
    
----

### map_fn


```python
keras.backend.map_fn(fn, elems, name=None, dtype=None)
```


Map the function fn over the elements elems and return the outputs.

__Arguments__

- __fn__: Callable that will be called upon each element in elems
- __elems__: tensor
- __name__: A string name for the map node in the graph
- __dtype__: Output data type.

__Returns__

Tensor with dtype `dtype`.
    
----

### foldl


```python
keras.backend.foldl(fn, elems, initializer=None, name=None)
```


Reduce elems using fn to combine them from left to right.

__Arguments__

- __fn__: Callable that will be called upon each element in elems and an
    accumulator, for instance `lambda acc, x: acc + x`
- __elems__: tensor
- __initializer__: The first value used (`elems[0]` in case of None)
- __name__: A string name for the foldl node in the graph

__Returns__

Tensor with same type and shape as `initializer`.
    
----

### foldr


```python
keras.backend.foldr(fn, elems, initializer=None, name=None)
```


Reduce elems using fn to combine them from right to left.

__Arguments__

- __fn__: Callable that will be called upon each element in elems and an
    accumulator, for instance `lambda acc, x: acc + x`
- __elems__: tensor
- __initializer__: The first value used (`elems[-1]` in case of None)
- __name__: A string name for the foldr node in the graph

__Returns__

Tensor with same type and shape as `initializer`.
    
----

### local_conv1d


```python
keras.backend.local_conv1d(inputs, kernel, kernel_size, strides, data_format=None)
```


Apply 1D conv with un-shared weights.

__Arguments__

- __inputs__: 3D tensor with shape: (batch_size, steps, input_dim)
- __kernel__: the unshared weight for convolution,
        with shape (output_length, feature_dim, filters)
- __kernel_size__: a tuple of a single integer,
             specifying the length of the 1D convolution window
- __strides__: a tuple of a single integer,
         specifying the stride length of the convolution
- __data_format__: the data format, channels_first or channels_last

__Returns__

the tensor after 1d conv with un-shared weights,
with shape (batch_size, output_length, filters)

__Raises__

- __ValueError__: If `data_format` is neither
    `"channels_last"` nor `"channels_first"`.
    
----

### local_conv2d


```python
keras.backend.local_conv2d(inputs, kernel, kernel_size, strides, output_shape, data_format=None)
```


Apply 2D conv with un-shared weights.

__Arguments__

- __inputs__: 4D tensor with shape:
        (batch_size, filters, new_rows, new_cols)
        if data_format='channels_first'
        or 4D tensor with shape:
        (batch_size, new_rows, new_cols, filters)
        if data_format='channels_last'.
- __kernel__: the unshared weight for convolution,
        with shape (output_items, feature_dim, filters)
- __kernel_size__: a tuple of 2 integers, specifying the
             width and height of the 2D convolution window.
- __strides__: a tuple of 2 integers, specifying the strides
         of the convolution along the width and height.
- __output_shape__: a tuple with (output_row, output_col)
- __data_format__: the data format, channels_first or channels_last

__Returns__

A 4d tensor with shape:
(batch_size, filters, new_rows, new_cols)
if data_format='channels_first'
or 4D tensor with shape:
(batch_size, new_rows, new_cols, filters)
if data_format='channels_last'.

__Raises__

- __ValueError__: if `data_format` is neither
            `channels_last` or `channels_first`.
    
----

### backend


```python
keras.backend.backend()
```


Publicly accessible method
for determining the current backend.

__Returns__

String, the name of the backend Keras is currently using.

__Example__

```python
>>> keras.backend.backend()
'tensorflow'
```
    






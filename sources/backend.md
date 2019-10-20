# Keras 백엔드

## "백앤드"는 무엇인가요? 

Keras는 딥러닝 모델을 개발하기 위한 고수준의 구성요성 요소를 제공하는 모델 레벨의 라이브러리입니다. Keras는 텐서 곱셈, 합성곱 등의 저수준의 연산을 제공하지 않습니다. 대신 Keras의 "백엔드 엔진" 역할을 하는 특수하고 잘 최적화 된 텐서 라이브러리에 의존합니다. 하나의 단일 텐서 라이브러리를 선택하고 Keras 구현을 해당 라이브러리에 묶는 대신, Keras는 모듈 방식으로 문제를 처리하여 여러 다른 백엔드 엔진들을 Keras에 매끄럽게 연결할 수 있게 합니다.

현재 Keras는 **TensorFlow**, **Theano**, 그리고 **CNTK**의 세 가지 백엔드를 지원합니다.

- [TensorFlow](http://www.tensorflow.org/) is an open-source symbolic tensor manipulation framework developed by Google.
- [Theano](http://deeplearning.net/software/theano/) is an open-source symbolic tensor manipulation framework developed by LISA Lab at Université de Montréal.
- [CNTK](https://www.microsoft.com/en-us/cognitive-toolkit/) is an open-source toolkit for deep learning developed by Microsoft.

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

또, 아래와 같이 환경 변수 `KERAS_BACKEND`를 정의해 설정 파일에 정의된 것을 재정의 할 수도 있습니다. :

```bash
KERAS_BACKEND=tensorflow python -c "from keras import backend"
Using TensorFlow backend.
```

Keras에서는 `"tensorflow"`, `"theano"` 그리고 `"cntk"`외에도 사용자가 지정한 임의의 백엔드를 로드하는 것이 가능합니다.
케라스는 외부 백엔드를 사용할 수 있으며, 'keras.json' 구성 파일과 "backend" 설정을 바꾸어 수행할 수 있습니다. 
만약 `my_module`이라는 이름의 Python 모듈을 외부 백엔드로 사용하고자 한다면,
`keras.json` 설정 파일의 `"backend"` 변수 값을 아래와 같이 바꿔주어야 합니다.  

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

* `image_data_format`: 문자열 타입<sub>string</sub>, <sub>channels_last</sub> 또는 <sub>channels_first</sub>로 케라스가 따르는 데이터 형식에 대한 규칙을 지정할 수 있습니다. (<sub>keras.backend.image_data_format()</sub>을 반환합니다..)
  - 2D 데이터의 경우 (예, 이미지), <sub>channels_first</sub>은 (channels, rows, cols)을 가정하고, <sub>channels_last</sub>은 (rows, cols, channels)을 가정합니다. 
  - 3D 데이터의 경우, <sub>channels_first</sub>은 (channels, conv_dim1, conv_dim2, conv_dim3)</sub>을 가정하고, <sub>channels_last</sub>은 (conv_dim1, conv_dim2, conv_dim3, channels)을 가정합니다. 
* `epsilon`: 실수형 타입<sub>Float</sub>, 일부 연산에서 0으로 나눠지지 않게 하기위한 퍼지상수.
* `floatx`: 문자열 타입<sub>string</sub>, <sub>float16</sub>, <sub>float32</sub>, or <sub>float64</sub>. 기본 부동소수점 정밀도.
* `backend`: 문자열 타입<sub>string</sub>, `"tensorflow"`, `"theano"`, 또는 `"cntk"`.



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


수치 식에 사용되는 fuzz factor의 값을 반환합니다.

__Returns__

실수형 타입.

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

- __e__: 실수형. 엡실론의 새로운 값.

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


기본 실수형 타입을 문자열로 반환합니다.
(e.g. 'float16', 'float32', 'float64').


__Returns__


현재 기본 실수형 타입이 문자열로 반환됩니다. 

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


디폴트 float 타입을 설정합니다.

__Arguments__

- __floatx__: String, 'float16', 'float32', or 'float64'.

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


NumPy 배열을 Keras의 디폴트 float 타입으로 변환합니다.

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


기본 이미지 데이터 형태의 규칙을 반환합니다.

__Returns__

<sag>channels_first</sag> 또는 <sag>channels_last</sag>를 문자열 타입으로 반환합니다.

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


데이터 형식 규칙에 대한 값을 설정합니다. 

__Arguments__

- __data_format__:  <sag>channels_first</sag> 또는 <sag>channels_last</sag>를 문자열 타입으로 반환합니다.

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


기본 그래프의 uid 값을 가져옵니다.

__Arguments__

- __prefix__: 그래프의 선택적 접두사입니다.

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

- __value__: 파이썬 불리언 타입.
    
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


희소

__Arguments__

- __tensor__: <sag>sparse</sag> 텐서일 수도 있는 

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

- __value__: A constant value (or list)
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

- __x__: 한개의 후보 플레이스홀더.

__Returns__

불리언 값.
    
----

### shape


```python
keras.backend.shape(x)
```

텐서 또는 변수의 <sag>symbolic shape</sag>을 반환합니다.

__Arguments__

- __x__: 한 개의 텐서 또는 변수. 

__Returns__

텐서 그 자체인 <sag>symbolic shape</sag>

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


<sag>integers</sag>(또는 <sag>None</sag>)의 튜플

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

하나의 넘파이 배열.

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

- __size__: Integer, 행과 열의 수. 
- __dtype__: String, 반환된 케라스 변수의 데이터 타입. 
- __name__: String,  반환된 케라스 변수의 이름. 

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
- __dtype__: String, 반환된 케라스 변수의 dtype.
     None uses the dtype of x.
- __name__: String, 생성할 변수의 이름. 
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

또 다른 텐서와 동일한 모양의 올인원 변수를 인스턴스화합니다
Instantiates an all-ones variable of the same shape as another tensor.

__Arguments__

- __x__: Keras variable or tensor.
- __dtype__: String, dtype of returned Keras variable.
     None uses the dtype of x.
- __name__: String, name for the variable to create.

__Returns__

A Keras variable with the shape of x filled with ones.

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
- __seed__: <sag>integer</sag>, random seed.

__Returns__


샘플들에서 가져온 케라스 변수.

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
- __seed__: <sag>integer</sag>, random seed.


균등 분포에서 가져온 값의 변수를 인스턴스화 합니다. 

__Arguments__


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

- __x__: Keras variable or tensor.

__Returns__

Integer, the number of elements in `x`, i.e., the product of the
array's static dimensions.

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

- __x__: Keras tensor (or variable).
- __dtype__: String, either (`'float16'`, `'float32'`, or `'float64'`).

__Returns__

Keras tensor with dtype `dtype`.

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


Update the value of `x` to `new_x`.

__Arguments__

- __x__: A `Variable`.
- __new_x__: A tensor of same shape as `x`.

__Returns__

The variable `x` updated.
    
----

### update_add


```python
keras.backend.update_add(x, increment)
```


Update the value of `x` by adding `increment`.

__Arguments__

- __x__: A `Variable`.
- __increment__: A tensor of same shape as `x`.

__Returns__

The variable `x` updated.
    
----

### update_sub


```python
keras.backend.update_sub(x, decrement)
```


Update the value of `x` by subtracting `decrement`.

__Arguments__

- __x__: A `Variable`.
- __decrement__: A tensor of same shape as `x`.

__Returns__

The variable `x` updated.
    
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


Multiplies 2 tensors (and/or variables) and returns a *tensor*.

When attempting to multiply a nD tensor
with a nD tensor, it reproduces the Theano behavior.
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

`batch_dot` is used to compute dot product of `x` and `y` when
`x` and `y` are data in batches, i.e. in a shape of
`(batch_size, :)`.
`batch_dot` results in a tensor or variable with less dimensions
than the input. If the number of dimensions is reduced to 1,
we use `expand_dims` to make sure that ndim is at least 2.

__Arguments__

- __x__: Keras tensor or variable with `ndim >= 2`.
- __y__: Keras tensor or variable with `ndim >= 2`.
- __axes__: int or tuple(int, int). Target dimensions to be reduced.

__Returns__

A tensor with shape equal to the concatenation of `x`'s shape
(less the dimension that was summed over) and `y`'s shape
(less the batch dimension and the dimension that was summed over).
If the final rank is 1, we reshape it to `(batch_size, 1)`.

__Examples__

Assume `x = [[1, 2], [3, 4]]` and `y = [[5, 6], [7, 8]]`
`batch_dot(x, y, axes=1) = [[17], [53]]` which is the main diagonal
of `x.dot(y.T)`, although we never have to calculate the off-diagonal
elements.

Pseudocode:
```
inner_products = []
for xi, yi in zip(x, y):
    inner_products.append(xi.dot(yi))
result = stack(inner_products)
```

Shape inference:
Let `x`'s shape be `(100, 20)` and `y`'s shape be `(100, 30, 20)`.
If `axes` is (1, 2), to find the output shape of resultant tensor,
loop through each dimension in `x`'s shape and `y`'s shape:

* `x.shape[0]` : 100 : append to output shape
* `x.shape[1]` : 20 : do not append to output shape,
dimension 1 of `x` has been summed over. (`dot_axes[0]` = 1)
* `y.shape[0]` : 100 : do not append to output shape,
always ignore first dimension of `y`
* `y.shape[1]` : 30 : append to output shape
* `y.shape[2]` : 20 : do not append to output shape,
dimension 2 of `y` has been summed over. (`dot_axes[1]` = 2)
`output_shape` = `(100, 30)`

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


텐서를 트렌스포즈한 후 반환합니다. 

__Arguments__

- __x__: Tensor or variable.

__Returns__

A tensor.

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


텐서 `reference`에서 `indices` 요소를 검색합니다.

__Arguments__

- __reference__: 텐서.
- __indices__: An integer tensor of indices.

__Returns__

A tensor of same type as `reference`.

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


Maximum value in a tensor.

__Arguments__

- __x__: A tensor or variable.
- __axis__: An integer or list of integers in [-rank(x), rank(x)),
    the axes to find maximum values. If `None` (default), finds the
    maximum over all dimensions.
- __keepdims__: A boolean, whether to keep the dimensions or not.
    If `keepdims` is `False`, the rank of the tensor is reduced
    by 1. If `keepdims` is `True`,
    the reduced dimension is retained with length 1.

__Returns__

A tensor with maximum values of `x`.

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


Minimum value in a tensor.

__Arguments__

- __x__: A tensor or variable.
- __axis__: An integer or list of integers in [-rank(x), rank(x)),
    the axes to find minimum values. If `None` (default), finds the
    minimum over all dimensions.
- __keepdims__: A boolean, whether to keep the dimensions or not.
    If `keepdims` is `False`, the rank of the tensor is reduced
    by 1. If `keepdims` is `True`,
    the reduced dimension is retained with length 1.

__Returns__

A tensor with miminum values of `x`.

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


Sum of the values in a tensor, alongside the specified axis.
지정된 축에따른 텐서의 값들의 합.

__Arguments__

- __x__: A tensor or variable.
- __axis__: An integer or list of integers in [-rank(x), rank(x)),
    the axes to sum over. If `None` (default), sums over all
    dimensions.
- __keepdims__: A boolean, whether to keep the dimensions or not.
    If `keepdims` is `False`, the rank of the tensor is reduced
    by 1. If `keepdims` is `True`,
    the reduced dimension is retained with length 1.

__Returns__

'x'의 합을 가진 텐서 

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

- __x__: 텐서 또는 변수. 
- __axis__: An integer or list of integers in [-rank(x), rank(x)) 범위 내 
    <sag>integers</sag>의 리스트 또는 <sag>integers</sag>로서, 곱을 계산한 축. 
    만약 <sag>None</sag>이라면 모든 차원에 대해 곱을 계산합니다. 
- __keepdims__: <sag>boolean</sag>, 차원이 유지되고 있는지 아닌지에 대한 진리값. 
    만약 `keepdims` 가 <sag>False</sag>라면, 텐서의 랭크가 1만큼 감소합니다. 
    만약 `keepdims` 가 <sag>True</sag>라면, 줄어든 차원이 길이 1만큼 유지됩니다. 

__Returns__


'x'의 요소들의 곱에대한 텐서

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

A tensor with the standard deviation of elements of `x`.
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

A uint8 tensor (0s and 1s).
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

- __x__: Tensor or variable.
- __axis__: [-rank(x), rank(x)) 범위의  <sag>integer</sag>타입 리스트 또는 <sag>integer</sag>
    <sag>None</sag> (default)이면 계산. 모든 차원에 대한 평균을 계산합니다. 
- __keepdims__: 감소한 축을 브로드캐스트 하는지 드롭하는지에 대한 여부.


__Returns__

A uint8 tensor (0s and 1s).
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


Returns the index of the maximum value along an axis.

__Arguments__

- __x__: Tensor or variable.
- __axis__: axis along which to perform the reduction.

__Returns__

A tensor.
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


Returns the index of the minimum value along an axis.

__Arguments__

- __x__: Tensor or variable.
- __axis__: axis along which to perform the reduction.

__Returns__

A tensor.
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


Element-wise square.

__Arguments__

- __x__: Tensor or variable.

__Returns__

A tensor.
    
----

### abs


```python
keras.backend.abs(x)
```


Element-wise absolute value.

__Arguments__

- __x__: Tensor or variable.

__Returns__

A tensor.
    
----

### sqrt


```python
keras.backend.sqrt(x)
```


Element-wise square root.

__Arguments__

- __x__: Tensor or variable.

__Returns__

A tensor.
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


Element-wise log.

__Arguments__

- __x__: Tensor or variable.

__Returns__

A tensor.
    
----

### logsumexp


```python
keras.backend.logsumexp(x, axis=None, keepdims=False)
```


Computes log(sum(exp(elements across dimensions of a tensor))).

This function is more numerically stable than log(sum(exp(x))).
It avoids overflows caused by taking the exp of large inputs and
underflows caused by taking the log of small inputs.

__Arguments__

- __x__: A tensor or variable.
- __axis__: axis: An integer or list of integers in [-rank(x), rank(x)),
    the axes to compute the logsumexp. If `None` (default), computes
    the logsumexp over all dimensions.
- __keepdims__: A boolean, whether to keep the dimensions or not.
    If `keepdims` is `False`, the rank of the tensor is reduced
    by 1. If `keepdims` is `True`, the reduced dimension is
    retained with length 1.

__Returns__

The reduced tensor.
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


Element-wise rounding to the closest integer.

In case of tie, the rounding mode used is "half to even".

__Arguments__

- __x__: Tensor or variable.

__Returns__

A tensor.
    
----

### sign


```python
keras.backend.sign(x)
```


Element-wise sign.

__Arguments__

- __x__: Tensor or variable.

__Returns__

A tensor.
    
----

### pow


```python
keras.backend.pow(x, a)
```


Element-wise exponentiation.

__Arguments__

- __x__: Tensor or variable.
- __a__: Python integer.

__Returns__

A tensor.
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


Element-wise value clipping.

__Arguments__

- __x__: Tensor or variable.
- __min_value__: Python float, integer or tensor.
- __max_value__: Python float, integer or tensor.

__Returns__

A tensor.
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


Element-wise equality between two tensors.

__Arguments__

- __x__: Tensor or variable.
- __y__: Tensor or variable.

__Returns__
ㅇ
A bool tensor.

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


Element-wise inequality between two tensors.

__Arguments__

- __x__: 텐서 또는 변수.
- __y__: 텐서 또는 변수.

__Returns__

A bool tensor.

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


Element-wise truth value of (x > y).

__Arguments__

- __x__: 텐서 또는 변수.
- __y__: 텐서 또는 변수.

__Returns__

A bool tensor.

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


Element-wise truth value of (x >= y).

__Arguments__

- __x__: 텐서 또는 변수.
- __y__: 텐서 또는 변수.

__Returns__

A bool tensor.

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


Element-wise truth value of (x < y).

__Arguments__

- __x__: 텐서 또는 변수.
- __y__: 텐서 또는 변수.

__Returns__

A bool tensor.

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


Element-wise truth value of (x <= y).

__Arguments__

- __x__: 텐서 또는 변수.
- __y__: 텐서 또는 변수.

__Returns__

A bool tensor.

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


Element-wise maximum of two tensors.

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


Element-wise minimum of two tensors.

__Arguments__

- __x__: 텐서 또는 변수.
- __y__: 텐서 또는 변수.

__Returns__

한 개의 텐서.

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


Computes sin of x element-wise.

__Arguments__

- __x__: 텐서 또는 변수.

__Returns__

한 개의 텐서.
    
----

### cos


```python
keras.backend.cos(x)
```


Computes cos of x element-wise.

__Arguments__

- __x__: 텐서 또는 변수.

__Returns__

한 개의 텐서.
    
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

A tuple length of 3, `(normalized_tensor, mean, variance)`.
    
----

### batch_normalization


```python
keras.backend.batch_normalization(x, mean, var, beta, gamma, axis=-1, epsilon=0.001)
```


Applies batch normalization on x given mean, var, beta and gamma.

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

한 개의 텐서.
    
----

### concatenate


```python
keras.backend.concatenate(tensors, axis=-1)
```


Concatenates a list of tensors alongside the specified axis.

__Arguments__

- __tensors__: list of tensors to concatenate.
- __axis__: concatenation axis.

__Returns__

한 개의 텐서.
    
----

### reshape


```python
keras.backend.reshape(x, shape)
```


Reshapes a tensor to the specified shape.

__Arguments__

- __x__: 텐서 또는 변수.
- __shape__: Target shape tuple.

__Returns__

한 개의 텐서.
    
----

### permute_dimensions


```python
keras.backend.permute_dimensions(x, pattern)
```


Permutes axes in a tensor.

__Arguments__

- __x__: 텐서 또는 변수.
- __pattern__: A tuple of
    dimension indices, e.g. `(0, 2, 1)`.

__Returns__

한 개의 텐서.
    
----

### resize_images


```python
keras.backend.resize_images(x, height_factor, width_factor, data_format, interpolation='nearest')
```


Resizes the images contained in a 4D tensor.

__Arguments__

- __x__: 텐서 또는 변수. to resize.
- __height_factor__: Positive integer.
- __width_factor__: Positive integer.
- __data_format__: string, `"channels_last"` or `"channels_first"`.
- __interpolation__: A string, one of `nearest` or `bilinear`.

__Returns__

한 개의 텐서.

__Raises__

- __ValueError__: if `data_format` is

neither `"channels_last"` or `"channels_first"`.
    
----

### resize_volumes


```python
keras.backend.resize_volumes(x, depth_factor, height_factor, width_factor, data_format)
```


Resizes the volume contained in a 5D tensor.

__Arguments__

- __x__: 텐서 또는 변수. to resize.
- __depth_factor__: Positive integer.
- __height_factor__: Positive integer.
- __width_factor__: Positive integer.
- __data_format__: string, `"channels_last"` or `"channels_first"`.

__Returns__

한 개의 텐서.

__Raises__

- __ValueError__: if `data_format` is

neither `"channels_last"` or `"channels_first"`.
    
----

### repeat_elements


```python
keras.backend.repeat_elements(x, rep, axis)
```


Repeats the elements of a tensor along an axis, like `np.repeat`.

If `x` has shape `(s1, s2, s3)` and `axis` is `1`, the output
will have shape `(s1, s2 * rep, s3)`.

__Arguments__

- __x__: 텐서 또는 변수.
- __rep__: Python integer, number of times to repeat.
- __axis__: Axis along which to repeat.

__Returns__

한 개의 텐서.
    
----

### repeat


```python
keras.backend.repeat(x, n)
```


Repeats a 2D tensor.

if `x` has shape (samples, dim) and `n` is `2`,
the output will have shape `(samples, 2, dim)`.

__Arguments__

- __x__: 텐서 또는 변수..
- __n__: Python integer, number of times to repeat.

__Returns__

A tensor.
    
----

### arange


```python
keras.backend.arange(start, stop=None, step=1, dtype='int32')
```


Creates a 1D tensor containing a sequence of integers.

The function arguments use the same convention as
Theano's arange: if only one argument is provided,
it is in fact the "stop" argument and "start" is 0.

The default type of the returned tensor is `'int32'` to
match TensorFlow's default.

__Arguments__

- __start__: Start value.
- __stop__: Stop value.
- __step__: Difference between two successive values.
- __dtype__: Integer dtype to use.

__Returns__

An integer tensor.


----

### tile


```python
keras.backend.tile(x, n)
```


Creates a tensor by tiling `x` by `n`.

__Arguments__

- __x__: A tensor or variable
- __n__: A list of integer. The length must be the same as the number of
    dimensions in `x`.

__Returns__

A tiled tensor.
    
----

### flatten


```python
keras.backend.flatten(x)
```


Flatten a tensor.

__Arguments__

- __x__: 텐서 또는 변수.

__Returns__

A tensor, reshaped into 1-D
    
----

### batch_flatten


```python
keras.backend.batch_flatten(x)
```


Turn a nD tensor into a 2D tensor with same 0th dimension.

In other words, it flattens each data samples of a batch.

__Arguments__

- __x__: 텐서 또는 변수.

__Returns__

A tensor.
    
----

### expand_dims


```python
keras.backend.expand_dims(x, axis=-1)
```


Adds a 1-sized dimension at index "axis".

__Arguments__

- __x__: 텐서 또는 변수.
- __axis__: Position where to add a new axis.

__Returns__

A tensor with expanded dimensions.
    
----

### squeeze


```python
keras.backend.squeeze(x, axis)
```


Removes a 1-dimension from the tensor at index "axis".

__Arguments__

- __x__: 텐서 또는 변수.
- __axis__: Axis to drop.

__Returns__

A tensor with the same data as `x` but reduced dimensions.
    
----

### temporal_padding


```python
keras.backend.temporal_padding(x, padding=(1, 1))
```


Pads the middle dimension of a 3D tensor.

__Arguments__

- __x__: 텐서 또는 변수.
- __padding__: Tuple of 2 integers, how many zeros to
    add at the start and end of dim 1.

__Returns__

A padded 3D tensor.
    
----

### spatial_2d_padding


```python
keras.backend.spatial_2d_padding(x, padding=((1, 1), (1, 1)), data_format=None)
```


Pads the 2nd and 3rd dimensions of a 4D tensor.

__Arguments__

- __x__: 텐서 또는 변수.
- __padding__: Tuple of 2 tuples, padding pattern.
- __data_format__: string, `"channels_last"` or `"channels_first"`.

__Returns__

A padded 4D tensor.

__Raises__

- __ValueError__: if `data_format` is

neither `"channels_last"` or `"channels_first"`.
    
----

### spatial_3d_padding


```python
keras.backend.spatial_3d_padding(x, padding=((1, 1), (1, 1), (1, 1)), data_format=None)
```


Pads 5D tensor with zeros along the depth, height, width dimensions.

Pads these dimensions with respectively
"padding[0]", "padding[1]" and "padding[2]" zeros left and right.

For 'channels_last' data_format,
the 2nd, 3rd and 4th dimension will be padded.
For 'channels_first' data_format,
the 3rd, 4th and 5th dimension will be padded.

__Arguments__

- __x__: 텐서 또는 변수.
- __padding__: Tuple of 3 tuples, padding pattern.
- __data_format__: string, `"channels_last"` or `"channels_first"`.

__Returns__

A padded 5D tensor.

__Raises__

- __ValueError__: if `data_format` is

neither `"channels_last"` or `"channels_first"`.


----

### stack


```python
keras.backend.stack(x, axis=0)
```


Stacks a list of rank `R` tensors into a rank `R+1` tensor.

__Arguments__

- __x__: List of tensors.
- __axis__: Axis along which to perform stacking.

__Returns__

A tensor.

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

- __indices__: nD integer tensor of shape
    `(batch_size, dim1, dim2, ... dim(n-1))`
- __num_classes__: Integer, number of classes to consider.

__Returns__

(n + 1)D one hot representation of the input
with shape `(batch_size, dim1, dim2, ... dim(n-1), num_classes)`
    
----

### reverse


```python
keras.backend.reverse(x, axes)
```


지정된 축을 따라 텐서를 반전시킵니다.

__Arguments__

- __x__: Tensor to reverse.
- __axes__: Integer or iterable of integers.
    Axes to reverse.

__Returns__

A tensor.

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


변수의 값을 반환한다.

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


Selects `x` in train phase, and `alt` otherwise.

Note that `alt` should have the *same shape* as `x`.

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

- __x__: 테스트 단계에서 반환 할 내용. 
    (tensor or callable that returns a tensor).
- __alt__: 다른 경우 반환 할 내용.
    (tensor or callable that returns a tensor).
- __training__: 학습 단계를 지정한 선택적 스칼라 텐서. 
    (<sag>Python boolean</sag> 또는 <sag>Python integer</sag>)



__Returns__

'learning_phase()'에 기반한 `x` 또는 `alt'.
    
----

### relu


```python
keras.backend.relu(x, alpha=0.0, max_value=None, threshold=0.0)
```


정제된 선형 단위.

With default values, it returns element-wise `max(x, 0)`.

Otherwise, it follows:
`f(x) = max_value` for `x >= max_value`,
`f(x) = x` for `threshold <= x < max_value`,
`f(x) = alpha * (x - threshold)` otherwise.

__Arguments__

- __x__: 텐서 또는 변수. 
- __alpha__: 음수 섹션의 스칼라, 기울기 (default=`0.`).
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

- __x__: 활성화 함수를 계산할 텐서 또는 변수 입니다. 
- __alpha__: 음수 섹션의 스칼라, 기울기. 

__Returns__

텐서. 

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

- __x__: 텐서 또는 변수. 
- __axis__: 차수 softmax가 수행 됩니다. 
    기본값은 -1을 나타내며 마지막 차원을 나타냅니다. 

__Returns__

한 개의 텐서.

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

- __target__: `output`과 같은 모양의 텐서.
- __output__: softmax의 결과 텐서.
    (unless `from_logits` is True, in which
    case `output` is expected to be the logits).
- __from_logits__: <sag>boolean</sag>, <sag>logits</sag>의 텐서이거나 softmax의 결과의 'output' 입니다. 
- __axis__: 채널 축을 지정합니다. `axis=-1`
    `channels_last`형식 데이터에 해당합니다,
    `channels_first` 데이터 형식은 `axis=1`에 해당 합니다. 

__Returns__

출력 텐서. 

__Raises__

- __ValueError__: `output`의 축 도 아니고 -1도 아닌 축.
    
----

### sparse_categorical_crossentropy


```python
keras.backend.sparse_categorical_crossentropy(target, output, from_logits=False, axis=-1)
```


정수 목표를 가진 범주형 크로스엔트로피.

__Arguments__

- __target__: An integer tensor.
- __output__: softmax의 결과로 나온 텐서. 
    (unless `from_logits` is True, in which
    case `output` is expected to be the logits).
- __from_logits__: Boolean, whether `output` is the
    result of a softmax, or is a tensor of logits.
- __axis__:
    `channels_last` 데이터 형식에 해당하는  Int 채널 축을 지정합니다. `axis=-1`
    and `axis=1` corresponds to data format `channels_first`.

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


출력 텐서와 목표 텐서 사나의 이진 크로스엔트로피.

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


각 세그먼트의 sigmoid 선형 근사.

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

.
전체 텐서를 스케일링하는 동안 'x'의 항목을 임의로 설정합니다. 

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


지정된 축을 따라 L2 norm으로 텐서를 정규화 시킨다. 


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

- __predictions__: `float32`타입과  `(batch_size, classes)`형식의 텐서.
- __targets__: `batch_size` and type `int32` or `int64`의 길이의 1차원 텐서. 
- __k__: An `int`, 고려해야 할 최상위 요소의 수. 

__Returns__

A 1D tensor of length `batch_size` and type `bool`.
만약 `predictions[i, targets[i]]` 이 top-`k`내에 있다면, `output[i]` 이 `True`.
`predictions[i]'의 값. 
    
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

- __x__: 텐서 또는 변수.
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

- __x__: 텐서 또는 변수.
- __kernel__: 커널 텐서. 
- __output_shape__: 1D int tensor 출력 형식에 대해 1차원 <sag>int</sag>텐서 
- __strides__: 스트라이드 튜플. 
- __padding__: <sag>string</sag>, `"same"` 또는 `"valid"`.
- __data_format__: <sag>string</sag>, `"channels_last"` 또는 `"channels_first"`.
    inputs/kernels/outputs에 대한 Theano 또는 TensorFlow/CNTK 데이터 형태 
- __dilation_rate__: 2 <sag>integers</sag>의 튜플.

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


분리가능한 필터와 1차원 컨볼루션 연산.

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


분리가능한 필터와 2차원 컨볼루션 연산.

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

출력 텐서. 

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

- __x__: 텐서 또는 변수.
- __kernel__: kernel tensor.
- __strides__: strides tuple.
- __padding__: string, `"same"` or `"valid"`.
- __data_format__: string, `"channels_last"` or `"channels_first"`.
    Whether to use Theano or TensorFlow/CNTK data format
    for inputs/kernels/outputs.
- __dilation_rate__: tuple of 3 integers.

__Returns__

3차원 컨볼루션 연산 결과.

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

트렌스포즈된 3차원 컨볼루션 연산결과 텐서.

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

- __x__: 텐서 또는 변수.
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

- __x__: 텐서 또는 변수.
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


텐서에 대한 바이어스 벡터 추가. 

__Arguments__

- __x__: 텐서 또는 변수.
- __bias__: 추가 할 바이어스 텐서. 
- __data_format__: <sag>string</sag>, `"channels_last"` 또는 `"channels_first"`.

__Returns__

결과 텐서.

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


값의 정규분포를 포함한 텐서를 반환 합니다.

__Arguments__

- __shape__: <sag>integers</sag>의 튜플, 생성할 텐서의 형식.
- __mean__: <sag>float</sag>, 정규 분포의 평균 그리기.
- __stddev__: <sag>float</sag>, 정규 분포의 표준편차 그리기.
- __dtype__: <sag>string</sag>, 반환된 텐서의 dtype.
- __seed__: <sag>Integer</sag>, random seed.

__Returns__

텐서.
    
----

### random_uniform


```python
keras.backend.random_uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None)
```


값의 균등분포를 포함한 텐서를 반환 합니다. 

__Arguments__

- __shape__: <sag>integers</sag>의 튜플, 생성할 텐서의 형식.
- __minval__: <sag>float</sag>, 균등 분포의 하한 샘플 그리기.
- __maxval__: <sag>float</sag>, 균등 분포의 상한 샘플 그리기.
- __dtype__: <sag>string</sag>, 반환된 텐서의 dtype.
- __seed__: <sag>Integer</sag>, random seed.

__Returns__

A tensor.
    
----

### random_binomial


```python
keras.backend.random_binomial(shape, p=0.0, dtype=None, seed=None)
```



값의 임의의 이항 분포의 텐서를 반환합니다. 

__Arguments__

- __shape__: <sag>integers</sag>의 튜플, 생성할 텐서의 형식.
- __p__: <sag>float</sag>, `0. <= p <= 1`범위의 이항 분포의 확률
- __dtype__: <sag>string</sag>, 반환된 텐서의 dtype.
- __seed__: <sag>Integer</sag>, random seed.

__Returns__

텐서.
    
----

### truncated_normal


```python
keras.backend.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None)
```


값의 임의의 정규분포가 잘린 텐서를 반환합니다. 


평균에 대한 두 표준편차가 제거되고 다시 지정되어 크기가 더 큰 값을 제외한 뒤
지정된 평균과 표준편차로 정규푼보에 따라 생성된 값.

__Arguments__

- __shape__: <sag>integers</sag>의 튜플, 생성할 텐서의 형식.
- __mean__: 값들의 평균.
- __stddev__: 값들의 표준편차. 
- __dtype__: <sag>string</sag>, 반환된 텐서의 dtype.
- __seed__: <sag>Integer</sag>, random seed.

__Returns__

텐서.
    
----

### ctc_label_dense_to_sparse


```python
keras.backend.ctc_label_dense_to_sparse(labels, label_lengths)
```


<sag>dense</sag>에서 <sag>sparse</sag>로 CTC레이블을 변환합니다.


__Arguments__

- __labels__: <sag>dense</sag> CTC 레이블.
- __label_lengths__: 레이블의 길이.

__Returns__


레이블의 희소 텐서 표현.
    
----

### ctc_batch_cost


```python
keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
```


각 배치에서 CTC손실 알고리즘을 수행합니다.  

__Arguments__

- __y_true__: truth 레이블을 포함한 `(samples, max_string_length)` 텐서.
- __y_pred__: softmax의 출력 또는 예측값을 포함한 `(samples, time_steps, num_categories)` 텐서.
- __input_length__: `y_pred`의 각 배치 항목의 시퀀스 길이를 포함하는 `(samples, 1)`텐서.
- __label_length__:  `y_true`의 각 배치 항목의 시퀀스 길이를 포함하는 `(samples, 1)`텐서.

__Returns__


각 요소의 CTC 손실값을 포함한 텐서의 (samples,1)형식
    
----

### ctc_decode


```python
keras.backend.ctc_decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1)
```


소프트맥스의 결과를 해석.


그리디 탐색(최적화)이나 제한적인 딕셔너리 탐색이 가능합니다.

__Arguments__

- __y_pred__: 예측을 포함한  `(samples, time_steps, num_categories)` 텐서 또는 소프트맥스의 출력.
- __input_length__: `y_pred`의 각 배치 항목에 대한 시퀀스 길이를 포함한 `(samples, )`텐서. 
- __greedy__: 만약 `true`라면 훨씬 더 빠르고 좋은 탐색을 수행합니다. 딕셔너리 자료형을 사용하지 않습니다. 
- __beam_width__: `greedy`가 `false`일 때, beam 탐색 디코더가 너비의 beam으로 사용됩니다. 
- __top_paths__: `greedy`가 `false`일 때, 가장 가능할만한 경로 중에 얼마나 많은 경로가 있는지 반환합니다. 
   

__Returns__

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

__Arguments__

- __fn__: <sag>elems</sag>에 있는 각 요소에 대해 호출가능.
- __elems__: 텐서
- __name__: 그래프에서 맵 노드에 대한 문자열 이름. 
- __dtype__: 출력 데이터 타입.

__Returns__

`dtype`의 텐서.
    
----

### foldl


```python
keras.backend.foldl(fn, elems, initializer=None, name=None)
```


왼쪽에서 오른쪽으로 결합하기위해 <sag>fn</sag>을 사용해 요소를 감소시킵니다.  

__Arguments__

- __fn__: <sag>elems</sag>에서 각 요소에 호출 될 연산기, 예를 들어, `lambda acc, x: acc + x` 
- __elems__: 텐서
- __initializer__: 사용된 첫 번째 값. (`elems[0]` in case of None)
- __name__: 그래프 fodl 노드에 대한 문자열 이름.

__Returns__

`initializer` 모양과 같은 타입의 텐서.
    
----

### foldr


```python
keras.backend.foldr(fn, elems, initializer=None, name=None)
```


<sag>fn</sag>인자를 사용하여 오른쪽에서 왼쪽으로 텐서 요소들을 줄인다.

__Arguments__

- __fn__: <sag>elems</sag>에서 호출가능한 각 요소와 누산기. 
    예를들어, `lambda acc, x: acc + x`
- __elems__: 텐서
- __initializer__: 사용된 첫번 째 값 (`elems[-1]` in case of None)
- __name__: 그래프에서 <sag>foldr node</sag>의 문자열 이름

__Returns__

`initializer` 모양과 같은 타입의 텐서.
    
----

### local_conv1d


```python
keras.backend.local_conv1d(inputs, kernel, kernel_size, strides, data_format=None)
```

공유되지 않은 가중치를 1D 컨볼루션에 적용합니다.

__Arguments__

- __inputs__: 3D 텐서의 형식: (batch_size, steps, input_dim)
- __kernel__: (output_length, feature_dim, filters)형식의 컨볼루션의 공유되지 않은 가중치.
- __kernel_size__: 1d 컨볼루션 윈도우의 길이를 지정한 단일 <sag>integer</sag> 튜플.
- __strides__: 컨볼루션의 스타라이드 길이를 지정한 단일 <sag>integer</sag> 튜플.
- __data_format__: 데이터 형식, channels_first 또는 channels_last

__Returns__


(batch_size, output_length, filters)형식: 공유되지 않은 가중치로 1d 컨볼루션 연산 후의 텐서.

__Raises__

- __ValueError__: If `data_format`가 
    <sag>channels_last</sag> 또는 <sag>channels_first"`이 아닐 때, 오류.
    
----

### local_conv2d


```python
keras.backend.local_conv2d(inputs, kernel, kernel_size, strides, output_shape, data_format=None)
```


2D 컨볼루션에 공유되지 않은 가중치를 적용합니다.


__Arguments__

- __inputs__: 
        data_format='channels_first'일 때, 4D 텐서 형식:
        (batch_size, filters, new_rows, new_cols)
        data_format='channels_last'일 때, 4D 텐서 형식:
        (batch_size, new_rows, new_cols, filters)
- __kernel__: (output_items, feature_dim, filters) 형식의 컨볼루션 연산을 위한 공유되지 않은 가중치
- __kernel_size__: 2차원 컨볼루션 윈도우의 너비와 높이를 지정한 2<sag>integers</sag>의 튜플.
- __strides__: 2<sag>integers</sag>인 튜플,  너비와 높이에 따른 컨볼루션의 스트라이드를 지정합니다. 
- __output_shape__: (output_row, output_col)형태의 튜플
- __data_format__: 데이터 형식, channels_first 또는 channels_last

__Returns__

4d 텐서의 형식:
data_format='channels_first'일 때,
(batch_size, filters, new_rows, new_cols)

 4d 텐서의 형식:
data_format='channels_last'일 때,
(batch_size, new_rows, new_cols, filters)


__Raises__

- __ValueError__: <sag>data_format</sag>가 
            <sag>channels_last</sag> 또는 <sag>channels_first</sag>이 아니었을 때, 오류.
    
----

### backend


```python
keras.backend.backend()
```



백엔드를 결정하기 위한 공개접근방식.


__Returns__

<sag>string</sag>, 현재 사용 중인 케라스 백엔드 이름.

__Example__

```python
>>> keras.backend.backend()
'tensorflow'
```
    

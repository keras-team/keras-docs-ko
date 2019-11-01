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


수치 식에 사용되는 fuzz factor의 값을 반환합니다.

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

- __e__: float. New value of epsilon.

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

Learning phase (scalar integer tensor or Python integer).

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


Returns whether a tensor is a sparse tensor.

__Arguments__

- __tensor__: A tensor instance.

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


Converts a sparse tensor into a dense tensor and returns it.

__Arguments__

- __tensor__: A tensor instance (potentially sparse).

__Returns__

A dense tensor.

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


Instantiates a placeholder tensor and returns it.

__Arguments__

- __shape__: Shape of the placeholder
    (integer tuple, may include `None` entries).
- __ndim__: Number of axes of the tensor.
    At least one of {`shape`, `ndim`} must be specified.
    If both are specified, `shape` is used.
- __dtype__: Placeholder type.
- __sparse__: Boolean, whether the placeholder should have a sparse type.
- __name__: Optional name string for the placeholder.

__Returns__

Tensor instance (with Keras metadata included).

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


Returns whether `x` is a placeholder.

__Arguments__

- __x__: A candidate placeholder.

__Returns__

Boolean.
    
----

### shape


```python
keras.backend.shape(x)
```


Returns the symbolic shape of a tensor or variable.

__Arguments__

- __x__: A tensor or variable.

__Returns__

A symbolic shape (which is itself a tensor).

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


Returns the shape of tensor or variable as a tuple of int or None entries.

__Arguments__

- __x__: Tensor or variable.

__Returns__

A tuple of integers (or None entries).

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


Returns the number of axes in a tensor, as an integer.

__Arguments__

- __x__: Tensor or variable.

__Returns__

Integer (scalar), number of axes.

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


Returns the dtype of a Keras tensor or variable, as a string.

__Arguments__

- __x__: Tensor or variable.

__Returns__

String, dtype of `x`.

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


Evaluates the value of a variable.

__Arguments__

- __x__: A variable.

__Returns__

A Numpy array.

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


Instantiates an all-zeros variable and returns it.

__Arguments__

- __shape__: Tuple of integers, shape of returned Keras variable
- __dtype__: String, data type of returned Keras variable
- __name__: String, name of returned Keras variable

__Returns__

A variable (including Keras metadata), filled with `0.0`.
Note that if `shape` was symbolic, we cannot return a variable,
and will return a dynamically-shaped tensor instead.

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


Instantiates an all-ones variable and returns it.

__Arguments__

- __shape__: Tuple of integers, shape of returned Keras variable.
- __dtype__: String, data type of returned Keras variable.
- __name__: String, name of returned Keras variable.

__Returns__

A Keras variable, filled with `1.0`.
Note that if `shape` was symbolic, we cannot return a variable,
and will return a dynamically-shaped tensor instead.

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


Instantiate an identity matrix and returns it.

__Arguments__

- __size__: Integer, number of rows/columns.
- __dtype__: String, data type of returned Keras variable.
- __name__: String, name of returned Keras variable.

__Returns__

A Keras variable, an identity matrix.

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


Instantiates an all-zeros variable of the same shape as another tensor.

__Arguments__

- __x__: Keras variable or Keras tensor.
- __dtype__: String, dtype of returned Keras variable.
     None uses the dtype of x.
- __name__: String, name for the variable to create.

__Returns__

A Keras variable with the shape of x filled with zeros.

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


Returns a tensor with the same content as the input tensor.

__Arguments__

- __x__: The input tensor.
- __name__: String, name for the variable to create.

__Returns__

A tensor of the same shape, type and content.
    
----

### random_uniform_variable


```python
keras.backend.random_uniform_variable(shape, low, high, dtype=None, name=None, seed=None)
```


Instantiates a variable with values drawn from a uniform distribution.

__Arguments__

- __shape__: Tuple of integers, shape of returned Keras variable.
- __low__: Float, lower boundary of the output interval.
- __high__: Float, upper boundary of the output interval.
- __dtype__: String, dtype of returned Keras variable.
- __name__: String, name of returned Keras variable.
- __seed__: Integer, random seed.

__Returns__

A Keras variable, filled with drawn samples.

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


Instantiates a variable with values drawn from a normal distribution.

__Arguments__

- __shape__: Tuple of integers, shape of returned Keras variable.
- __mean__: Float, mean of the normal distribution.
- __scale__: Float, standard deviation of the normal distribution.
- __dtype__: String, dtype of returned Keras variable.
- __name__: String, name of returned Keras variable.
- __seed__: Integer, random seed.

__Returns__

A Keras variable, filled with drawn samples.

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


Returns the static number of elements in a Keras variable or tensor.

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


Casts a tensor to a different dtype and returns it.

You can cast a Keras variable but it still returns a Keras tensor.

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


Compute the moving average of a variable.

__Arguments__

- __x__: A `Variable`.
- __value__: A tensor with the same shape as `x`.
- __momentum__: The moving average momentum.

__Returns__

An operation to update the variable.
    
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

- __x__: Tensor or variable.
- __y__: Tensor or variable.

__Returns__

A tensor, dot product of `x` and `y`.

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


Batchwise dot product.

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


Transposes a tensor and returns it.

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


Retrieves the elements of indices `indices` in the tensor `reference`.

__Arguments__

- __reference__: A tensor.
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

A tensor with sum of `x`.

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


Multiplies the values in a tensor, alongside the specified axis.

__Arguments__

- __x__: A tensor or variable.
- __axis__: An integer or list of integers in [-rank(x), rank(x)),
    the axes to compute the product. If `None` (default), computes
    the product over all dimensions.
- __keepdims__: A boolean, whether to keep the dimensions or not.
    If `keepdims` is `False`, the rank of the tensor is reduced
    by 1. If `keepdims` is `True`,
    the reduced dimension is retained with length 1.

__Returns__

A tensor with the product of elements of `x`.

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


Cumulative sum of the values in a tensor, alongside the specified axis.

__Arguments__

- __x__: A tensor or variable.
- __axis__: An integer, the axis to compute the sum.

__Returns__

A tensor of the cumulative sum of values of `x` along `axis`.
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


Cumulative product of the values in a tensor, alongside the specified axis.

__Arguments__

- __x__: A tensor or variable.
- __axis__: An integer, the axis to compute the product.

__Returns__

A tensor of the cumulative product of values of `x` along `axis`.
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


Variance of a tensor, alongside the specified axis.

__Arguments__

- __x__: A tensor or variable.
- __axis__: An integer or list of integers in [-rank(x), rank(x)),
    the axes to compute the variance. If `None` (default), computes
    the variance over all dimensions.
- __keepdims__: A boolean, whether to keep the dimensions or not.
    If `keepdims` is `False`, the rank of the tensor is reduced
    by 1. If `keepdims` is `True`,
    the reduced dimension is retained with length 1.

__Returns__

A tensor with the variance of elements of `x`.
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


Standard deviation of a tensor, alongside the specified axis.

__Arguments__

- __x__: A tensor or variable.
- __axis__: An integer or list of integers in [-rank(x), rank(x)),
    the axes to compute the standard deviation. If `None` (default),
    computes the standard deviation over all dimensions.
- __keepdims__: A boolean, whether to keep the dimensions or not.
    If `keepdims` is `False`, the rank of the tensor is reduced
    by 1. If `keepdims` is `True`,
    the reduced dimension is retained with length 1.

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


Mean of a tensor, alongside the specified axis.

__Arguments__

- __x__: A tensor or variable.
- __axis__: An integer or list of integers in [-rank(x), rank(x)),
    the axes to compute the mean. If `None` (default), computes
    the mean over all dimensions.
- __keepdims__: A boolean, whether to keep the dimensions or not.
    If `keepdims` is `False`, the rank of the tensor is reduced
    by 1 for each entry in `axis`. If `keepdims` is `True`,
    the reduced dimensions are retained with length 1.

__Returns__

A tensor with the mean of elements of `x`.
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


Bitwise reduction (logical OR).

__Arguments__

- __x__: Tensor or variable.
- __axis__: An integer or list of integers in [-rank(x), rank(x)),
    the axes to compute the logical or. If `None` (default), computes
    the logical or over all dimensions.
- __keepdims__: whether the drop or broadcast the reduction axes.

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


Bitwise reduction (logical AND).

__Arguments__

- __x__: Tensor or variable.
- __axis__: An integer or list of integers in [-rank(x), rank(x)),
    the axes to compute the logical and. If `None` (default), computes
    the logical and over all dimensions.
- __keepdims__: whether the drop or broadcast the reduction axes.

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

- __x__: Tensor or variable.
- __y__: Tensor or variable.

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

- __x__: Tensor or variable.
- __y__: Tensor or variable.

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

- __x__: Tensor or variable.
- __y__: Tensor or variable.

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

- __x__: Tensor or variable.
- __y__: Tensor or variable.

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

- __x__: Tensor or variable.
- __y__: Tensor or variable.

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

- __x__: Tensor or variable.
- __y__: Tensor or variable.

__Returns__

A tensor.

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

- __x__: Tensor or variable.
- __y__: Tensor or variable.

__Returns__

A tensor.

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

- __x__: Tensor or variable.

__Returns__

A tensor.
    
----

### cos


```python
keras.backend.cos(x)
```


Computes cos of x element-wise.

__Arguments__

- __x__: Tensor or variable.

__Returns__

A tensor.
    
----

### normalize_batch_in_training


```python
keras.backend.normalize_batch_in_training(x, gamma, beta, reduction_axes, epsilon=0.001)
```


Computes mean and std for batch then apply batch_normalization on batch.

__Arguments__

- __x__: Input tensor or variable.
- __gamma__: Tensor by which to scale the input.
- __beta__: Tensor with which to center the input.
- __reduction_axes__: iterable of integers,
    axes over which to normalize.
- __epsilon__: Fuzz factor.

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

- __x__: Input tensor or variable.
- __mean__: Mean of batch.
- __var__: Variance of batch.
- __beta__: Tensor with which to center the input.
- __gamma__: Tensor by which to scale the input.
- __axis__: Integer, the axis that should be normalized.
    (typically the features axis).
- __epsilon__: Fuzz factor.

__Returns__

A tensor.
    
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

A tensor.
    
----

### reshape


```python
keras.backend.reshape(x, shape)
```


Reshapes a tensor to the specified shape.

__Arguments__

- __x__: Tensor or variable.
- __shape__: Target shape tuple.

__Returns__

A tensor.
    
----

### permute_dimensions


```python
keras.backend.permute_dimensions(x, pattern)
```


Permutes axes in a tensor.

__Arguments__

- __x__: Tensor or variable.
- __pattern__: A tuple of
    dimension indices, e.g. `(0, 2, 1)`.

__Returns__

A tensor.
    
----

### resize_images


```python
keras.backend.resize_images(x, height_factor, width_factor, data_format, interpolation='nearest')
```


Resizes the images contained in a 4D tensor.

__Arguments__

- __x__: Tensor or variable to resize.
- __height_factor__: Positive integer.
- __width_factor__: Positive integer.
- __data_format__: string, `"channels_last"` or `"channels_first"`.
- __interpolation__: A string, one of `nearest` or `bilinear`.

__Returns__

A tensor.

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

- __x__: Tensor or variable to resize.
- __depth_factor__: Positive integer.
- __height_factor__: Positive integer.
- __width_factor__: Positive integer.
- __data_format__: string, `"channels_last"` or `"channels_first"`.

__Returns__

A tensor.

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

- __x__: Tensor or variable.
- __rep__: Python integer, number of times to repeat.
- __axis__: Axis along which to repeat.

__Returns__

A tensor.
    
----

### repeat


```python
keras.backend.repeat(x, n)
```


Repeats a 2D tensor.

if `x` has shape (samples, dim) and `n` is `2`,
the output will have shape `(samples, 2, dim)`.

__Arguments__

- __x__: Tensor or variable.
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

- __x__: A tensor or variable.

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

- __x__: A tensor or variable.

__Returns__

A tensor.
    
----

### expand_dims


```python
keras.backend.expand_dims(x, axis=-1)
```


Adds a 1-sized dimension at index "axis".

__Arguments__

- __x__: A tensor or variable.
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

- __x__: A tensor or variable.
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

- __x__: Tensor or variable.
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

- __x__: Tensor or variable.
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

- __x__: Tensor or variable.
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


Computes the one-hot representation of an integer tensor.

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


Reverses a tensor along the specified axes.

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


Extracts a slice from a tensor.

__Arguments__

- __x__: Input tensor.
- __start__: Integer list/tuple or tensor
    indicating the start indices of the slice
    along each axis.
- __size__: Integer list/tuple or tensor
    indicating how many dimensions to slice
    along each axis.

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

- __x__: input variable.

__Returns__

A Numpy array.
    
----

### batch_get_value


```python
keras.backend.batch_get_value(ops)
```


Returns the value of more than one tensor variable.

__Arguments__

- __ops__: list of ops to run.

__Returns__

A list of Numpy arrays.
    
----

### set_value


```python
keras.backend.set_value(x, value)
```


Sets the value of a variable, from a Numpy array.

__Arguments__

- __x__: Tensor to set to a new value.
- __value__: Value to set the tensor to, as a Numpy array
    (of the same shape).
    
----

### batch_set_value


```python
keras.backend.batch_set_value(tuples)
```


Sets the values of many tensor variables at once.

__Arguments__

- __tuples__: a list of tuples `(tensor, value)`.
    `value` should be a Numpy array.
    
----

### print_tensor


```python
keras.backend.print_tensor(x, message='')
```


Prints `message` and the tensor value when evaluated.

Note that `print_tensor` returns a new tensor identical to `x`
which should be used in the following code. Otherwise the
print operation is not taken into account during evaluation.

__Example__

```python
>>> x = K.print_tensor(x, message="x is: ")
```

__Arguments__

- __x__: Tensor to print.
- __message__: Message to print jointly with the tensor.

__Returns__

The same tensor `x`, unchanged.
    
----

### function


```python
keras.backend.function(inputs, outputs, updates=None)
```


Instantiates a Keras function.

__Arguments__

- __inputs__: List of placeholder tensors.
- __outputs__: List of output tensors.
- __updates__: List of update ops.
- __**kwargs__: Passed to `tf.Session.run`.

__Returns__

Output values as Numpy arrays.

__Raises__

- __ValueError__: if invalid kwargs are passed in.
    
----

### gradients


```python
keras.backend.gradients(loss, variables)
```


Returns the gradients of `loss` w.r.t. `variables`.

__Arguments__

- __loss__: Scalar tensor to minimize.
- __variables__: List of variables.

__Returns__

A gradients tensor.
    
----

### stop_gradient


```python
keras.backend.stop_gradient(variables)
```


Returns `variables` but with zero gradient w.r.t. every other variable.

__Arguments__

- __variables__: tensor or list of tensors to consider constant with respect
    to any other variable.

__Returns__

A single tensor or a list of tensors (depending on the passed argument)
    that has constant gradient with respect to any other variable.
    
----

### rnn


```python
keras.backend.rnn(step_function, inputs, initial_states, go_backwards=False, mask=None, constants=None, unroll=False, input_length=None)
```


Iterates over the time dimension of a tensor.

__Arguments__

- __step_function__:
    Parameters:
        inputs: Tensor with shape (samples, ...) (no time dimension),
            representing input for the batch of samples at a certain
            time step.
        states: List of tensors.
    Returns:
        outputs: Tensor with shape (samples, ...) (no time dimension),
        new_states: List of tensors, same length and shapes
            as 'states'.
- __inputs__: Tensor of temporal data of shape (samples, time, ...)
    (at least 3D).
- __initial_states__: Tensor with shape (samples, ...) (no time dimension),
    containing the initial values for the states used in
    the step function.
- __go_backwards__: Boolean. If True, do the iteration over the time
    dimension in reverse order and return the reversed sequence.
- __mask__: Binary tensor with shape (samples, time),
    with a zero for every element that is masked.
- __constants__: A list of constant values passed at each step.
- __unroll__: Whether to unroll the RNN or to use a symbolic loop
    (`while_loop` or `scan` depending on backend).
- __input_length__: Static number of timesteps in the input.

__Returns__

A tuple, `(last_output, outputs, new_states)`.

last_output: The latest output of the rnn, of shape `(samples, ...)`
outputs: Tensor with shape `(samples, time, ...)` where each
entry `outputs[s, t]` is the output of the step function
at time `t` for sample `s`.
new_states: List of tensors, latest states returned by
the step function, of shape `(samples, ...)`.

__Raises__

- __ValueError__: If input dimension is less than 3.
- __ValueError__: If `unroll` is `True`
    but input timestep is not a fixed number.
- __ValueError__: If `mask` is provided (not `None`)
    but states is not provided (`len(states)` == 0).

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


Switches between two operations depending on a scalar value.

Note that both `then_expression` and `else_expression`
should be symbolic tensors of the *same shape*.

__Arguments__

- __condition__: tensor (`int` or `bool`).
- __then_expression__: either a tensor, or a callable that returns a tensor.
- __else_expression__: either a tensor, or a callable that returns a tensor.

__Returns__

The selected tensor.

__Raises__

- __ValueError__: If rank of `condition` is greater than rank of expressions.

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

- __x__: What to return in train phase
    (tensor or callable that returns a tensor).
- __alt__: What to return otherwise
    (tensor or callable that returns a tensor).
- __training__: Optional scalar tensor
    (or Python boolean, or Python integer)
    specifying the learning phase.

__Returns__

Either `x` or `alt` based on the `training` flag.
the `training` flag defaults to `K.learning_phase()`.
    
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
    






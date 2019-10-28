<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/merge.py#L200)</span>

### Add

```python
keras.layers.Add()
```

입력 리스트에 대해 덧셈을 수행하는 층<sub>layer</sub>입니다.

동일한 형태<sub>shape</sub>의 텐서 리스트를 입력으로 받아, 동일한 형태의 하나의 텐서를 반환합니다.



__예시__

```python
import keras

input1 = keras.layers.Input(shape=(16,))
x1 = keras.layers.Dense(8, activation='relu')(input1)
input2 = keras.layers.Input(shape=(32,))
x2 = keras.layers.Dense(8, activation='relu')(input2)
# equivalent to added = keras.layers.add([x1, x2])
added = keras.layers.Add()([x1, x2])

out = keras.layers.Dense(4)(added)
model = keras.models.Model(inputs=[input1, input2], outputs=out)
```

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/merge.py#L231)</span>

### Subtract

```python
keras.layers.Subtract()
```

입력 리스트에 대해 뺄셈을 수행하는 층입니다.  

두 개의 동일한 형태의 텐서로 이루어진 리스트를 입력으로 받아
동일한 형태의 텐서 (inputs[0] - inputs[1]) 한 개를 반환합니다. 



__예시__

```python
import keras

input1 = keras.layers.Input(shape=(16,))
x1 = keras.layers.Dense(8, activation='relu')(input1)
input2 = keras.layers.Input(shape=(32,))
x2 = keras.layers.Dense(8, activation='relu')(input2)
# Equivalent to subtracted = keras.layers.subtract([x1, x2])
subtracted = keras.layers.Subtract()([x1, x2])

out = keras.layers.Dense(4)(subtracted)
model = keras.models.Model(inputs=[input1, input2], outputs=out)
```

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/merge.py#L268)</span>

### Multiply

```python
keras.layers.Multiply()
```

입력 리스트에 대해 원소별<sub>element-wise</sub> 곱셈을 수행하는 층입니다.

동일한 형태의 텐서 리스트를 입력으로 받아, 동일한 형태의 하나의 텐서를 반환합니다.

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/merge.py#L283)</span>

### Average

```python
keras.layers.Average()
```

입력 리스트의 평균을 계산하는 층입니다

동일한 형태의 텐서 리스트를 입력으로 받아, 동일한 형태의 하나의 텐서를 반환합니다.

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/merge.py#L298)</span>

### Maximum

```python
keras.layers.Maximum()
```

입력 리스트의 원소별 최댓값을 계산하는 층입니다.

동일한 형태의 텐서 리스트를 입력으로 받아, 동일한 형태의 하나의 텐서를 반환합니다.

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/merge.py#L313)</span>

### Minimum

```python
keras.layers.Minimum()
```

입력 리스트의 원소별 최솟값을 계산하는 층입니다.

동일한 형태의 텐서 리스트를 입력으로 받아, 동일한 형태의 하나의 텐서를 반환합니다.

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/merge.py#L328)</span>

### Concatenate

```python
keras.layers.Concatenate(axis=-1)
```

입력 리스트를 이어 붙이는 층입니다.

동일한 형태(이어 붙이는 축을 제외한)의 텐서 리스트를 입력으로 받아,
모든 입력이 이어 붙여진 하나의 텐서를 반환합니다.



__인자__

- __axis__: 이어 붙이기를 수행할 축입니다.
- __**kwargs__: 케라스 층에 표준적으로 사용되는 키워드 인자입니다.

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/merge.py#L416)</span>

### Dot

```python
keras.layers.Dot(axes, normalize=False)
```

두 텐서의 샘플들 간에 내적 연산을 수행하는 층입니다.

만약 형태가 `(batch_size, n)`인 두 텐서 `a`와 `b`를 갖는 리스트에 대해 내적 연산을 수행할 경우, 출력 텐서의 형태는 `(batch_size, 1)`가 됩니다. 이때 출력 텐서의 `i`번째 원소는 `a[i]`와 `b[i]`간 내적 연산의 결과값입니다. 

__인자__

- __axes__:  `int` 또는 `int`로 이루어진 튜플. 내적 연산을 수행할 축을 의미합니다.
- __normalize__: `bool`. 내적 연산을 수행하기 전에 샘플들에 L2 정규화<sub>normalization</sub>를 적용할지 결정합니다. `True`로 설정하면, 내적 연산의 결과는 두 샘플들 간의 코사인 유사도와 같습니다. 기본값은 `True`입니다. 
- __**kwargs__: 케라스 층에 표준적으로 사용되는 키워드 인자입니다.

------

### add

```python
keras.layers.add(inputs)
```

`Add` 층의 함수형 인터페이스입니다.

__인자__

- __inputs__: 입력 텐서의 리스트입니다. 최소 2개 이상의 텐서를 포함해야 합니다.
- __**kwargs__: 케라스 층에 표준적으로 사용되는 키워드 인자입니다.

__반환값__

입력값의 합을 텐서로 반환합니다. 

__예시__

```python
import keras

input1 = keras.layers.Input(shape=(16,))
x1 = keras.layers.Dense(8, activation='relu')(input1)
input2 = keras.layers.Input(shape=(32,))
x2 = keras.layers.Dense(8, activation='relu')(input2)
added = keras.layers.add([x1, x2])

out = keras.layers.Dense(4)(added)
model = keras.models.Model(inputs=[input1, input2], outputs=out)
```

------

### subtract

```python
keras.layers.subtract(inputs)
```

`Subtract` 층의 함수형 인터페이스입니다.

__인자__

- __inputs__: 입력 텐서의 리스트입니다. 정확히 2개의 텐서만을 포함해야 합니다.
- __**kwargs__: 케라스 층에 표준적으로 사용되는 키워드 인자입니다.

__반환값__

두 입력값의 차를 텐서로 반환합니다.

__예시__

```python
import keras

input1 = keras.layers.Input(shape=(16,))
x1 = keras.layers.Dense(8, activation='relu')(input1)
input2 = keras.layers.Input(shape=(32,))
x2 = keras.layers.Dense(8, activation='relu')(input2)
subtracted = keras.layers.subtract([x1, x2])

out = keras.layers.Dense(4)(subtracted)
model = keras.models.Model(inputs=[input1, input2], outputs=out)
```

------

### multiply

```python
keras.layers.multiply(inputs)
```

`Multiply` 층의 함수형 인터페이스입니다.

__인자__

- __inputs__: 입력 텐서의 리스트입니다. 최소 2개 이상의 텐서를 포함해야 합니다.
- __**kwargs__: 케라스 층에 표준적으로 사용되는 키워드 인자입니다.

__반환값__


입력값의 원소별 곱셈의 결과를 텐서로 반환합니다.
    

------

### average

```python
keras.layers.average(inputs)
```

`Average` 층의 함수형 인터페이스입니다.

__인자__

- __inputs__: 입력 텐서의 리스트입니다. 최소 2개 이상의 텐서를 포함해야 합니다.
- __**kwargs__: 케라스 층에 표준적으로 사용되는 키워드 인자입니다.

__반환값__


입력값의 평균을 텐서로 반환합니다.

    

------

### maximum

```python
keras.layers.maximum(inputs)
```

`Maximum` 층의 함수형 인터페이스입니다.

__인자__

- __inputs__: 입력 텐서의 리스트입니다. 최소 2개 이상의 텐서를 포함해야 합니다.
- __**kwargs__: 케라스 층에 표준적으로 사용되는 키워드 인자입니다.

__반환값__


입력값의 원소별 최댓값을 텐서로 반환합니다.
    

------

### minimum

```python
keras.layers.minimum(inputs)
```

`Minimum` 층의 함수형 인터페이스입니다.

__인자__

- __inputs__: 입력 텐서의 리스트입니다. 최소 2개 이상의 텐서를 포함해야 합니다.
- __**kwargs__: 케라스 층에 표준적으로 사용되는 키워드 인자입니다.

__반환값__


입력값의 원소별 최솟값을 텐서로 반환합니다.

------

### concatenate

```python
keras.layers.concatenate(inputs, axis=-1)
```

`Concatenate` 층의 함수형 인터페이스입니다.

__인자__

- __inputs__: 입력 텐서의 리스트입니다. 최소 2개 이상의 텐서를 포함해야 합니다.
- __axis__: 이어 붙이기를 수행할 축입니다.
- __**kwargs__: 케라스 층에 표준적으로 사용되는 키워드 인자입니다.

__반환값__


`axis`를 따라 입력값을 이어 붙인 텐서를 반환합니다.  
    

------

### dot

```python
keras.layers.dot(inputs, axes, normalize=False)
```

`Dot` 층의 함수형 인터페이스입니다.

__인자__

- __inputs__: 입력 텐서의 리스트입니다. 최소 2개 이상의 텐서를 포함해야 합니다.
- __axes__: `int` 또는 `int`로 이루어진 튜플. 내적 연산을 수행할 축을 의미합니다.
- __normalize__: `bool`. 내적 연산을 수행하기 전에 샘플들에 L2 정규화를 적용할지 결정합니다. `True`로 설정하면, 내적 연산의 결과는 두 샘플들 간의 코사인 유사도와 같습니다. 기본값은 `True`입니다. 
- __**kwargs__: 케라스 층에 표준적으로 사용되는 키워드 인자입니다.

__반환값__


입력 샘플들 간에 내적 연산을 수행한 결과를 텐서로 반환합니다.
    

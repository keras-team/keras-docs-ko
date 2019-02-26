<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/merge.py#L200)</span>
### Add

```python
keras.layers.Add()
```

입력 리스트를 더하는 레이어입니다. 

입력으로 모든 형태의 텐서 리스트를 받아 같은 모양을 가진 하나의 텐서로 반환합니다. 

__예시__


```python
import keras

input1 = keras.layers.Input(shape=(16,))
x1 = keras.layers.Dense(8, activation='relu')(input1)
input2 = keras.layers.Input(shape=(32,))
x2 = keras.layers.Dense(8, activation='relu')(input2)
# added = keras.layers.add([x1, x2])와 동일합니다.
added = keras.layers.Add()([x1, x2])

out = keras.layers.Dense(4)(added)
model = keras.models.Model(inputs=[input1, input2], outputs=out)
```
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/merge.py#L231)</span>
### Subtract

```python
keras.layers.Subtract()
```

두 인풋에 대해 뺄셈을 수행하는 레이어입니다. 

동일한 크기를 가진 텐서리스트(2개의 텐서)를 입력으로 받아 동일한 크기의 단일 텐서로 반환합니다. (inputs[0] - inputs[1]),

__예시__


```python
import keras

input1 = keras.layers.Input(shape=(16,))
x1 = keras.layers.Dense(8, activation='relu')(input1)
input2 = keras.layers.Input(shape=(32,))
x2 = keras.layers.Dense(8, activation='relu')(input2)
# subtracted = keras.layers.subtract([x1, x2])와 동일합니다. 
subtracted = keras.layers.Subtract()([x1, x2])

out = keras.layers.Dense(4)(subtracted)
model = keras.models.Model(inputs=[input1, input2], outputs=out)
```
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/merge.py#L268)</span>
### Multiply

```python
keras.layers.Multiply()
```

입력 리스트에 대해 요소 단위로 곱하는 레이어입니다. 

입력으로 모든 형태의 텐서 리스트를 받아 같은 모양을 가진 하나의 텐서로 반환합니다. 

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/merge.py#L283)</span>
### Average

```python
keras.layers.Average()
```

입력 리스트에 대해 평균을 계산하는 레이어입니다. 

입력으로 모든 형태의 텐서 리스트를 받아 같은 모양을 가진 하나의 텐서로 반환합니다. 

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/merge.py#L298)</span>
### Maximum

```python
keras.layers.Maximum()
```

입력 리스트에 대해 최댓값(요소 단위)을 계산하는 레이어입니다.

입력으로 모든 형태의 텐서 리스트를 받아 같은 모양을 가진 하나의 텐서로 반환합니다. 

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/merge.py#L328)</span>
### Concatenate

```python
keras.layers.Concatenate(axis=-1)
```

입력 리스트에 대해 이어붙이기(Concatenate) 연산을 수행하는 레이어입니다. 

concatenation 축을 제외하고 동일한 크기의 모든 텐서 리스트를 입력으로 받아 모든 입력이 이어붙여진 단일 텐서를 반환합니다. 

__인자 설명__

- __axis__: 이어붙이기를 수행할 축입니다. 
- __**kwargs__: 표준 레이어 키워드 인자입니다. 
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/merge.py#L416)</span>
### Dot

```python
keras.layers.Dot(axes, normalize=False)
```

두 개의 텐서 안에 있는 두 샘플 간에 내적 연산을 수행하는 레이어입니다. 

예시: 만약 크기가 `(batch_size, n)`인 두 텐서 `a`와 `b`를 갖는 리스트에서 수행될 경우, 출력 텐서의 크기는 `(batch_size, 1)`가 됩니다. 출력 텐서의 각 요소 `i`는 `a[i]`와 `b[i]` 간의 내적 연산입니다. 

__인자 설명__

- __axes__: 단일 Integer이거나 튜플 형태의 Integer이어야 하며, 내적 연산을 수행할 축 또는 축들을 의미합니다. 
- __normalize__: 내적 연산을 수행하기 전에 샘플들에 대해 L2 정규화(L2-normalize)를 수행할지에 대한 여부입니다. True로 설정하면, 내적 연산의 결과는 두 샘플 간의 코싸인 유사도와 같습니다. 
- __**kwargs__: 표준 레이어 키워드 인자입니다. 
    
----

### add


```python
keras.layers.add(inputs)
```


`Add` 레이어에 대해 함수처럼 제공되는 인터페이스입니다.

__인자 설명__

- __inputs__: 입력 텐서에 대한 리스트입니다. 최소 2개 이상의 텐서를 포함해야 합니다.
- __**kwargs__: 표준 레이어 키워드 인자입니다. 

__반환값__

입력에 대해 덧셈 연산이 수행된 단일 텐서를 반환합니다. 

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
    
----

### subtract


```python
keras.layers.subtract(inputs)
```


`Subtract` 레이어에 대해 함수처럼 제공되는 인터페이스입니다.

__인자 설명__

- __inputs__: 입력 텐서에 대한 리스트입니다. (정확히 2개의 텐서이어야 합니다.)
- __**kwargs__: 표준 레이어 키워드 인자입니다. 

__반환값__

입력되는 두 개의 텐서에 대한 차이값이 단일 텐서로 반환됩니다. 

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
    
----

### multiply


```python
keras.layers.multiply(inputs)
```


`Multiply` 레이어에 대해 함수처럼 제공되는 인터페이스입니다.

__인자 설명__

- __inputs__: 입력 텐서에 대한 리스트입니다. 최소 2개 이상의 텐서를 포함해야 합니다.
- __**kwargs__: 표준 레이어 키워드 인자입니다.

__반환값__

입력 텐서들에 대한 요소별 연산 결과가 단일 텐서로 반환됩니다. 
    
----

### average


```python
keras.layers.average(inputs)
```


`Average` 레이어에 대해 함수처럼 제공되는 인터페이스입니다.

__인자 설명__

- __inputs__: 입력 텐서에 대한 리스트입니다. 최소 2개 이상의 텐서를 포함해야 합니다.
- __**kwargs__: 표준 레이어 키워드 인자입니다.

__반환값__

입력 텐서들에 대한 평균값 연산 결과가 단일 텐서로 반환됩니다. 
    
----

### maximum


```python
keras.layers.maximum(inputs)
```


`Maximum` 레이어에 대해 함수처럼 제공되는 인터페이스입니다.

__인자 설명__

- __inputs__: 입력 텐서에 대한 리스트입니다. 최소 2개 이상의 텐서를 포함해야 합니다.
- __**kwargs__: 표준 레이어 키워드 인자입니다.

__반환값__

입력 텐서들에 대해 입력 단위로 최댓값 연산을 수행된 결과가 단일 텐서로 반환됩니다. 
    
----

### concatenate


```python
keras.layers.concatenate(inputs, axis=-1)
```


`Concatenate` 레이어에 대해 함수처럼 제공되는 인터페이스입니다.

__인자 설명__

- __inputs__: 입력 텐서에 대한 리스트입니다. 최소 2개 이상의 텐서를 포함해야 합니다.
- __axis__: 이어붙이기(Concatenation)를 수행할 축입니다. 
- __**kwargs__: 표준 레이어 키워드 인자입니다.

__반환값__

입력들에 대해 축 `axis`를 통해 이어붙이기(concatenation)를 수행한 결과를 단일 텐서로 반환합니다. 
    
----

### dot


```python
keras.layers.dot(inputs, axes, normalize=False)
```


`Dot` 레이어에 대해 함수처럼 제공되는 인터페이스입니다. 

__인자 설명__

- __inputs__: 입력 텐서에 대한 리스트입니다. 최소 2개 이상의 텐서를 포함해야 합니다.
- __axes__: 단일 Integer이거나 튜플 형태의 Integer이어야 하며, 내적 연산을 수행할 축 또는 축들을 의미합니다. 
- __normalize__: 내적 연산을 수행하기 전에 샘플들에 대해 L2 정규화(L2-normalize)를 수행할지에 대한 여부입니다. True로 설정하면, 내적 연산의 결과는 두 샘플 간의 코싸인 유사도와 같습니다. 
- __**kwargs__: 표준 레이어 키워드 인자입니다.

__반환값__

입력의 샘플에 대해 내적 연산을 수행한 결과가 단일 텐서로 반환됩니다. 
    

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L69)</span>
### MaxPooling1D

```python
keras.layers.MaxPooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last')
```

시계열 자료형에 대한 최대값 풀링<sub>Max Pooling</sub>.

__인수__

- __pool_size__: `int`, 최대값 풀링 창<sub>Window</sub>의 크기.
- __strides__: `int`, 혹은 `None`. 차원을 축소할 정도.
    예: 2는 입력 값을 반으로 줄입니다.
    `None`일 경우, 기본값으로 `pool_size`을 사용합니다.  
- __padding__: `"valid"` 또는 `"same"`(대소문자 무시).
- __data_format__: `string`,
    `channels_last`(기본값) 혹은 `channels_first`.
    입력 인수의 순서.
    `channels_last`는 `(batch, steps, features)`, `channels_first`는 `(batch, features, steps)` 형태를 의미합니다.

__입력값 형태__

- `data_format='channels_last'`이면
    `(batch_size, steps, features)`
    형태의 3D 텐서<sub>Tensor</sub>.
- `data_format='channels_first'`이면
    `(batch_size, features, steps)`
    형태의 3D 텐서.

__출력값 형태__

- `data_format='channels_last'`이면
    `(batch_size, downsampled_steps, features)`
    형태의 3D 텐서.
- `data_format='channels_first'`이면
    `(batch_size, features, downsampled_steps)`
    형태의 3D 텐서.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L217)</span>
### MaxPooling2D

```python
keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
```

공간 자료형에 대한 최대값 풀링.

__인수__

- __pool_size__: `int` 혹은 2개의 `int`이루어진 튜플,
    (가로, 세로)의 차원을 축소할 정도.
    예: (2, 2)는 입력값을 두 차원에서 반으로 축소합니다.
    `int` 하나만 설정된 경우, 두 차원에 동일한 창 크기를 사용합니다.
- __strides__: `int`, `int` 2개로 이루어진 튜플, 혹은 `None`. 스트라이드.
    `None`일 경우, 기본값으로 `pool_size`를 사용합니다.
- __padding__: `"valid"` 혹은 `"same"`(대소문자 무시).
- __data_format__: `string`,
    `channels_last` (기본값) 혹은 `channels_first`.
    입력값의 형태.
    `channels_last`는 `(batch, height, width, channels)`, `channels_first`는
    `(batch, channels, height, width)` 형태를 의미합니다.
    기본 설정은 `~/.keras/keras.json`의
    `image_data_format`에서 설정할 수 있습니다.
    따로 변경하지 않으면, 기본 설정은 `"channels_last"`입니다.

__입력값 형태__

- `data_format='channels_last'`이면
    `(batch_size, rows, cols, channels)`
    형태의 4D 텐서.
- `data_format='channels_first'`이면
    `(batch_size, channels, rows, cols)`
    형태의 4D 텐서.

__출력값 형태__

- `data_format='channels_last'`이면
    `(batch_size, pooled_rows, pooled_cols, channels)`
    형태의 4D 텐서.
- `data_format='channels_first'`이면
    `(batch_size, channels, pooled_rows, pooled_cols)`
    형태의 4D 텐서.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L386)</span>
### MaxPooling3D

```python
keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=None, padding='valid', data_format=None)
```

(공간 혹은 시공간) 3D 자료형에 대한 최대값 풀링.

__인수__

- __pool_size__: 3개의 `int`로 이루어진 튜플,
    (dim1, dim2, dim3)의 차원을 축소할 정도.
    예: (2, 2, 2)는 3D 입력값을 각 차원에서 반으로 축소합니다.
- __strides__: `int`, 3개의 `int`로 이루어진 튜플, 혹은 `None`. 스트라이드.
- __padding__: `"valid"` 혹은 `"same"`(대소문자 무시).
- __data_format__:`string`,
    `channels_last` (기본값) 혹은 `channels_first`.
    입력 인수의 순서. `channels_last`는
    `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
    , `channels_first`는
    `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`
    형태를 의미합니다.
    기본 설정은 `~/.keras/keras.json`의
    `image_data_format`에서 설정할 수 있습니다.
    따로 변경하지 않으면, 기본 설정은 `"channels_last"`입니다.

__입력값 형태__

- `data_format='channels_last'`이면
    `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
    형태의 5D 텐서.
- `data_format='channels_first'`이면
    `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`
    형태의 5D 텐서.

__출력값 형태__

- `data_format='channels_last'`이면
    `(batch_size, pooled_dim1, pooled_dim2, pooled_dim3, channels)`
    형태의 5D 텐서.
- `data_format='channels_first'`이면
    `(batch_size, channels, pooled_dim1, pooled_dim2, pooled_dim3)`
    형태의 5D 텐서.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L117)</span>
### AveragePooling1D

```python
keras.layers.AveragePooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last')
```

시계열 자료형에 대한 평균 풀링.

__인수__

- __pool_size__: `int`, 평균 풀링 창의 크기.
- __strides__: `int`, 혹은 `None`. 차원을 축소할 정도.
    예: 2는 입력값을 반으로 축소합니다.
    `None`일 경우, 기본값인 `pool_size`을 사용합니다.
- __padding__: `"valid"` 혹은 `"same"`(대소문자 무시).
- __data_format__: `string`,
    `channels_last`(기본값) 혹은 `channels_first`.
    입력값의 형태.
    `channels_last`는 `(batch, steps, features)`, `channels_first`는
    `(batch, features, steps)` 형태를 의미합니다.

__입력값 형태__

- `data_format='channels_last'`이면
    `(batch_size, steps, features)`
    형태의 3D 텐서.
- `data_format='channels_first'`이면
    `(batch_size, features, steps)`
    형태의 3D 텐서.

__출력값 형태__

- `data_format='channels_last'`이면
    `(batch_size, downsampled_steps, features)`
    형태의 3D 텐서.
- `data_format='channels_first'`이면
    `(batch_size, features, downsampled_steps)`
    형태의 3D 텐서.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L272)</span>
### AveragePooling2D

```python
keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
```

공간 자료형에 대한 평균 풀링.

__인수__
- __pool_size__: `int` 혹은 2개의 `int`이루어진 튜플,
    (가로, 세로)의 차원을 축소할 정도.
    예: (2, 2)는 입력값을 두 차원 모두에서 반으로 축소합니다.
    `int` 하나만 설정된 경우, 두 차원에 동일한 창 크기를 사용합니다.
- __strides__: `int`, `int` 2개로 이루어진 튜플, 혹은 `None`. 스트라이드
    `None`일 경우, 기본값으로 `pool_size`를 사용합니다.
- __padding__: `"valid"` 혹은 `"same"`(대소문자 무시).
- __data_format__: `string`,
    `channels_last` (기본값) 혹은 `channels_first`.
    입력 인수의 형태.
    `channels_last`는 `(batch, height, width, channels)`, `channels_first`는
    `(batch, channels, height, width)` 형태를 의미합니다.
    기본 설정은 `~/.keras/keras.json`의
    `image_data_format`에서 설정할 수 있습니다.
    따로 변경하지 않으면, `"channels_last"`입니다.

__입력값 형태__

- `data_format='channels_last'`이면
    `(batch_size, rows, cols, channels)`
    형태의 4D 텐서.
- `data_format='channels_first'`이면
    `(batch_size, channels, rows, cols)`
    형태의 4D 텐서.

__출력값 형태__

- `data_format='channels_last'`이면
    `(batch_size, pooled_rows, pooled_cols, channels)`
    형태의 4D 텐서.
- `data_format='channels_first'`이면
    `(batch_size, channels, pooled_rows, pooled_cols)`
    형태의 4D 텐서.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L436)</span>
### AveragePooling3D

```python
keras.layers.AveragePooling3D(pool_size=(2, 2, 2), strides=None, padding='valid', data_format=None)
```

(공간 혹은 시공간) 3D 자료형에 대한 평균 풀링.

__인수__

- __pool_size__: 3개의 `int`로 이루어진 튜플,
    (dim1, dim2, dim3)의 차원을 축소할 정도.
    예: (2, 2, 2)는 3D 입력값을 각 차원에서 반으로 축소합니다.
- __strides__: `int`, 3개의 `int`로 이루어진 튜플, 혹은 `None`. 스트라이드.
- __padding__: `"valid"` 혹은 `"same"`(대소문자 무시).
- __data_format__: `string`,
    `channels_last` (기본값) 혹은 `channels_first`.
    입력값의 형태. `channels_last`는
    `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`, `channels_first`는
    `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`
    형태를 의미합니다.
    기본 설정은 `~/.keras/keras.json`의 `image_data_format`에서 설정할 수 있습니다.
    따로 변경하지 않으면, `"channels_last"`입니다.

__입력값 형태__

- `data_format='channels_last'`이면
    `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
    형태의 5D 텐서.
- `data_format='channels_first'`이면
    `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`
    형태의 5D 텐서.

__출력값 형태__

- `data_format='channels_last'`이면
    `(batch_size, pooled_dim1, pooled_dim2, pooled_dim3, channels)`
    형태의 5D 텐서.
- `data_format='channels_first'`이면
    `(batch_size, channels, pooled_dim1, pooled_dim2, pooled_dim3)`
    형태의 5D 텐서.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L557)</span>
### GlobalMaxPooling1D

```python
keras.layers.GlobalMaxPooling1D(data_format='channels_last')
```

시계열 자료형에 대한 전역 최대값 풀링.

__인수__

- __data_format__: `string`,
    `channels_last` (기본값) 혹은 `channels_first`.
    입력값의 형태.
    `channels_last`는 `(batch, steps, features)`, `channels_first`는
    `(batch, features, steps)` 형태를 의미합니다.

__입력값 형태__

- `data_format='channels_last'`이면
    `(batch_size, steps, features)`
    형태의 3D 텐서.
- `data_format='channels_first'`이면
    `(batch_size, features, steps)`
    형태의 3D 텐서.

__출력값 형태__

`(batch_size, features)`
형태의 2D 텐서.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L511)</span>
### GlobalAveragePooling1D

```python
keras.layers.GlobalAveragePooling1D(data_format='channels_last')
```

Global average pooling operation for temporal data.

__인수__

- __data_format__: `string`,
    `channels_last` (기본값) 혹은 `channels_first`.
    입력값의 형태.
    `channels_last`는 `(batch, steps, features)`, `channels_first`는
    `(batch, features, steps)` 형태를 의미합니다.

__입력값 형태__

- `data_format='channels_last'`이면
    `(batch_size, steps, features)`
    형태의 3D 텐서.
- `data_format='channels_first'`이면
    `(batch_size, features, steps)`
    형태의 3D 텐서.

__출력값 형태__

`(batch_size, features)`
형태의 2D 텐서.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L647)</span>
### GlobalMaxPooling2D

```python
keras.layers.GlobalMaxPooling2D(data_format=None)
```

공간 자료형에 대한 전역 최대값 풀링.

__인수__

- __data_format__: `string`,
    `channels_last` (기본값) 혹은 `channels_first`.
    입력값의 형태.
    `channels_last`는 `(batch, height, width, channels)`, `channels_first`는
    `(batch, channels, height, width)` 형태를 의미합니다.
    기본 설정은 `~/.keras/keras.json`의 `image_data_format`에서 설정할 수 있습니다.
    따로 변경하지 않으면, `"channels_last"`입니다.

__입력값 형태__

- `data_format='channels_last'`이면
    `(batch_size, rows, cols, channels)`
    형태의 4D 텐서.
- `data_format='channels_first'`이면
    `(batch_size, channels, rows, cols)`
    형태의 4D 텐서.

__출력값 형태__

`(batch_size, channels)`
형태의 2D 텐서.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L612)</span>
### GlobalAveragePooling2D

```python
keras.layers.GlobalAveragePooling2D(data_format=None)
```

공간 자료형에 대한 전역 평균 풀링.

__인수__

- __data_format__: `string`,
    `channels_last` (기본값) 혹은 `channels_first`.
    입력값의 형태.
    `channels_last`는 `(batch, height, width, channels)`, `channels_first`는
    `(batch, channels, height, width)` 형태를 의미합니다.
    기본 설정은 `~/.keras/keras.json`의 `image_data_format`에서 설정할 수 있습니다.
    따로 변경하지 않으면, `"channels_last"`입니다.

__입력값 형태__

- `data_format='channels_last'`이면
    `(batch_size, rows, cols, channels)`
    형태의 4D 텐서.
- `data_format='channels_first'`이면
    `(batch_size, channels, rows, cols)`
    형태의 4D 텐서.

__출력값 형태__

`(batch_size, channels)`
형태의 2D 텐서.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L742)</span>
### GlobalMaxPooling3D

```python
keras.layers.GlobalMaxPooling3D(data_format=None)
```

3D 데이터에 대한 전역 최대값 풀링

__인수__

- __data_format__: `string`,
    `channels_last` (기본값) 혹은 `channels_first`.
    입력값의 형태. `channels_last`는
    `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`, `channels_first`는
    `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)` 형태를 의미합니다.
    기본 설정은 `~/.keras/keras.json`의 `image_data_format`에서 설정할 수 있습니다.
    따로 변경하지 않으면, `"channels_last"`입니다.

__입력값 형태__

- `data_format='channels_last'`이면
    `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
    형태의 5D 텐서.
- `data_format='channels_first'`이면
    `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`
    형태의 5D 텐서.

__출력값 형태__

`(batch_size, channels)`
형태의 2D 텐서.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L707)</span>
### GlobalAveragePooling3D

```python
keras.layers.GlobalAveragePooling3D(data_format=None)
```

3D 자료형에 대한 전역 평균 풀링

__인수__

- __data_format__: `string`,
    `channels_last` (기본값) 혹은 `channels_first`.
    입력값의 형태. `channels_last`는
    `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`,
    `channels_first`는 `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`
    형태를 의미합니다.
    기본 설정은 `~/.keras/keras.json`의 `image_data_format`에서 설정할 수 있습니다.
    따로 변경하지 않으면, `"channels_last"`입니다.

__입력값 형태__

- `data_format='channels_last'`이면
    `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
    형태의 5D 텐서.
- `data_format='channels_first'`이면
    `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`
    형태의 5D 텐서.

__출력값 형태__

`(batch_size, channels)`
형태의 2D 텐서.
    

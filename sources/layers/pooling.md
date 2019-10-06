<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L69)</span>
### MaxPooling1D

```python
keras.layers.MaxPooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last')
```

시계열 데이터에 대한 최대값 풀링<sub>Max Pooling</sub>.

__인수__

- __pool_size__: `int`, 최대값 풀링 창의 크기.
- __strides__: `int`, 혹은 `None`. 차원을 줄이는 정도.
    예: 2는 입력 값을 반으로 줄입니다.
    `None`일 경우, 기본값으로 `pool_size`을 사용합니다.
- __padding__: `"valid"`, `"same"` 중 하나 (대소문자 무시).
- __data_format__: `string`,
    `channels_last`(기본값) 혹은 `channels_first`.
    입력값의 데이터 순서.
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

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L217)</span>
### MaxPooling2D

```python
keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
```

공간적 데이터에 대한 최대값 풀링 작업.

__인수__

- __pool_size__: 정수 혹은 2개 정수의 튜플,
    축소 인수 (가로, 세로).
    (2, 2)는 인풋을 두 공간 차원에 대해 반으로 축소합니다.
    한 정수만 특정된 경우, 동일한 윈도우 길이가
    두 차원 모두에 대해 적용됩니다.
- __strides__: 정수, 2개 정수의 튜플, 혹은 None.
    보폭 값.
    None일 경우, 디폴트 값인 `pool_size`가 됩니다.
- __padding__: `"valid"` 혹은 `"same"` 중 하나 (대소문자 무시).
- __data_format__: 문자열,
    `channels_last` (디폴트 값) 혹은 `channels_first` 중 하나.
    인풋 차원의 순서.
    `channels_last`는 `(batch, height, width, channels)` 형태의
    인풋에 호응하고, `channels_first`는
    `(batch, channels, height, width)` 형태의
    인풋에 호응합니다.
    디폴트 값은 `~/.keras/keras.json`에 위치한
    케라스 구성 파일의 `image_data_format` 값으로 지정됩니다.
    따로 설정하지 않으면, 이는 `"channels_last"`가 됩니다.

__인풋 형태__

- `data_format='channels_last'`이면
    `(batch_size, rows, cols, channels)`
    형태의 4D 텐서.
- `data_format='channels_first'`이면
    `(batch_size, channels, rows, cols)`
    형태의 4D 텐서.

__아웃풋 형태__

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

(공간적 혹은 시공간적) 3D 데이터에 대한 최대값 풀링 작업.

__인수__

- __pool_size__: 3개 정수의 튜플,
    축소 인수 (dim1, dim2, dim3).
    (2, 2, 2)는 3D 인풋을 각 공간 차원에 대해 반으로 축소합니다.
- __strides__: 정수, 3개 정수의 튜플, 혹은 None. 보폭 값.
- __padding__: `"valid"` 혹은 `"same"` 중 하나 (대소문자 무시).
- __data_format__: 문자열,
    `channels_last` (디폴트 값) 혹은 `channels_first` 중 하나.
    인풋 차원의 순서. `channels_last`는
    `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
    형태의 인풋에 호응하고, `channels_first`는
    `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`
    형태의 인풋에 호응합니다.
    디폴트 값은 `~/.keras/keras.json`에 위치한
    케라스 구성 파일의 `image_data_format` 값으로 지정됩니다.
    따로 설정하지 않으면, 이는 `"channels_last"`가 됩니다.

__인풋 형태__

- `data_format='channels_last'`이면
    `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
    형태의 5D 텐서.
- `data_format='channels_first'`이면
    `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`
    형태의 5D 텐서.

__아웃풋 형태__

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

시간적 데이터에 대한 평균 풀링.

__인수__

- __pool_size__: 정수, 평균 풀링 위도우의 크기.
- __strides__: 정수, 혹은 None. 축소 인수.
    예. 2는 인풋을 반으로 축소합니다.
    None일 경우, 디폴트 값인 `pool_size`가 됩니다.
- __padding__: `"valid"` 혹은 `"same"` 중 하나 (대소문자 무시).
- __data_format__: 문자열,
    `channels_last` (디폴트 값) 혹은 `channels_first` 중 하나.
    인풋 차원의 순서.
    `channels_last`는 `(batch, steps, features)` 형태의
    인풋에 호응하고, `channels_first`는
    `(batch, features, steps)` 형태의
    인풋에 호응합니다.

__인풋 형태__

- `data_format='channels_last'`이면
    `(batch_size, steps, features)`
    형태의 3D 텐서.
- `data_format='channels_first'`이면
    `(batch_size, features, steps)`
    형태의 3D 텐서.

__아웃풋 형태__

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

공간적 데이터에 대한 평균 풀링 작업.

__인수__

- __pool_size__: 정수 혹은 2개 정수의 튜플,
    축소 인수 (가로, 세로).
    (2, 2)는 인풋을 두 공간 차원에 대해 반으로 축소합니다.
    한 정수만 특정된 경우, 동일한 윈도우 길이가
    두 차원 모두에 대해 적용됩니다.
- __strides__: 정수, 2개 정수의 튜플, 혹은 None.
    보폭 값.
    None일 경우, 디폴트 값인 `pool_size`가 됩니다.
- __padding__: `"valid"` 혹은 `"same"` 중 하나 (대소문자 무시).
- __data_format__: 문자열,
    `channels_last` (디폴트 값) 혹은 `channels_first` 중 하나.
    인풋 차원의 순서.
    `channels_last`는 `(batch, height, width, channels)` 형태의
    인풋에 호응하고, `channels_first`는
    `(batch, channels, height, width)` 형태의
    인풋에 호응합니다.
    디폴트 값은 `~/.keras/keras.json`에 위치한
    케라스 구성 파일의 `image_data_format` 값으로 지정됩니다.
    따로 설정하지 않으면, 이는 `"channels_last"`가 됩니다.

__인풋 형태__

- `data_format='channels_last'`이면
    `(batch_size, rows, cols, channels)`
    형태의 4D 텐서.
- `data_format='channels_first'`이면
    `(batch_size, channels, rows, cols)`
    형태의 4D 텐서.

__아웃풋 형태__

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

(공간적 혹은 시공간적) 3D 데이터에 대한 평균 풀링 작업.

__인수__

- __pool_size__: 3개 정수의 튜플,
    축소 인수 (dim1, dim2, dim3).
    (2, 2, 2)는 3D 인풋을 각 공간 차원에 대해 반으로 축소합니다.
- __strides__: 정수, 3개 정수의 튜플, 혹은 None. 보폭 값.
- __padding__: `"valid"` 혹은 `"same"` 중 하나 (대소문자 무시).
- __data_format__: 문자열,
    `channels_last` (디폴트 값) 혹은 `channels_first` 중 하나.
    인풋 차원의 순서. `channels_last`는
    `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
    형태의 인풋에 호응하고, `channels_first`는
    `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`
    형태의 인풋에 호응합니다.
    디폴트 값은 `~/.keras/keras.json`에 위치한
    케라스 구성 파일의 `image_data_format` 값으로 지정됩니다.
    따로 설정하지 않으면, 이는 `"channels_last"`가 됩니다.

__인풋 형태__

- `data_format='channels_last'`이면
    `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
    형태의 5D 텐서.
- `data_format='channels_first'`이면
    `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`
    형태의 5D 텐서.

__아웃풋 형태__

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

시간적 데이터에 대한 글로벌 최대값 풀링 작업.

__인수__

- __data_format__: 문자열,
    `channels_last` (디폴트 값) 혹은 `channels_first` 중 하나.
    인풋 차원의 순서.
    `channels_last`는 `(batch, steps, features)` 형태의
    인풋에 호응하고, `channels_first`는
    `(batch, features, steps)` 형태의
    인풋에 호응합니다.

__인풋 형태__

- `data_format='channels_last'`이면
    `(batch_size, steps, features)`
    형태의 3D 텐서.
- `data_format='channels_first'`이면
    `(batch_size, features, steps)`
    형태의 3D 텐서.

__아웃풋 형태__

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

- __data_format__: 문자열,
    `channels_last` (디폴트 값) 혹은 `channels_first` 중 하나.
    인풋 차원의 순서.
    `channels_last`는 `(batch, steps, features)` 형태의
    인풋에 호응하고, `channels_first`는
    `(batch, features, steps)` 형태의
    인풋에 호응합니다.

__인풋 형태__

- `data_format='channels_last'`이면
    `(batch_size, steps, features)`
    형태의 3D 텐서.
- `data_format='channels_first'`이면
    `(batch_size, features, steps)`
    형태의 3D 텐서.

__아웃풋 형태__

`(batch_size, features)`
형태의 2D 텐서.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L647)</span>
### GlobalMaxPooling2D

```python
keras.layers.GlobalMaxPooling2D(data_format=None)
```

공간적 데이터에 대한 글로벌 최대값 풀링 작업

__인수__

- __data_format__: 문자열,
    `channels_last` (디폴트 값) 혹은 `channels_first` 중 하나.
    인풋 차원의 순서.
    `channels_last`는 `(batch, height, width, channels)` 형태의
    인풋에 호응하고, `channels_first`는
    `(batch, channels, height, width)` 형태의
    인풋에 호응합니다.
    디폴트 값은 `~/.keras/keras.json`에 위치한
    케라스 구성 파일의 `image_data_format` 값으로 지정됩니다.
    따로 설정하지 않으면, 이는 `"channels_last"`가 됩니다.

__인풋 형태__

- `data_format='channels_last'`이면
    `(batch_size, rows, cols, channels)`
    형태의 4D 텐서.
- `data_format='channels_first'`이면
    `(batch_size, channels, rows, cols)`
    형태의 4D 텐서.

__아웃풋 형태__

`(batch_size, channels)`
형태의 2D 텐서.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L612)</span>
### GlobalAveragePooling2D

```python
keras.layers.GlobalAveragePooling2D(data_format=None)
```

공간적 데이터에 대한 글로벌 평균 풀링 작업.

__인수__

- __data_format__: 문자열,
    `channels_last` (디폴트 값) 혹은 `channels_first` 중 하나.
    인풋 차원의 순서.
    `channels_last`는 `(batch, height, width, channels)` 형태의
    인풋에 호응하고, `channels_first`는
    `(batch, channels, height, width)` 형태의
    인풋에 호응합니다.
    디폴트 값은 `~/.keras/keras.json`에 위치한
    케라스 구성 파일의 `image_data_format` 값으로 지정됩니다.
    따로 설정하지 않으면, 이는 `"channels_last"`가 됩니다.

__인풋 형태__

- `data_format='channels_last'`이면
    `(batch_size, rows, cols, channels)`
    형태의 4D 텐서.
- `data_format='channels_first'`이면
    `(batch_size, channels, rows, cols)`
    형태의 4D 텐서.

__아웃풋 형태__

`(batch_size, channels)`
형태의 2D 텐서.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L742)</span>
### GlobalMaxPooling3D

```python
keras.layers.GlobalMaxPooling3D(data_format=None)
```

3D 데이터에 대한 글로벌 최대값 풀링

__인수__

- __data_format__: 문자열,
    `channels_last` (디폴트 값) 혹은 `channels_first` 중 하나.
    인풋 차원의 순서. `channels_last`는
    `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
    형태의 인풋에 호응하고, `channels_first`는
    `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`
    형태의 인풋에 호응합니다.
    디폴트 값은 `~/.keras/keras.json`에 위치한
    케라스 구성 파일의 `image_data_format` 값으로 지정됩니다.
    따로 설정하지 않으면, 이는 `"channels_last"`가 됩니다.

__인풋 형태__

- `data_format='channels_last'`이면
    `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
    형태의 5D 텐서.
- `data_format='channels_first'`이면
    `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`
    형태의 5D 텐서.

__아웃풋 형태__

`(batch_size, channels)`
형태의 2D 텐서.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L707)</span>
### GlobalAveragePooling3D

```python
keras.layers.GlobalAveragePooling3D(data_format=None)
```

3D 데이터에 대한 글로벌 평균 풀링 작업

__인수__

- __data_format__: 문자열,
    `channels_last` (디폴트 값) 혹은 `channels_first` 중 하나.
    인풋 차원의 순서. `channels_last`는
    `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
    형태의 인풋에 호응하고, `channels_first`는
    `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`
    형태의 인풋에 호응합니다.
    디폴트 값은 `~/.keras/keras.json`에 위치한
    케라스 구성 파일의 `image_data_format` 값으로 지정됩니다.
    따로 설정하지 않으면, 이는 `"channels_last"`가 됩니다.

__인풋 형태__

- `data_format='channels_last'`이면
    `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
    형태의 5D 텐서.
- `data_format='channels_first'`이면
    `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`
    형태의 5D 텐서.

__아웃풋 형태__

`(batch_size, channels)`
형태의 2D 텐서.
    

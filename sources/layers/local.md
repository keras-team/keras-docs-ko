<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/local.py#L19)</span>
### LocallyConnected1D

```python
keras.layers.LocallyConnected1D(filters, kernel_size, strides=1, padding='valid', data_format=None, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

1D 인풋용 국소 연결 레이어.

`LocallyConnected1D` 레이어는 `Conv1D` 레이어와 비슷한 방식으로 작동하지만
가중치가 공유되지 않는다는 점에서 다른데,
이는 인풋의 각 부분에 각기 다른 필터 세트가 적용된다는 
의미입니다.

__예시__

```python
# 공유되지 않은 길이 3의 1D 컨볼루션 가중치를 10개의 시간 단계와
# 64개의 아웃풋 필터로 이루어진 시퀀스에 적용합니다.
model = Sequential()
model.add(LocallyConnected1D(64, 3, input_shape=(10, 32)))
# 현재 model.output_shape == (None, 8, 64)
# 그 위에 새로운 conv1d를 추가합니다
model.add(LocallyConnected1D(32, 3))
# 현재 model.output_shape == (None, 6, 32)
```

__인수__

- __filters__: 정수, 출력 공간의 차원
    (다시 말해 컨볼루션의 아웃풋 필터의 개수).
- __kernel_size__: 단일 정수 혹은 단일 정수의 튜플/리스트,
    1D 컨볼루션 윈도우의 길이를 특정합니다.
- __strides__: 단일 정수 혹은 단일 정수의 튜플/리스트,
    컨볼루션의 보폭 길이를 특정합니다.
    보폭 값이 1이 아니도록 특정하는 경우는 `dilation_rate`의 값이
    1이 아니도록 특정하는 경우와 양립할 수 없습니다.
- __padding__: 현재는 (대소문자 구분없이) `"valid"`만을 지원합니다.
    차후 `"same"`을 지원할 계획입니다.
- __data_format__: String, one of `channels_first`, `channels_last`.    
- __activation__: 사용할 활성화 함수
    ([활성화](../activations.md) 참조).
    따로 특정하지 않는 경우 활성화가 적용되지 않습니다
    (다시 말해 "선형적" 활성화: `a(x) = x`).
- __use_bias__: 불리언, 레이어가 편향 벡터를 사용하는지 여부.
- __kernel_initializer__: `kernel` 가중치 행렬용 초기값 설정기
    ([초기값 설정기](../initializers.md) 참조).
- __bias_initializer__: 편향 벡터용 초기값 설정기
    ([초기값 설정기](../initializers.md) 참조).
- __kernel_regularizer__: `kernel` 가중치 행렬에 적용되는
    정규화 함수 
    ([정규화](../regularizers.md) 참조).
- __bias_regularizer__: 편향 벡터에 적용되는 정규화 함수
    ([정규화](../regularizers.md) 참조).
- __activity_regularizer__: 레이어의 아웃풋(레이어의 "활성화")에
    적용되는 정규화 함수.
    ([정규화](../regularizers.md) 참조).
- __kernel_constraint__: 커널 행렬에 적용되는 제약 함수
    ([제약](../constraints.md) 참조).
- __bias_constraint__: 편향 벡터에 적용되는 제약 함수
    ([제약](../constraints.md) 참조).

__인풋 형태__

3D 텐서: `(batch_size, steps, input_dim)`의 형태

__아웃풋 형태__

3D 텐서: `(batch_size, new_steps, filters)`의 형태
패딩 혹은 보폭의 결과로 `steps` 값이 변했을 수도 있습니다.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/local.py#L183)</span>
### LocallyConnected2D

```python
keras.layers.LocallyConnected2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

2D 인풋용 국소 연결 레이어.

`LocallyConnected2D` 레이어는 `Conv2D` 레이어와 비슷한 방식으로 작동하지만
가중치가 공유되지 않는다는 점에서 다른데,
이는 인풋의 각 부분에 각기 다른 필터 세트가 적용된다는 
의미입니다.

__예시__

```python
# 64개의 아웃풋 필터와 더불어 3x3 비공유 가중치 컨볼루션을
# `data_format="channels_last"`으로 설정된 32x32 이미지에 적용합니다:
model = Sequential()
model.add(LocallyConnected2D(64, (3, 3), input_shape=(32, 32, 3)))
# 현재 model.output_shape == (None, 30, 30, 64)
# 이 레이어가 (30*30)*(3*3*3*64) + (30*30)*64개의
# 매개변수를 사용한다는 점을 유의하십시오

# 32개의 아웃풋 필터와 더불어 3x3 비공유 가중치 컨볼루션을 상부에 추가하십시오:
model.add(LocallyConnected2D(32, (3, 3)))
# 현재 model.output_shape == (None, 28, 28, 32)
```

__인수__

- __filters__: 정수, 아웃풋 공간의 차원
    (다시 말해 컨볼루션의 아웃풋 필터의 개수).
- __kernel_size__: 단일 정수, 혹은 2D 컨볼루션 윈도우의
    넓이와 높이를 특정하는 2개 정수의 튜플/리스트.
    단일 정수로는 모든 공간 차원에 대해서
    같은 값을 특정할 수 있습니다.
- __strides__: 단일 정수, 혹은 넓이와 높이에 따라
    컨볼루션 보폭을 특정하는 2개 정수의 튜플/리스트.
    단일 정수로는 모든 공간 차원에 대해서
    같은 값을 특정할 수 있습니다.
- __padding__: 현재는 (대소문자 구분없이) `"valid"`만을 지원합니다.
    차후 `"same"`을 지원할 계획입니다.
- __data_format__: 문자열,
    `channels_last` (디폴트 값) 혹은 `channels_first` 중 하나.
    인풋 내 차원의 순서.
    `channels_last`는 `(batch, height, width, channels)`의 형태를 갖는
    인풋에 호응하는 반면, `channels_first`는
    `(batch, channels, height, width)`의 형태를
    갖는 인풋에 호응합니다.
    디폴트 값은 `~/.keras/keras.json`에 위치한
    케라스 구성 파일 내 `image_data_format` 값으로 설정됩니다.
    따로 지정하지 않았다면, 이 값은 "channels_last"입니다.
- __activation__: 사용할 활성화 함수
    ([활성화](../activations.md) 참조).
    따로 특정하지 않는 경우 활성화가 적용되지 않습니다
    (다시 말해 "선형적" 활성화: `a(x) = x`).
- __use_bias__: 불리언, 레이어가 편향 벡터를 사용하는지 여부.
- __kernel_initializer__: `kernel` 가중치 행렬용 초기값 설정기
    ([초기값 설정기](../initializers.md) 참조).
- __bias_initializer__: 편향 벡터용 초기값 설정기
    ([초기값 설정기](../initializers.md) 참조).
- __kernel_regularizer__: `kernel` 가중치 행렬에 적용되는
    정규화 함수 
    ([정규화](../regularizers.md) 참조).
- __bias_regularizer__: 편향 벡터에 적용되는 정규화 함수
    ([정규화](../regularizers.md) 참조).
- __activity_regularizer__: 레이어의 아웃풋(레이어의 "활성화")에
    적용되는 정규화 함수.
    ([정규화](../regularizers.md) 참조).
- __kernel_constraint__: 커널 행렬에 적용되는 제약 함수
    ([제약](../constraints.md) 참조).
- __bias_constraint__: 편향 벡터에 적용되는 제약 함수
    ([제약](../constraints.md) 참조).

__인풋 형태__

4D 텐서:
data_format='channels_first'의 경우 `(samples, channels, rows, cols)`의 형태, 
혹은 data_format='channels_last'의 경우
`(samples, rows, cols, channels)`의 형태를 갖습니다.

__아웃풋 형태__

4D 텐서:
data_format='channels_first'의 경우 `(samples, filters, new_rows, new_cols)`의 형태
혹은 data_format='channels_last'의 경우
`(samples, new_rows, new_cols, filters)`의 형태를 갖습니다.
패딩의 결과로 `rows`와 `cols` 값이 달라졌을 수도 있습니다.
    

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/local.py#L19)</span>
### LocallyConnected1D

```python
keras.layers.LocallyConnected1D(filters, kernel_size, strides=1, padding='valid', data_format=None, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

입력값<sub>Input</sub>이 1D인 부분 연결<sub>Locally Connectded</sub> 층<sub>Layer</sub>.

`LocallyConnected1D` 층은 `Conv1D` 층과 비슷한 방식으로 작동하지만
가중치<sub>Weight</sub>가 공유되지 않는다는 점에서 다른데,
이는 입력값의 각 부분에 각기 다른 필터 세트가 적용된다는 
의미입니다.

__예시__

```python
# 공유되지 않은 길이 3의 1D 컨볼루션 가중치를 10개의 시간 단계와
# 64개의 출력 필터로 이루어진 시퀀스에 적용합니다.
model = Sequential()
model.add(LocallyConnected1D(64, 3, input_shape=(10, 32)))
# 현재 model.output_shape == (None, 8, 64)
# 그 위에 새로운 conv1d를 추가합니다
model.add(LocallyConnected1D(32, 3))
# 현재 model.output_shape == (None, 6, 32)
```

__인자__

- __filters__: 정수, 출력 공간의 차원
    (다시 말해 컨볼루션의 출력값 필터의 개수).
- __kernel_size__: 단일 정수 혹은 단일 정수의 튜플/리스트,
    1D 컨볼루션 창<sub>Window</sub>의 길이를 특정합니다.
- __strides__: 단일 정수 혹은 단일 정수의 튜플/리스트,
    컨볼루션의 스트라이드 길이를 특정합니다.
    스트라이드 값이 1이 아니도록 특정하는 경우는 `dilation_rate`의 값이
    1이 아니도록 특정하는 경우와 양립할 수 없습니다.
- __padding__: 현재는 (대소문자 구분없이) `"valid"`만을 지원합니다.
    차후 `"same"`을 지원할 계획입니다.
- __data_format__: String, one of `channels_first`, `channels_last`.    
- __activation__: 사용할 활성화 함수<sub>Activation</sub>
    ([활성화](../activations.md) 참조).
    따로 특정하지 않는 경우 활성화가 적용되지 않습니다
    (다시 말해 "선형적" 활성화: `a(x) = x`).
- __use_bias__: 불리언, 층에서 편향<sub>Bias</sub> 벡터를 사용하는지 여부.
- __kernel_initializer__: `kernel` 가중치 행렬의 초기화 함수
    ([초기화 함수](../initializers.md) 참조).
- __bias_initializer__: 편향 벡터의 초기화 함수
    ([초기 함수](../initializers.md) 참조).
- __kernel_regularizer__: `kernel` 가중치 행렬에 적용되는
    규제 함수<sub>Regularizer</sub>
    ([정규화](../regularizers.md) 참조).
- __bias_regularizer__: 편향 벡터에 적용되는 규제 함수
    ([정규화](../regularizers.md) 참조).
- __activity_regularizer__: 층의 출력값(층의 "활성화")에
    적용되는 규제 함수.
    ([정규화](../regularizers.md) 참조).
- __kernel_constraint__: 커널 행렬에 적용되는 제약<sub>Constraint</sub>
    ([제약](../constraints.md) 참조).
- __bias_constraint__: 편향 벡터에 적용되는 제약
    ([제약](../constraints.md) 참조).

__입력값 형태__

3D 텐서<sub>Tensor</sub>: `(batch_size, steps, input_dim)`의 형태

__출력값 형태__

3D 텐서: `(batch_size, new_steps, filters)`의 형태
패딩 혹은 스트라이드의 결과로 `steps` 값이 변했을 수도 있습니다.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/local.py#L183)</span>
### LocallyConnected2D

```python
keras.layers.LocallyConnected2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

입력값이 2D인 부분 연결 층.

`LocallyConnected2D` 층은 `Conv2D` 층과 비슷한 방식으로 작동하지만
가중치가 공유되지 않는다는 점에서 다른데,
이는 입력값의 각 부분에 각기 다른 필터 세트가 적용된다는 
의미입니다.

__예시__

```python
# 64개의 출력 필터와 더불어 3x3 비공유 가중치 컨볼루션을
# `data_format="channels_last"`으로 설정된 32x32 이미지에 적용합니다:
model = Sequential()
model.add(LocallyConnected2D(64, (3, 3), input_shape=(32, 32, 3)))
# 현재 model.output_shape == (None, 30, 30, 64)
# 이 층이 (30*30)*(3*3*3*64) + (30*30)*64개의
# 매개변수를 사용한다는 점을 유의하십시오

# 32개의 출력 필터와 더불어 3x3 비공유 가중치 컨볼루션을 상부에 추가하십시오:
model.add(LocallyConnected2D(32, (3, 3)))
# 현재 model.output_shape == (None, 28, 28, 32)
```

__인자__

- __filters__: 정수, 출력 공간의 차원
    (다시 말해 컨볼루션의 출력 필터의 개수).
- __kernel_size__: 단일 정수, 혹은 2D 컨볼루션 창의
    넓이와 높이를 특정하는 2개 정수의 튜플/리스트.
    단일 정수로는 모든 공간 차원에 대해서
    같은 값을 특정할 수 있습니다.
- __strides__: 단일 정수, 혹은 넓이와 높이에 따라
    컨볼루션 스트라이드를 특정하는 2개 정수의 튜플/리스트.
    단일 정수로는 모든 공간 차원에 대해서
    같은 값을 특정할 수 있습니다.
- __padding__: 현재는 (대소문자 구분없이) `"valid"`만을 지원합니다.
    차후 `"same"`을 지원할 계획입니다.
- __data_format__: 문자열,
    `channels_last` (기본값) 혹은 `channels_first`.
    입력값의 형태.
    `channels_last`는 `(batch, height, width, channels)`, `channels_first`는
    `(batch, channels, height, width)`의 형태를 의미합니다.
    기본 설정은 `~/.keras/keras.json`의 `image_data_format` 값에서 설정할 수 있습니다.
    따로 변경하지 않았다면, "channels_last"입니다.
- __activation__: 사용할 활성화 함수
    ([활성화](../activations.md) 참조).
    따로 특정하지 않는 경우 활성화가 적용되지 않습니다
    (다시 말해 "선형적" 활성화: `a(x) = x`).
- __use_bias__: 불리언, 층이 편향 벡터를 사용하는지 여부.
- __kernel_initializer__: `kernel` 가중치 행렬의 초기화 함수
    ([초기화 함수](../initializers.md) 참조).
- __bias_initializer__: 편향 벡터의 초기화 함수
    ([초기화 함수](../initializers.md) 참조).
- __kernel_regularizer__: `kernel` 가중치 행렬에 적용되는
    규제 함수 
    ([규제 함수](../regularizers.md) 참조).
- __bias_regularizer__: 편향 벡터에 적용되는 정규화 함수
    ([규제 함수](../regularizers.md) 참조).
- __activity_regularizer__: 층의 출력값(층의 "활성화")에
    적용되는 규제 함수.
    ([규제 함수](../regularizers.md) 참조).
- __kernel_constraint__: 커널 행렬에 적용되는 제약
    ([제약](../constraints.md) 참조).
- __bias_constraint__: 편향 벡터에 적용되는 제약
    ([제약](../constraints.md) 참조).

__입력값 형태__

4D 텐서:
data_format='channels_first'의 경우 `(samples, channels, rows, cols)`, 
data_format='channels_last'의 경우 `(samples, rows, cols, channels)`의 형태를 갖습니다.

__출력 형태__

4D 텐서:
data_format='channels_first'의 경우 `(samples, filters, new_rows, new_cols)`
data_format='channels_last'의 경우 `(samples, new_rows, new_cols, filters)`의 형태를 갖습니다.
패딩의 결과로 `rows`와 `cols` 값이 달라졌을 수도 있습니다.

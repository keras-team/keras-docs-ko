<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/wrappers.py#L116)</span>
### TimeDistributed

```python
keras.layers.TimeDistributed(layer)
```

이 래퍼는 입력 값의 모든 시계열 단위에 층<sub>layer</sub>을 적용합니다. 

입력 값은 최소 3D의 형태를 가져야 하며, 그 중 index 1(두번째 원소)은 시간 차원을 나타냅니다.

32개의 표본으로 구성된 배치에서, 10개의 16차원 벡터로 이루어진 시퀀스<sub>sequence</sub>의 예시를 보겠습니다.
이 예시에서 층의 배치 입력 형태는 `(32, 10, 16)`이고,
배치 차원을 제외한 `input_shape`는 `(10, 16)`입니다.

`TimeDistributed`를 사용하여 10개의 시간 단계<sub>timestamps</sub>에 독립적으로 `Dense` 층을 적용할 수 있습니다:

```python
# 모델의 첫 번째 층으로 사용되었습니다.
model = Sequential()
model.add(TimeDistributed(Dense(8), input_shape=(10, 16)))
# model.output_shape == (None, 10, 8)
```

이 결과 출력 값은 `(32, 10, 8)`의 형태를 가집니다.

첫 번째 층 이후에는 `input_shape`를 명시할 필요가 없습니다.

```python
model.add(TimeDistributed(Dense(32)))
# now model.output_shape == (None, 10, 32)
```

이 결과 출력 값은 `(32, 10, 32)`의 형태를 가집니다.

`TimeDistributed`는 'Dense' 가 아닌 케라스에서 사용 가능한 층에서도 사용할 수 있습니다.
(ex:`Conv2D` 층)

```python
model = Sequential()
model.add(TimeDistributed(Conv2D(64, (3, 3)),
                          input_shape=(10, 299, 299, 3)))
```

__인자__

- __layer__: 층 인스턴스.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/wrappers.py#L335)</span>
### Bidirectional

```python
keras.engine.base_layer.wrapped_fn()
```

순환 신경망에 사용하는 양방향 래퍼.

__인자__

- __layer__: `Recurrent` 인스턴스.
- __merge_mode__: 정방향 순환 신경망과
    역방향 순환 신경망의 출력 값이 병합되는 방식을 {'sum', 'mul', 'concat', 'ave', None} 중에 선택합니다.
    None의 경우, 출력 값이 합쳐지지 않고, 리스트로 반환됩니다.
- __weights__: Bidirectional 모델에 사용 될 초기 가중치

__오류처리__

- __ValueError__: `merge_mode` 인자가 유효하지 않은 경우 오류 메시지를 전달합니다.

__예시__

```python
model = Sequential()
model.add(Bidirectional(LSTM(10, return_sequences=True),
                        input_shape=(5, 10)))
model.add(Bidirectional(LSTM(10)))
model.add(Dense(5))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
```

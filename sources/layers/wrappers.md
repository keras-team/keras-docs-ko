<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/wrappers.py#L116)</span>
### TimeDistributed

```python
keras.layers.TimeDistributed(layer)
```

이 래퍼는 입력 값의 모든 시계열 단위에 레이어를 적용합니다. 

32개의 샘플로 구성된 배치에서, 10개의 16차원 벡터로 이루어진 시퀀스의 예시를 보겠습니다.
이 예시에서 레이어의 배치 입력 형태는 `(32, 10, 16)`이고,
배치 차원을 제외한 `input_shape`는 `(10, 16)`입니다.

`TimeDistributed`를 사용하여 10개의 시간 단계에 독립적으로 `Dense` 레이어를 적용할 수 있습니다:

```python
# 모델의 첫 번째 레이어로 사용되었습니다.
model = Sequential()
model.add(TimeDistributed(Dense(8), input_shape=(10, 16)))
# model.output_shape == (None, 10, 8)
```

이 결과 아웃풋은 `(32, 10, 8)`의 형태를 가집니다.

첫 번째 레이어 이후에는 `input_shape`를 명시할 필요가 없습니다.

```python
model.add(TimeDistributed(Dense(32)))
# now model.output_shape == (None, 10, 32)
```

이 결과 아웃풋은 `(32, 10, 32)`의 형태를 가집니다.

`TimeDistributed`는 'Dense' 가 아닌 케라스에서 사용 가능한 레이어에서도 사용할 수 있습니다.
(ex:`Conv2D` 레이어)

```python
model = Sequential()
model.add(TimeDistributed(Conv2D(64, (3, 3)),
                          input_shape=(10, 299, 299, 3)))
```

__인수__

- __layer__: 레이어 인스턴스.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/wrappers.py#L335)</span>
### Bidirectional

```python
keras.engine.base_layer.wrapped_fn()
```

순환 신경망에 사용하는 양방향 래퍼.

__인수__

- __layer__: `Recurrent` 인스턴스.
- __merge_mode__: 정방향 순환 신경망과
    역방향 순환 신경망의 아웃풋이 병합되는 방식.
    {'sum', 'mul', 'concat', 'ave', None} 중 하나를 인수로 받습니다.
    None의 경우, 아웃풋이 합쳐지지 않고, 리스트로 반환됩니다.
-__weights__: Bidirectional 모델에 사용 될 초기 가중치

__오류처리__

- __ValueError__: `merge_mode` 인수가 유효하지 않은 경우 오류메시지를 전달합니다.

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
    

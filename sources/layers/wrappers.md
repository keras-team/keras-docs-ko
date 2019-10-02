<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/wrappers.py#L116)</span>
### TimeDistributed

```python
keras.layers.TimeDistributed(layer)
```

이 래퍼는 인풋의 모든 시간적 조각에 대해 레이어를 적용합니다.

인풋은 적어도 3D 이상이어야 하며, 그 중 색인 1의 차원은
시간 차원으로 인식합니다.

32개 샘플로 구성된 배치에서,
각 샘플은 10개의 16차원 벡터로 이루어진 시퀀스라고 가정합니다.
그렇다면 레이어의 배치 인풋 형태는 `(32, 10, 16)`이고,
샘플 차원을 제외한 `input_shape`은 `(10, 16)`이 됩니다.

이어서 `TimeDistributed`를 사용해 10개의 시간 단계 각각에
독립적으로 `Dense` 레이어를 적용할 수 있습니다:

```python
# 모델의 첫 번째 레이어로써
model = Sequential()
model.add(TimeDistributed(Dense(8), input_shape=(10, 16)))
# 현재 model.output_shape == (None, 10, 8)
```

이 결과 아웃풋은 `(32, 10, 8)`의 형태를 갖습니다.

차우 레이어에서는 `input_shape`이 필요없습니다:

```python
model.add(TimeDistributed(Dense(32)))
# now model.output_shape == (None, 10, 32)
```

이 결과 아웃풋은 `(32, 10, 32)`의 형태를 갖습니다.

`TimeDistributed`는 `Dense`만이 아닌, 예를 들면 `Conv2D` 레이어와 같은
임의의 레이어와 함께 사용할 수 있습니다:

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
    {'sum', 'mul', 'concat', 'ave', None} 중 하나.
    None의 경우, 아웃풋이 합쳐지지 않고,
    리스트로 반환됩니다.  
- __weights__: 'Bidirectional' 모델에 사용 될 초기 가중치

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
    

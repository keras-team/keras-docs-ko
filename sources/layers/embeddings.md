<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/embeddings.py#L16)</span>
### 임베딩

```python
keras.layers.Embedding(input_dim, output_dim, embeddings_initializer='uniform', embeddings_regularizer=None, activity_regularizer=None, embeddings_constraint=None, mask_zero=False, input_length=None)
```

양의 정수(색인)를 고정된 크기의 밀집 벡터로 전환합니다.
예. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]

이 레이어는 모델의 첫 번째 레이어로만 사용할 수 있습니다.

__예시__


```python
model = Sequential()
model.add(Embedding(1000, 64, input_length=10))
# 모델은 (batch, input_length)의 크기를 가진 정수 행렬만 인풋으로 전달받습니다.
# 인풋 내 가장 큰 정수(다시 말해 단어 색인)는
# 999(어휘목록 사이즈)보다 커서는 안됩니다.
# 현재 model.output_shape == (None, 10, 64), 여기서 배치 차원은 None입니다.

input_array = np.random.randint(1000, size=(32, 10))

model.compile('rmsprop', 'mse')
output_array = model.predict(input_array)
assert output_array.shape == (32, 10, 64)
```

__인수__

- __input_dim__: 정수 > 0. 어휘목록의 크기,
    다시 말해, 최대 정수 색인 + 1.
- __output_dim__: 정수 >= 0. 밀집 임베딩의 차원.
- __embeddings_initializer__: `embeddings` 행렬의 초기값 설정기
    ([초기값 설정기](../initializers.md) 참조).
- __embeddings_regularizer__: `embeddings` 행렬에 적용되는
    정규화 함수 
    ([정규화](../regularizers.md) 참조).
- __embeddings_constraint__: `embeddings`행렬에 적용되는
    제약 함수
    ([제약](../constraints.md) 참조).
- __mask_zero__: 인풋 값 0을 마스크 처리해야 할 특수 "패딩"값으로
    다룰 것인지 여부.
    [순환 레이어](recurrent.md)를 사용해 가변적
    길이의 인풋을 전달받을 때 유용합니다.
    이 값이 `True`인 경우 모델 내 차후 레이어는
    마스킹을 지원해야 하며 그렇지 않은 경우 예외가 발생됩니다.
    mask_zero가 참으로 설정된 경우, 색인 0은 
    어휘목록에서 사용할 수 없습니다 (input_dim이 vocabulary + 1과
    동일한 크기를 가져야 하기 때문입니다).
- __input_length__: 인풋 시퀀스의 길이로, 그 길이가 불변해야 합니다.
    상위 `Dense` 레이어와 `Flatten` 레이어를 연결하려면
    이 인수가 필요합니다   
    (그렇지 않고서는 밀집 아웃풋의 형태를 계산할 수 없습니다).

__인풋 형태__

2D 텐서: `(batch_size, sequence_length)`의 형태.

__아웃풋 형태__

3D 텐서: `(batch_size, sequence_length, output_dim)`의 형태.

__참조__

- [A Theoretically Grounded Application of Dropout in
   Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
    

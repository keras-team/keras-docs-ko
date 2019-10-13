<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/embeddings.py#L16)</span>
### Embedding

```python
keras.layers.Embedding(input_dim, output_dim, embeddings_initializer='uniform', embeddings_regularizer=None, activity_regularizer=None, embeddings_constraint=None, mask_zero=False, input_length=None)
```

양의 정수(인덱스)를 고정된 크기의 벡터로 변환합니다.
예: [[4], [20]] → [[0.25, 0.1], [0.6, -0.2]]

이 층<sub>layer</sub>은 모델의 첫 번째 층으로만 사용할 수 있습니다.

__예시__


```python
model = Sequential()
model.add(Embedding(1000, 64, input_length=10))
# Embedding은 (batch, input_length)의 형태를 가진 정수 행렬을 입력으로 받습니다.
# 입력 내 가장 큰 정수(단어 인덱스)는
# 999(어휘목록 사이즈)보다 커서는 안됩니다.
# 이 때 model.output_shape == (None, 10, 64)이며, 여기서 None은 배치 차원입니다.

input_array = np.random.randint(1000, size=(32, 10))

model.compile('rmsprop', 'mse')
output_array = model.predict(input_array)
assert output_array.shape == (32, 10, 64)
```

__인자__

- __input_dim__: `int > 0`. 어휘 목록의 크기를 나타내며, 최대 정수 인덱스+1의 값을 가집니다.  
- __output_dim__: `int >= 0`. 임베딩 벡터의 차원을 나타냅니다.
- __embeddings_initializer__: 임베딩 행렬(임베딩 벡터로 이루어진 `(input_dim, ouput_dim)`형태의 행렬)의 초기화 함수([초기화 함수](../initializers.md) 참조).
- __embeddings_regularizer__: 임베딩 행렬에 적용되는 규제화 함수([규제화 함수](../regularizers.md) 참조).
- __activity_regularizer__: 층의 출력값에 적용되는 규제화 함수. 자세한 내용은 아래의 [논문](https://arxiv.org/abs/1708.01009)을 참조하십시오.
- __embeddings_constraint__: 임베딩 벡터에 적용되는 제약 함수([제약 함수](../constraints.md) 참조).
- __mask_zero__: 입력값 0을 마스크 처리해야 할 특수 "패딩<sub>padding</sub>"값으로
    다룰 것인지를 결정합니다.
    [순환 층](recurrent.md)를 사용해 가변
    길이의 입력을 전달받을 때 유용합니다.
    이 값이 `True`인 경우 모델 내에서 이후의 층은
    마스킹을 지원해야 하며, 그렇지 않은 경우 예외가 발생됩니다.
    또한, 색인 0은 어휘목록에서 사용할 수 없습니다(`input_dim`이 '어휘 목록 크기 + 1'과
    동일한 크기를 가져야 하기 때문입니다).
- __input_length__: 입력 시퀀스의 길이로, 그 길이가 불변해야 합니다.
    상위 `Dense` 층과 `Flatten` 층를 연결하려면
    이 인자가 필요합니다(이 인자가 주어지지 않은 경우, 밀집 출력의 형태를 계산할 수 없습니다).

__입력 형태__

`(batch_size, sequence_length)`의 형태의 2D 텐서

__출력 형태__

`(batch_size, sequence_length, output_dim)`의 형태의 3D 텐서

__참조__

- [A Theoretically Grounded Application of Dropout in
   Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
- [Revisiting Activation Regularization for Language RNNs](https://arxiv.org/abs/1708.01009)
    

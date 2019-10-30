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
# 이 때 model.output_shape == (None, 10, 64)이며, None은 배치 차원입니다.

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
- __mask_zero__: 0을 패딩<sub>paddng</sub>값으로 사용하였는지 여부를 알려줍니다.
입력값의 길이가 다양한 경우의 순환 신경망<sub>RNN</sub>에서 유용하게 사용됩니다.
이 값이 `True`인 경우 이후 사용하는 모든 층들은 마스킹을 지원해야 하며, 0은 패딩값으로 사용되기 때문에
단어 인덱스로는 사용할 수 없습니다. 
- __input_length__: 입력값의 시계열(순서형) 길이. 모든 배치가 단일한 길이를 가지는 경우 하나의 상수값을 배정합니다. 이후 `Flatten`에 이어 `Dense`층을 연결하려면 이 인자가 필요합니다(이 인자가 주어지지 않은 경우, `Dense`층의 출력을 계산할 수 없습니다).

__입력 형태__

`(batch_size, sequence_length)`의 형태의 2D 텐서.

__출력 형태__

`(batch_size, sequence_length, output_dim)`의 형태의 3D 텐서.

__참조__

- [A Theoretically Grounded Application of Dropout in
   Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
- [Revisiting Activation Regularization for Language RNNs](https://arxiv.org/abs/1708.01009)
    

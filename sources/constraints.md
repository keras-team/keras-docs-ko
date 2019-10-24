## 제약의 사용법

`constraints` 모듈의 함수는 최적화 과정에서 네트워크 매개변수에 제약(예시. 비음수)을 설정할 수 있도록 합니다.

페널티는 각 레이어 별로 적용됩니다. API의 구체사항은 레이어마다 다를 수 있지만, `Dense`, `Conv1D`, `Conv2D` 그리고 `Conv3D` 레이어는 통일된 API를 가집니다.

이러한 레이어들은 2가지 키워드 인수를 노출합니다:

- 주요 가중치 행렬에 대한 `kernel_constraint`
- 편향에 대한 `bias_constraint`


```python
from keras.constraints import max_norm
model.add(Dense(64, kernel_constraint=max_norm(2.)))
```

---

## 사용가능한 제약

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/constraints.py#L22)</span>
### MaxNorm

```python
keras.constraints.MaxNorm(max_value=2, axis=0)
```

MaxNorm 가중치 제약.

각 히든 레이어에 대응하는 가중치를 제약해서
가중치의 노름이 특정 값 이하가 되도록 합니다.

__인수__

- __max_value__: 입력 가중치의 최대 노름
- __axis__: 정수, 가중치 노름을 계산할 축.
    예를 들어, 어느 `Dense` 레이어의 가중치 행렬이
    `(input_dim, output_dim)`의 형태를 취할 때,
    `axis`를 `0`으로 설정해서 `(input_dim,)`의 길이를 갖는
    각 가중치 벡터를 제약할 수 있습니다.
    `data_format="channels_last"`의 데이터 포맷을 갖는 `Conv2D`레이어의 경우,
    가중치 텐서는
    `(rows, cols, input_depth, output_depth)`의 형태를 가지며,
    `axis`를 `[0, 1, 2]`로 설정하여
    `(rows, cols, input_depth)`의 형태를 갖는
    각 필터 텐서의 가중치를 제약할 수 있습니다.

__참고__

- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](
   http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/constraints.py#L61)</span>
### NonNeg

```python
keras.constraints.NonNeg()
```

가중치가 비음수가 되도록 제약합니다.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/constraints.py#L69)</span>
### UnitNorm

```python
keras.constraints.UnitNorm(axis=0)
```

각 히든 레이어 유닛에 대응하는 가중치가 단위 노름을 가지도록 제약합니다.

__인수__

- __axis__: 정수,가중치 노름을 계산할 축.
    예를 들어, 어느 `Dense` 레이어의 가중치 행렬이
    `(input_dim, output_dim)`의 형태를 취할 때,
    `axis`를 `0`으로 설정해서 `(input_dim,)`의 길이를 갖는
    각 가중치 벡터를 제약할 수 있습니다.
    `data_format='channels_last'`의 데이터 포맷을 갖는 `Conv2D`레이어의 경우,
    가중치 텐서는
    `(rows, cols, input_depth, output_depth)`의 형태를 가지며,
    `axis`를 `[0, 1, 2]`로 설정하여
    `(rows, cols, input_depth)`의 형태를 갖는
    각 필터 텐서의 가중치를 제약할 수 있습니다.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/constraints.py#L98)</span>
### MinMaxNorm

```python
keras.constraints.MinMaxNorm(min_value=0.0, max_value=1.0, rate=1.0, axis=0)
```

MinMaxNorm 가중치 제약.

각 히든 레이어에 대응하는 가중치를 제약해서
가중치의 노름이 상한과 하한 사이의 값을 가지도록 합니다.

__인수__

- __min_value__: 입력 가중치의 최소 노름.
- __max_value__: 입력 가중치의 최대 노름.
- __rate__: 제약을 시행하는 속도:
    가중치를 리스케일하여
    `(1 - rate) * norm + rate * norm.clip(min_value, max_value)`의 값을 산출하도록 합니다.
    이는 실질적으로 rate=1.0의 경우 제약을 엄격하게 실행함을 의미하고
    실행함을 의미하고, 반대로 rate<1.0의 경우
    매 단계마다 가중치가 리스케일되어
    원하는 간격 사이의 값에 천천히 가까워지도록 함을 말합니다. 
- __axis__: 정수, 가중치 노름을 계산할 축.
    예를 들어, 어느 `Dense` 레이어의 가중치 행렬이
    `(input_dim, output_dim)`의 형태를 취할 때,
    `axis`를 `0`으로 설정해서 `(input_dim,)`의 길이를 갖는
    각 가중치 벡터를 제약할 수 있습니다.
    `data_format='channels_last'`의 데이터 포맷을 갖는 `Conv2D`레이어의 경우,
    가중치 텐서는
    `(rows, cols, input_depth, output_depth)`의 형태를 가지며,
    `axis`를 `[0, 1, 2]`로 설정하여
    `(rows, cols, input_depth)`의 형태를 갖는
    각 필터 텐서의 가중치를 제약할 수 있습니다.
    

---


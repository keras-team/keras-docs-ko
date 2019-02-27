## Regularizer의 사용법

Regularizer는 최적화 과정 중에 각 층별 파라미터 또는 출력값에 대하여 페널티를 적용할 수 있게 해줍니다. 이러한 페널티는 네트워크가 최적화 하려는 손실 함수의 일부로 포함됩니다.

페널티는 층별로 다르게 적용될 수 있습니다. 정확한 API는 층마다 서로 다르지만, `Dense`, `Conv1D`, `Conv2D` 그리고 `Conv3D`는 동일한 API를 가지고 있습니다.

이 층들은 다음 세 개의 키워드 인자들을 공통적으로 가지고 있습니다.

- `kernel_regularizer`: `keras.regularizers.Regularizer`의 객체입니다.
- `bias_regularizer`: `keras.regularizers.Regularizer`의 객체입니다.
- `activity_regularizer`: `keras.regularizers.Regularizer`의 객체입니다.


## 예제

```python
from keras import regularizers
model.add(Dense(64, input_dim=64,
                kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))
```

## 사용 가능한 페널티

```python
keras.regularizers.l1(0.)
keras.regularizers.l2(0.)
keras.regularizers.l1_l2(l1=0.01, l2=0.01)
```

## 새로운 regularizer를 개발하려면

아래와 같이 가중치 행렬을 입력으로 받고, 그에 해당하는 페널티를 계산해 반환하는 함수라면 모두 regularizer로 사용할 수 있습니다.

```python
from keras import backend as K

def l1_reg(weight_matrix):
    return 0.01 * K.sum(K.abs(weight_matrix))

model.add(Dense(64, input_dim=64,
                kernel_regularizer=l1_reg))
```

좀 더 객체 지향적인 방법을 원한다면 [여기](https://github.com/keras-team/keras/blob/master/keras/regularizers.py)에서 예시를 확인할 수 있습니다.

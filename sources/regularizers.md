## Regularizer 의 용법

Regularizer는 최적화 과정 중에 계층 파라메터 또는 계층 활성에 대하여 페널티를 적용할 수 있게 해 줍니다. 이러한 페널티는 네트워크가 최적화 하려는 손실 함수의 일부로 포함됩니다.

페널티는 계층 마다 적용될 수 있습니다. 정확한 API는 계층마다 다르지만, `Dense`, `Conv1D`, `Conv2D` 그리고 `Conv3D`는 통일화된 API를 가지고 있습니다.

이 계층들은 다음 세 개의 주요 인자를 노출시킵니다:

- `kernel_regularizer`: `keras.regularizers.Regularizer`의 인스턴스 입니다.
- `bias_regularizer`: `keras.regularizers.Regularizer`의 인스턴스 입니다.
- `activity_regularizer`: `keras.regularizers.Regularizer`의 인스턴스 입니다.


## 예제

```python
from keras import regularizers
model.add(Dense(64, input_dim=64,
                kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))
```

## 가용한 페널티

```python
keras.regularizers.l1(0.)
keras.regularizers.l2(0.)
keras.regularizers.l1_l2(l1=0.01, l2=0.01)
```

## 새로운 regularizer를 개발하는 것

가중치 행렬을 받아들여서 손실 기여 tensor를 반환하기만 하면 어떤 함수든지 regularizer로서 사용될 수 있습니다. 예를 들어서:

```python
from keras import backend as K

def l1_reg(weight_matrix):
    return 0.01 * K.sum(K.abs(weight_matrix))

model.add(Dense(64, input_dim=64,
                kernel_regularizer=l1_reg))
```

또는 대신에, 객체-지향적인 방법으로 여러분만의 regularizer를 작성할 수도 있습니다;
그 예를 위해서 [keras/regularizers.py](https://github.com/keras-team/keras/blob/master/keras/regularizers.py) 모듈을 살펴보시기 바랍니다.

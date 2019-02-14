## 정규화 기법의 사용법

정규화 기법은 최적화 과정 중 레이어 매개변수나 레이어 활성화 과정에 페널티를 적용할 수 있도록 합니다. 이러한 페널티는 네트워크가 최적화하는 손실 함수에 포함됩니다.

페널티는 각 레이어 별로 적용됩니다. API의 구체사항은 레이어에 따라 변하지만, `Dense`, `Conv1D`, `Conv2D` 그리고 `Conv3D`와 같은 레이어는 통일된 API를 갖습니다.

이 레이어들은 3가지 인자 키워드를 노출합니다:

- `kernel_regularizer`: `keras.regularizers.Regularizer`의 인스텐스
- `bias_regularizer`: `keras.regularizers.Regularizer`의 인스텐스
- `activity_regularizer`: `keras.regularizers.Regularizer`의 인스텐스


## 예시

```python
from keras import regularizers
model.add(Dense(64, input_dim=64,
                kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))
```

## 사용가능한 페널티

```python
keras.regularizers.l1(0.)
keras.regularizers.l2(0.)
keras.regularizers.l1_l2(l1=0.01, l2=0.01)
```

## 새로운 정규화 기법 개발

가중치 행렬을 전달받고 손실 기여도 텐서를 반환하는 어느 함수나 정규화 기법으로 사용할 수 있습니다. 예시:

```python
from keras import backend as K

def l1_reg(weight_matrix):
    return 0.01 * K.sum(K.abs(weight_matrix))

model.add(Dense(64, input_dim=64,
                kernel_regularizer=l1_reg))
```

혹은, 객체 지향적 방식으로 본인의 최적화 기법을 직접 작성할 수 있습니다;
예시를 보려면 [keras/regularizers.py](https://github.com/keras-team/keras/blob/master/keras/regularizers.py) 모듈을 참고하십시오.

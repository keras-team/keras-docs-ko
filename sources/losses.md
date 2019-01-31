
## 손실 함수의 사용

손실 함수 (목적 함수 또는 최적화 점수 함수)는 하나의 모델을 구성하기 위해 필요한 두 개의 매개 변수 중 하나 입니다:

```python
model.compile(loss='mean_squared_error', optimizer='sgd')
```

```python
from keras import losses

model.compile(loss=losses.mean_squared_error, optimizer='sgd')
```

기존의 손실 함수를 이름으로 전달하거나 TensorFlow/Theano 의 심볼릭 함수를 매개 변수로 전달할 수 있습니다. 심볼릭 함수는 다음의 두 인자를 받아 각각의 데이터 포인트에 대하여 스칼라를 반환합니다:

- __y_true__: 정답 라벨. TensorFlow/Theano 텐서.
- __y_pred__: 예측치. y_true와 같은 모양(shape)의 TensorFlow/Theano 텐서.

실제로 최적화 되는  모든 데이터 포인트에 걸친 출력 배열의 평균값입니다.

손실함수의 예시는 [losses source](https://github.com/keras-team/keras/blob/master/keras/losses.py)에서 확인할 수 있습니다.

## 사용 가능한 손실 함수

### mean_squared_error


```python
keras.losses.mean_squared_error(y_true, y_pred)
```

----

### mean_absolute_error


```python
keras.losses.mean_absolute_error(y_true, y_pred)
```

----

### mean_absolute_percentage_error


```python
keras.losses.mean_absolute_percentage_error(y_true, y_pred)
```

----

### mean_squared_logarithmic_error


```python
keras.losses.mean_squared_logarithmic_error(y_true, y_pred)
```

----

### squared_hinge


```python
keras.losses.squared_hinge(y_true, y_pred)
```

----

### hinge


```python
keras.losses.hinge(y_true, y_pred)
```

----

### categorical_hinge


```python
keras.losses.categorical_hinge(y_true, y_pred)
```

----

### logcosh


```python
keras.losses.logcosh(y_true, y_pred)
```


예측 오차의 쌍곡 코사인 로그값.

`log(cosh(x))`는 작은 `x`에 대하여 `(x ** 2) / 2`와, 큰 `x` 에 대하여
 `abs(x) - log(2)` 와 거의 같은 값을 가집니다. 다시 말해 'logcosh'는 대부분 
평균 제곱 오차와 비슷한 양상을 보이지만, 가끔 발생하는 부정확한 예측에 의한 영향을
크게 받지는 않습니다.

__Arguments__

- __y_true__: 실제 타겟의 텐서.
- __y_pred__: 예측 타겟의 텐서.

__Returns__

샘플당 하나의 스칼라값 손실 개체를 가지는 텐서
    
----

### categorical_crossentropy


```python
keras.losses.categorical_crossentropy(y_true, y_pred)
```

----

### sparse_categorical_crossentropy


```python
keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
```

----

### binary_crossentropy


```python
keras.losses.binary_crossentropy(y_true, y_pred)
```

----

### kullback_leibler_divergence


```python
keras.losses.kullback_leibler_divergence(y_true, y_pred)
```

----

### poisson


```python
keras.losses.poisson(y_true, y_pred)
```

----

### cosine_proximity


```python
keras.losses.cosine_proximity(y_true, y_pred)
```


----

**Note**: `categorical_crossentropy` 손실 함수의 경우 사용되는 타겟들은 범주 형식 (categorical format)을 따라야 합니다. 예를 들어 10개의 클래스(범주) 중 하나에 속하는 데이터에 대하여 각 샘플은 타겟 클래스에 해당하는 하나의 인덱스만 1의 값을 가지고 이외의 값들은 모두 0이어야 합니다. Keras의 기능인 `to_categorical`를 통해 *정수 타겟*을 *범주 타겟*으로 변환할 수 있습니다:

```python
from keras.utils import to_categorical

categorical_labels = to_categorical(int_labels, num_classes=None)
```

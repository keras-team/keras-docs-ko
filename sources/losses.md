
## 손실 함수의 사용

손실 함수(목적 함수 또는 최적화 스코어 함수)는 모델을 컴파일하기 위해 필요한 두 개의 매개 변수 중 하나입니다.

```python
model.compile(loss='mean_squared_error', optimizer='sgd')
```

```python
from keras import losses

model.compile(loss=losses.mean_squared_error, optimizer='sgd')
```

케라스가 제공하는 손실 함수의 이름 문자열<sub>string</sub> 또는 TensorFlow/Theano의 심볼릭 함수<sub>symbolic function</sub>를 매개 변수로 전달할 수 있습니다. 심볼릭 함수는 다음의 두 인자를 받아 각각의 데이터 포인트에 대한 스칼라를 반환합니다.

- __y_true__: 정답 레이블. TensorFlow/Theano 텐서.
- __y_pred__: 예측값. `y_true`와 같은 크기<sub>shape</sub>의 TensorFlow/Theano 텐서.

실제로 최적화되는 값은 모든 데이터 포인트에서의 출력값의 평균값입니다.

손실 함수의 예시는 [여기](https://github.com/keras-team/keras/blob/master/keras/losses.py)에서 확인할 수 있습니다.


## 사용 가능한 손실 함수

### mean_squared_error


```python
keras.losses.mean_squared_error(y_true, y_pred)
```


예측값과 목표값의 평균 제곱 오차<sub>(MSE, mean squared error)</sub>를 계산합니다.  
`mean((square(y_pred - y_true)))` 

__인자__

- __y_true__: 목표값 텐서.
- __y_pred__: 예측값 텐서.

__반환값__

샘플당 하나의 스칼라 손실값 텐서.

----

### mean_absolute_error


```python
keras.losses.mean_absolute_error(y_true, y_pred)
```


예측값과 목표값의 평균 절대 오차<sub>(MAE, mean absolute error)</sub>를 계산합니다.  
`mean(abs(y_pred - y_true))` 

__인자__

- __y_true__: 목표값 텐서.
- __y_pred__: 예측값 텐서.

__반환값__

샘플당 하나의 스칼라 손실값 텐서.

----

### mean_absolute_percentage_error


```python
keras.losses.mean_absolute_percentage_error(y_true, y_pred)
```


예측값과 목표값의 평균 절대 퍼센트 오차<sub>(MAPE, mean absolute percentage error)</sub>를 계산합니다.    
`100.*mean((abs(y_pred - y_true)))` 

__인자__

- __y_true__: 목표값 텐서.
- __y_pred__: 예측값 텐서.

__반환값__

샘플당 하나의 스칼라 손실값 텐서.

----

### mean_squared_logarithmic_error


```python
keras.losses.mean_squared_logarithmic_error(y_true, y_pred)
```


예측값과 목표값의 평균 제곱 로그 오차<sub>(MSLE, mean squared logarithmic error)</sub>를 계산합니다.   
`mean(square(log(y_pred + 1) - log(y_true + 1)))`

__인자__

- __y_true__: 목표값 텐서.
- __y_pred__: 예측값 텐서.

__반환값__

샘플당 하나의 스칼라 손실값 텐서.

----

### squared_hinge


```python
keras.losses.squared_hinge(y_true, y_pred)
```


예측값과 목표값의 'squared hinge' 손실값을 계산합니다.  
`mean(square(maximum(1 - y_true * y_pred, 0)))`

__인자__

- __y_true__: 목표값 텐서.
- __y_pred__: 예측값 텐서.

__반환값__

샘플당 하나의 스칼라 손실값 텐서.

----

### hinge


```python
keras.losses.hinge(y_true, y_pred)
```


예측값과 목표값의 'hinge' 손실값을 계산합니다.  
`mean(maximum(1 - y_true * y_pred, 0))`

__인자__

- __y_true__: 목표값 텐서.
- __y_pred__: 예측값 텐서.

__반환값__

샘플당 하나의 스칼라 손실값 텐서.

----

### categorical_hinge


```python
keras.losses.categorical_hinge(y_true, y_pred)
```


예측값과 목표값의 'categorical hinge' 손실값을 계산합니다.  
`maximum(0, max((1 - y_true) * y_pred) - sum(y_true * y_pred) + 1)`

__인자__

- __y_true__: 목표값 텐서.
- __y_pred__: 예측값 텐서.

__반환값__

샘플당 하나의 스칼라 손실값 텐서.

----

### logcosh


```python
keras.losses.logcosh(y_true, y_pred)
```


예측 오차의 하이퍼볼릭 코사인 로그값.

`log(cosh(x))`는 `x`가 작은 경우에는 `(x ** 2) / 2`, `x`가 큰 경우에는
 `abs(x) - log(2)`와 거의 같은 값을 가집니다. 다시 말해 `logcosh`는 대부분 
평균 제곱 오차와 비슷한 양상을 보이지만, 가끔 예측이 틀리더라도 영향을 크게
받지 않습니다.

__인자__

- __y_true__: 목표값 텐서.
- __y_pred__: 예측값 텐서.

__반환값__

샘플당 하나의 스칼라 손실값 텐서.
    
----

### categorical_crossentropy


```python
keras.losses.categorical_crossentropy(y_true, y_pred)
```

예측값과 목표값 사이의 크로스 엔트로피값을 계산합니다.  
입/출력은 원-핫 인코딩<one-hot encoding> 형태를 가져야 합니다.
 

__인자__

- __y_true__: 목표값 텐서.
- __y_pred__: 예측값 텐서.

__반환값__

샘플당 하나의 스칼라 손실값 텐서.

----

### sparse_categorical_crossentropy

```python
keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
```

예측값과 목표값 사이의 크로스 엔트로피값을 계산합니다.  
입/출력은 `int` 형태를 가져야 합니다.

__인자__

- __y_true__: 목표값 텐서.
- __y_pred__: 예측값 텐서.

__반환값__

샘플당 하나의 스칼라 손실값 텐서.

----

### binary_crossentropy


```python
keras.losses.binary_crossentropy(y_true, y_pred)
```

예측값과 목표값 사이의 크로스 엔트로피값을 계산합니다.  
입/출력은 이진<sub>binary</sub> 형태를 가져야 합니다.


__인자__

- __y_true__: 목표값 텐서.
- __y_pred__: 예측값 텐서.

__반환값__

샘플당 하나의 스칼라 손실값 텐서.

----

### kullback_leibler_divergence


```python
keras.losses.kullback_leibler_divergence(y_true, y_pred)
```

예측값과 목표값 사이의 KL 발산<sub>kullback Leibler divergence</sub> 값을 계산합니다.

`sum(y_true * log(y_true / y_pred))`

__인자__

- __y_true__: 목표값 텐서.
- __y_pred__: 예측값 텐서.

__반환값__

샘플당 하나의 스칼라 손실값 텐서.

----

### poisson

```python
keras.losses.poisson(y_true, y_pred)
```
예측값과 목표값 사이의 포아송 손실값<sub>poisson loss</sub>을 계산합니다.  
목표값이 포아송 분포를 따른다고 생각될 때 사용합니다.

`mean(y_pred - y_true * log(y_pred + epsilon()))`

__인자__

- __y_true__: 목표값 텐서.
- __y_pred__: 예측값 텐서.

__반환값__

샘플당 하나의 스칼라 손실값 텐서.

----

### cosine_proximity


```python
keras.losses.cosine_proximity(y_true, y_pred)
```

예측값과 목표값 사이의 코사인 유사도<sub>cosine proximity</sub> 값을 계산합니다.

__인자__

- __y_true__: 목표값 텐서.
- __y_pred__: 예측값 텐서.

__반환값__

샘플당 하나의 스칼라 손실값 텐서.

----

### is_categorical_crossentropy

```python
kearas.losses.is_categorical_crossentropy(loss)
```

----
**Note**: 손실 함수 `categorical_crossentropy`의 목표값은 범주 형식<sub>categorical format</sub>을 따라야 합니다. 예를 들어 범주(클래스)가 10개라면, 각 샘플의 목표값은 샘플 클래스에 해당하는 인덱스에서는 1이고 나머지 인덱스에서는 0인 10차원 벡터가 되어야 합니다.
케라스의 `to_categorical`을 통해 정수형 목표값(*integer target*)을 범주형 목표값(*categorical target*)으로 변환할 수 있습니다.

```python
from keras.utils import to_categorical

categorical_labels = to_categorical(int_labels, num_classes=None)
```

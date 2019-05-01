
## 측정항목의 사용법

측정항목은 모델의 성능을 평가하는데 사용되는 함수입니다. 측정항목 함수는 모델이 컴파일 될 때 `metrics` 매개변수를 통해 공급됩니다.

```python
model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['mae', 'acc'])
```

```python
from keras import metrics

model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=[metrics.mae, metrics.categorical_accuracy])
```

측정항목 함수는 [loss function](/losses)와 비슷하지만, 측정항목을 평가한 결과는 모델을 학습시키는데 사용되지 않는다는 점에서 다릅니다. 어느 손실 함수나 측정항목 함수로 사용할 수 있습니다.

기존의 측정항목의 이름을 전달하거나 Theano/TensorFlow 심볼릭 함수를 전달하시면 됩니다. ([Custom metrics](#custom-metrics)를 참조하십시오)

#### 인수
  - __y_true__: 참 라벨. Theano/TensorFlow 텐서.
  - __y_pred__: 예측. y_true와 같은 형태를 취하는 Theano/TensorFlow 텐서.

#### 반환값
  모든 데이터포인트에 대한 아웃풋 배열의 평균을 나타내는
  하나의 텐서 값.

----

## 사용가능한 측정항목


### binary_accuracy


```python
keras.metrics.binary_accuracy(y_true, y_pred)
```

----

### categorical_accuracy


```python
keras.metrics.categorical_accuracy(y_true, y_pred)
```

----

### sparse_categorical_accuracy


```python
keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
```

----

### top_k_categorical_accuracy


```python
keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=5)
```

----

### sparse_top_k_categorical_accuracy


```python
keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=5)
```


위의 측정항목 외에도, [loss function](/losses) 페이지에 기술된 어떠한 손실 함수를 사용해도 좋습니다.

----

## 커스텀 측정항목

컴파일 단계에서 커스텀 측정항목을 전달할 수 있습니다.
커스텀 측정항목 함수는 `(y_true, y_pred)`를 인수로 받아야 하며
하나의 텐서 값을 반환해야 합니다.

```python
import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy', mean_pred])
```

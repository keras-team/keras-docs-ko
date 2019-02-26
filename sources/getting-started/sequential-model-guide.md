# 케라스 Sequential 모델 시작하기

`Sequential` 모델은 레이어를 선형적으로 쌓은 더미입니다.

생성자에 레이어 인스턴스 리스트를 전달해서 `Sequential` 모델을 만들 수 있습니다:

```python
from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential([
    Dense(32, input_shape=(784,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])
```

또한 `.add()` 메서드를 사용해 간단하게 레이어를 추가할 수 있습니다:

```python
model = Sequential()
model.add(Dense(32, input_dim=784))
model.add(Activation('relu'))
```

----

## 인풋 형태 특정하기

모델이 예상되는 인풋 형태를 알아야 하기에 `Sequential` 모델의 첫 레이어는 인풋 형태에 대한 정보를 받아야 합니다 (이는 첫 레이어에만 해당되는데, 차후 레이어는 자동 형태 유추를 할 수 있기 때문입니다). 이에는 여러가지 방법이 있습니다:

`input_shape` 인수를 첫 레이어에 전달합니다. 이는 형태 튜플(정수 튜플 혹은 `None` 입력값, 여기서 `None`은 양의 정수가 예상된다는 의미입니다)입니다. `input_shape`에 배치 차원은 포함되지 않습니다.
`Dense`와 같은 몇몇 2D 레이어는 `input_dim` 인수를 통해서 인풋 형태를 특정하는 방식을 지원하고, 몇몇 3D 레이어는 `input_dim`과 `input_length` 인수를 지원합니다.
인풋의 고정 배치 사이즈를 특정하려면 (이는 상태 순환 망에 유용합니다), `batch_size` 인수를 레이어에 전달하면 됩니다. 레이어에 `batch_size=32`와 `input_shape=(6, 8)`를 동시에 전달하면, 인풋의 모든 배치가 `(32, 6, 8)` 형태를 가진다고 예상하게 됩니다.

다음의 코드도 정확히 동일합니다:
```python
model = Sequential()
model.add(Dense(32, input_shape=(784,)))
```
```python
model = Sequential()
model.add(Dense(32, input_dim=784))
```

----

## 컴파일

모델을 학습시키기 전에 `compile` 메서드를 통해서 학습 과정을 구성해야 합니다. 이는 세 가지 인수를 전달받습니다:

- 옵티마이저. 기존 옵티마이저의 문자열 식별자(예. `rmsprop` 혹은 `adagrad`), 혹은 `Optimizer` 클래스의 인스턴스를 사용할 수 있습니다. 참조: [옵티마이저](/optimizers).
- 손실 함수. 이는 모델이 최소화 하려는 목표입니다. 기존 손실 함수의 문자열 식별자(예. `categorical_crossentropy` 혹은 `mse`), 혹은 목적 함수를 사용할 수 있습니다. 참조: [손실](/losses).
- 측정 항목의 리스트. 모든 분류 문제에 대해, 이 인수를 `metrics=['accuracy']`로 설정하십시오. 측정 항목으로는 기존 측정 항목의 문자열 식별자나 커스텀 측정 항목 함수를 사용할 수 있습니다.

```python
# 다중 클래스 분류 문제
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 이진 분류 문제
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 평균 제곱 오차 회귀 문제
model.compile(optimizer='rmsprop',
              loss='mse')

# 커스텀 측정항목
import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy', mean_pred])
```

----

## 학습

케라스 모델은 인풋 데이터와 라벨의 Numpy 배열에 대해서 학습됩니다. 보통 `fit` 함수를 사용하여 모델을 학습시킵니다. [여기서 관련 설명을 읽어보십시오](/models/sequential).

```python
# 2 클래스의 단일-인풋 모델 (이진 분류):

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 임의의 데이터를 만듭니다
import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000, 1))

# 데이터에 대해 32 샘플의 배치 단위로 반복하여 모델을 학습시킵니다
model.fit(data, labels, epochs=10, batch_size=32)
```

```python
# 10 클래스의 단일 인풋 모델 (범주형 분류):

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 임의의 데이터를 만듭니다
import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(10, size=(1000, 1))

# 라벨을 범주형 원-핫 인코딩으로 변환합니다
one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)

# 데이터에 대해 32 샘플의 배치 단위로 반복하여 모델을 학습시킵니다
model.fit(data, one_hot_labels, epochs=10, batch_size=32)
```

----


## 예시

시작하려면 다음의 예시를 참고하세요!

[예시 폴더](https://github.com/keras-team/keras/tree/master/examples)에서도 실제 데이터셋에 대한 예시 모델을 확인할 수 있습니다:

- CIFAR10 소형 이미지 분류: 실시간 데이터 증강을 사용한 컨볼루션 신경망(CNN)
- IMDB 영화 리뷰 감정 분석: 단어 시퀀스에 대한 장단기 메모리
- 로이터 뉴스 토픽 분류: 다층 퍼셉트론(MLP)
- MNIST 손글씨 숫자 분류: MLP & CNN
- 장단기 메모리를 이용한 부호 수준 텍스트 생성

...그 외 다수.


### 다중 클래스 다층 퍼셉트론 (MLP) 소프트맥스 분류:

```python
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

# 임의의 데이터를 만듭니다
import numpy as np
x_train = np.random.random((1000, 20))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
x_test = np.random.random((100, 20))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

model = Sequential()
# Dense(64)는 64개의 은닉 유닛으로 구성된 완전 연결 레이어입니다.
# 첫 레이어에는, 예상되는 인풋 데이터 형태를 특정해야 합니다.
# 다음은 20 차원 벡터입니다.
model.add(Dense(64, activation='relu', input_dim=20))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)
```


### 다층 퍼셉트론 이진 분류:

```python
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

# 임의의 데이터를 만듭니다
x_train = np.random.random((1000, 20))
y_train = np.random.randint(2, size=(1000, 1))
x_test = np.random.random((100, 20))
y_test = np.random.randint(2, size=(100, 1))

model = Sequential()
model.add(Dense(64, input_dim=20, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)
```


### VGG와 유사한 convent:

```python
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

# 임의의 데이터를 만듭니다
x_train = np.random.random((100, 100, 100, 3))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
x_test = np.random.random((20, 100, 100, 3))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)

model = Sequential()
# 인풋: 3채널의 100x100 이미지-> (100, 100, 3) 텐서
# 이는 각 3x3 크기의 컨볼루션 필터 32개를 적용합니다
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(x_train, y_train, batch_size=32, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=32)
```


### 장단기 메모리를 이용한 시퀀스 분류:

```python
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM

max_features = 1024

model = Sequential()
model.add(Embedding(max_features, output_dim=256))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=16, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=16)
```

### 1D 컨볼루션을 이용한 시퀀스 분류:

```python
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D

seq_length = 64

model = Sequential()
model.add(Conv1D(64, 3, activation='relu', input_shape=(seq_length, 100)))
model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(128, 3, activation='relu'))
model.add(Conv1D(128, 3, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=16, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=16)
```

### 다층 장단기 메모리를 이용한 시퀀스 분류

이 모델은 3개의 장단기 메모리를 쌓아서
고수준의 시간적 표현을 배울 수 있도록 합니다.

처음 두 개의 장단기 메모리는 완전한 아웃풋 시퀀스를 반환하지만, 마지막은
아웃풋 시퀀스의 마지막 스텝만 반환하여 시간적 차원을 제거합니다
(다시 말해, 인풋 시퀀스를 단일 벡터로 전환합니다).

<img src="https://keras.io/img/regular_stacked_lstm.png" alt="stacked LSTM" style="width: 300px;"/>

```python
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

data_dim = 16
timesteps = 8
num_classes = 10

# 예상되는 인풋 데이터 형태: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(32, return_sequences=True,
               input_shape=(timesteps, data_dim)))  # 32 차원 벡터의 시퀀스를 반환합니다
model.add(LSTM(32, return_sequences=True))  # 32 차원 벡터의 시퀀스를 반환합니다
model.add(LSTM(32))  # 32 차원의 단일 벡터를 반환합니다
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# 임의의 학습 데이터를 만듭니다
x_train = np.random.random((1000, timesteps, data_dim))
y_train = np.random.random((1000, num_classes))

# 임의의 검증 데이터를 만듭니다
x_val = np.random.random((100, timesteps, data_dim))
y_val = np.random.random((100, num_classes))

model.fit(x_train, y_train,
          batch_size=64, epochs=5,
          validation_data=(x_val, y_val))
```


### "상태형으로" 바꾼 동일한 다층 장단기 메모리

상태 순환 모델은 샘플 배치를 처리한 후 획득한 내적 상태 (메모리)를 다음
배치의 샘플의 초기 상태로 재활용합니다. 이는 계산의 복잡성을 감당할 만한 수준으로
유지해, 보다 긴 시퀀스를 처리할 수 있도록 해 줍니다.

[FAQ에서 상태 순환 신경망에 대해서 더 알아보십시오.](/getting-started/faq/#how-can-i-use-stateful-rnns)

```python
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

data_dim = 16
timesteps = 8
num_classes = 10
batch_size = 32

# 예상되는 인풋 배치 형태 (batch_size, timesteps, data_dim)
# 네트워크가 상태형이므로 완전한 batch_input_shape을 제공해야 한다는 점을 참고하십시오.
# 배치 k 내 색인 i의 샘플은 배치 k-1의 샘플 i의 후속 샘플입니다.
model = Sequential()
model.add(LSTM(32, return_sequences=True, stateful=True,
               batch_input_shape=(batch_size, timesteps, data_dim)))
model.add(LSTM(32, return_sequences=True, stateful=True))
model.add(LSTM(32, stateful=True))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# 임의의 학습 데이터를 만듭니다.
x_train = np.random.random((batch_size * 10, timesteps, data_dim))
y_train = np.random.random((batch_size * 10, num_classes))

# 임의의 검증 데이터를 만듭니다.
x_val = np.random.random((batch_size * 3, timesteps, data_dim))
y_val = np.random.random((batch_size * 3, num_classes))

model.fit(x_train, y_train,
          batch_size=batch_size, epochs=5, shuffle=False,
          validation_data=(x_val, y_val))
```

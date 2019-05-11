# 케라스 Sequential 모델 시작하기

`Sequential` 모델은 레이어를 선형으로 연결하여 구성합니다. 
레이어 인스턴스를 생성자에게 넘겨줌으로써 `Sequential` 모델을 구성할 수 있습니다.

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

또한, `.add()` 메소드를 통해서 쉽게 레이어를 추가할 수 있습니다.

```python
model = Sequential()
model.add(Dense(32, input_dim=784))
model.add(Activation('relu'))
```

----

## 입력 형태 지정하기

만들어진 모델은 입력 형태에 대한 정보를 필요로 합니다. 때문에 `Sequential` 모델의 첫 번째 레이어는 입력 형태에 대한 정보를 받습니다. 두 번째 이후 레이어들은 자동으로 형태를 추정할 수 있기 때문에 형태 정보를 갖고 있을 필요는 없습니다. 형태 정보를 전달하기 위한 방법은 다음과 같습니다.

- 정수형 또는 `None`으로 구성된 형태 튜플(shape tuple)의 `input_shape` 인자를 첫번째 레이어에 전달합니다. 여기서 `None`은 음이 아닌 어떠한 정수를 받을 수 있음을 의미합니다. 참고로 `input_shape`에는 배치 차원은(batch dimension) 포함되지 않습니다.
- `Dense`와 같은 일부 2D 레이어의 경우, 입력 형태를 `input_dim` 인자를 통해 지정할 수 있으며 일부 의 임시적인 3D 레이어는 `input_dim`과 `input_length` 인자를 지원합니다. 
- 입력 데이터를 위해 고정된 배치 형태를 필요로 하는 경우에는 레이어에 `batch_size` 인자를 넘길 수 있습니다. 이는 순환 신경망(recurrent network)를 사용할 때 유용합니다. 예를 들어, `batch_size=32`와 `input_shape=(6,8)`을 레이어에 넘겨주면 이후의 모든 입력을 `32, 6, 8`의 형태로 기대하여 처리합니다.

따라서, 다음의 예시들은 동치입니다.
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

모델을 학습시키기 이전에, `compile` 메소드를 통해서 학습 방식에 대한 환경설정을 해야 합니다. 다음의 세 개의 인자를 입력으로 받습니다.

- 정규화기 (optimizer). `rmsprp`나 `adagrad`와 같은 기존의 정규화기에 대한 문자열 식별자 또는 `Optimizer` 클래스의 인스턴스를 사용할 수 있습니다. 참고: [정규화기](/optimizers) 
- 손실 함수 (loss function). 모델이 최적화에 사용되는 목적 함수입니다. `categorical_crossentropy` 또는 `mse`와 같은 기존의 손실 함수의 문자열 식별자 또는 목적 함수를 사용할 수 있습니다. 참고: [손실](/losses)
- 기준(metric) 리스트. 분류 문제에 대해서는 `metrics=['accuracy']`로 설정합니다. 기준은 문자열 식별자 또는 사용자 정의 기준 함수를 사용할 수 있습니다. 

```python
# For a multi-class classification problem
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# For a binary classification problem
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# For a mean squared error regression problem
model.compile(optimizer='rmsprop',
              loss='mse')

# For custom metrics
import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy', mean_pred])
```

----

## 학습

케라스 모델들은 입력 데이터와 라벨로 구성된 Numpy 배열 위에서 이루어집니다. 모델을 학습기키기 위해서는 일반적으로 `fit`함수를 사용합니다. [여기서 자세한 정보를 알 수 있습니다](/models/sequential).

```python
# For a single-input model with 2 classes (binary classification):

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Generate dummy data
import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000, 1))

# Train the model, iterating on the data in batches of 32 samples
model.fit(data, labels, epochs=10, batch_size=32)
```

```python
# For a single-input model with 10 classes (categorical classification):

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Generate dummy data
import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(10, size=(1000, 1))

# Convert labels to categorical one-hot encoding
one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)

# Train the model, iterating on the data in batches of 32 samples
model.fit(data, one_hot_labels, epochs=10, batch_size=32)
```

----


## 예시

바로 실험해볼 수 있는 예시들입니다!

[examples 폴더](https://github.com/keras-team/keras/tree/master/examples) 안에서 실제 데이터셋에 대한 예시 모델들을 볼 수 있습니다.

- CIFAR10 소형 이미지 분류: 실시간 데이터 증강을 포함하는 합성곱 인공 신경망 (CNN)
- IMDB 영화 감상 분류: 연속적인 문자에 대한 LSTM
- Reuters newswires 주제 분류: 다계층 신경망 (MLP)
- MNIST 손으로 쓴 숫자 이미지 분류: MLP & CNN
- LSTM을 이용한 문자열 수준의 텍스트 생성기

...등등.


### 다중 클레스 소프트맥스 분류를 위한 다계층 인공 신경망 (MLP) :

```python
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

# Generate dummy data
import numpy as np
x_train = np.random.random((1000, 20))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
x_test = np.random.random((100, 20))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
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


### 이진 분류를 위한 MLP:

```python
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

# Generate dummy data
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


### VGG 스타일 convnet:

```python
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

# Generate dummy data
x_train = np.random.random((100, 100, 100, 3))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
x_test = np.random.random((20, 100, 100, 3))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)

model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
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


### 시퀀스 분류를 위한 LSTM:

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

### 시퀀스 분류를 위한 1D 합성곱:

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

### 시퀀스 분류를 위한 축적형 LSTM

보다 고수준의 한시적 표현(temporal representation)을 학습하기 위하여 3개의 LSTM 레이어를 연결하여 사용합니다.

앞의 두 LSTM은 각각의 전체 출력 시퀀스를 반환하지만, 마지막 남은 LSTM은 출력 시퀀스의 마지막 단계만을 반환합니다. 결과적으로 한시적 차원(temporal dimension)을 생략하게 됩니다. 이를테면 입력 시퀀스를 하나의 벡터로 변환합니다.

<img src="https://keras.io/img/regular_stacked_lstm.png" alt="stacked LSTM" style="width: 300px;"/>

```python
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

data_dim = 16
timesteps = 8
num_classes = 10

# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(32, return_sequences=True,
               input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32))  # return a single vector of dimension 32
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Generate dummy training data
x_train = np.random.random((1000, timesteps, data_dim))
y_train = np.random.random((1000, num_classes))

# Generate dummy validation data
x_val = np.random.random((100, timesteps, data_dim))
y_val = np.random.random((100, num_classes))

model.fit(x_train, y_train,
          batch_size=64, epochs=5,
          validation_data=(x_val, y_val))
```


### 축적형 렌더링 LSTM

축적형 순환 신경망(stateful recurrent model)은 배치 샘플을 처리한 후의 내부 상태를 다음 배치 샘플에 대한 초기 상태로 재사용합니다. 이를 통해서 계산 복잡도가 지나치게 높지 않게끔 유지하면서 보다 긴 시퀀스를 처리할 수 있도록 합니다.
[FAQ에서 축적형 LSTM에 대한 정보를 더 보실 수 있습니다.](/getting-started/faq/#how-can-i-use-stateful-rnns)

```python
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

data_dim = 16
timesteps = 8
num_classes = 10
batch_size = 32

# Expected input batch shape: (batch_size, timesteps, data_dim)
# Note that we have to provide the full batch_input_shape since the network is stateful.
# the sample of index i in batch k is the follow-up for the sample i in batch k-1.
model = Sequential()
model.add(LSTM(32, return_sequences=True, stateful=True,
               batch_input_shape=(batch_size, timesteps, data_dim)))
model.add(LSTM(32, return_sequences=True, stateful=True))
model.add(LSTM(32, stateful=True))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Generate dummy training data
x_train = np.random.random((batch_size * 10, timesteps, data_dim))
y_train = np.random.random((batch_size * 10, num_classes))

# Generate dummy validation data
x_val = np.random.random((batch_size * 3, timesteps, data_dim))
y_val = np.random.random((batch_size * 3, num_classes))

model.fit(x_train, y_train,
          batch_size=batch_size, epochs=5, shuffle=False,
          validation_data=(x_val, y_val))
```

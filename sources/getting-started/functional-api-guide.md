# 케라스 함수형 API 시작하기

Keras 함수형 API<sub>functional API</sub>는 다중 출력 모델<sub>multi-output model</sub>, 유향 비순환 그래프<sub>directed acyclic graphs</sub>, 혹은 층<sub>layer</sub>들을 공유하는 모델과 같이 복잡한 모델을 정의하는 최적의 방법입니다.

이 가이드는 독자가 이미 `Sequential` 모델에 대한 지식이 있다고 가정합니다.

간단한 예시로 시작합시다.

-----

## 첫 번째 예시: 완전 연결 신경망<sub>densely-connected network</sub>

완전 연결 신경망에는 `Sequential` 모델이 더 적합하지만, 간단한 예시를 위해 Keras 함수형 API로 구현해 보겠습니다.

- 층 인스턴스<sub>instance</sub>는 텐서에 대해 호출 가능하고, 텐서를 반환합니다.
- 입력<sub>input</sub> 텐서와 출력<sub>output</sub> 텐서을 통해 `Model`을 정의할 수 있습니다.
- 이러한 모델은 Keras `Sequential` 모델과 동일한 방식으로 학습됩니다.

```python
from keras.layers import Input, Dense
from keras.models import Model

# 텐서를 반환합니다.
inputs = Input(shape=(784,))

# 층 인스턴스는 텐서를 호출할 수 있는 객체이며, 텐서를 반환합니다.
output_1 = Dense(64, activation='relu')(inputs)
output_2 = Dense(64, activation='relu')(output_1)
predictions = Dense(10, activation='softmax')(output_2)

# 입력 층과 3개의 완전 연결 층을
# 포함하는 모델을 만듭니다.
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data, labels)  # 학습을 시작합니다.
```

-----

## 모든 모델은 층 인스턴스처럼 호출 가능합니다

함수형 API를 사용하면 학습된 모델을 재사용하기 편리합니다. 어느 모델이건 텐서를 호출하여 층 인스턴스처럼 사용할 수 있습니다. 모델을 호출하면 모델의 *구조*만 재사용하는 것이 아니라 가중치<sub>weights</sub>까지 재사용하게 됩니다.

```python
x = Input(shape=(784,))
# 위에서 정의한 10-way softmax를 반환합니다.
y = model(x)
```

예를 들어, 이를 이용하면 시퀀스 데이터<sub>sequences</sub>를 처리할 수 있는 모델을 빠르게 만들 수 있습니다. 코드 한 줄로 이미지 분류 모델을 비디오 분류 모델로 바꿀 수 있습니다.

```python
from keras.layers import TimeDistributed

# 20개의 시간 단계를 갖는 시퀀스의 입력 텐서입니다.
# 각 시간 단계는 784 차원의 벡터입니다.
input_sequences = Input(shape=(20, 784))

# 입력 시퀀스의 모든 시간 단계에 이전 모델을 적용합니다.
# 앞선 모델의 출력이 10-way softmax였으므로,
# 아래에 주어진 layer 인스턴스의 출력은 10차원 벡터 20개로 이루어진 시퀀스입니다.
processed_sequences = TimeDistributed(model)(input_sequences)
```

-----

## 다중 입력<sub>multi-input</sub>과 다중 출력 모델

다중 입력과 출력을 가지는 모델은 함수형 API를 적용하기 적합한 사례입니다. 함수형 API를 사용해서 복잡하게 얽힌 많은 수의 데이터 흐름을 간편하게 관리할 수 있습니다.

다음 모델을 고려해 봅시다. 트위터에서 특정 뉴스 헤드라인이 얼마나 많은 리트윗과 좋아요를 받을지 예측하려고 합니다. 모델에 대한 주요 입력은 연속된 단어들로 표현된 헤드라인 자체이지만, 약간의 변화를 주기 위해서 보조 입력으로 헤드라인이 게시된 시간 등과 같은 추가 데이터를 받도록 합니다.
모델 학습에는 두 개의 손실 함수<sub>loss function</sub>가 사용됩니다. 주요 손실 함수를 모델의 초기 단계에 사용하는 것이 심층 모델에 있어 적절한 정규화<sub>regularization</sub> 메커니즘입니다.

다음은 모델이 어떻게 구성되어있는지 보여줍니다.

<img src="https://s3.amazonaws.com/keras.io/img/multi-input-multi-output-graph.png" alt="multi-input-multi-output-graph" style="width: 400px;"/>

함수형 API로 모델을 구현해 봅시다.

주요 입력은 `int` 시퀀스들의 형태로 헤드라인을 (각 `int`가 단어 하나를 인코딩하는) 전달받습니다.
1에서 10,000사이의 `int`이며 (10,000 단어의 어휘목록), 하나의 `int` 시퀀스는 100 단어로 이루어져 있습니다.

```python
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
import numpy as np
np.random.seed(0)  # 재현성을 위해 임의의 시드값을 설정합니다.

# 헤드라인 입력값: 1에서 10000 사이의 100개 정수로 이루어진 시퀀스들을 전달받습니다
# "name" 인자를 전달하여 층 인스턴스를 명명할 수 있음을 참고하십시오.
main_input = Input(shape=(100,), dtype='int32', name='main_input')

# Embedding layer는 입력 시퀀스를
# 512 차원 완전 연결 벡터들의 시퀀스로 인코딩합니다.
x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)

# LSTM은 벡터 시퀀스를 전체 시퀀스에 대한
# 정보를 포함하는 단일 벡터로 변환합니다.
lstm_out = LSTM(32)(x)
```

보조 손실 함수를 사용하여 모델에서 주요 손실 함수가 후기 단계에서 사용되더라도 LSTM 및 Embedding layer를 원활하게 학습 할 수 있습니다.

```python
auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)
```

이 시점에서 보조 입력 데이터를 LSTM 출력과 연결함으로써 보조 입력 데이터를 모델에 전달합니다.

```python
auxiliary_input = Input(shape=(5,), name='aux_input')
x = keras.layers.concatenate([lstm_out, auxiliary_input])

# 깊은 완전 연결 신경망을 맨 위에 쌓습니다
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)

# 끝으로 주요 로지스틱 회귀 층을 추가합니다
main_output = Dense(1, activation='sigmoid', name='main_output')(x)
```

다음은 두 개의 입력과 두 개의 출력을 갖는 모델을 정의합니다.

```python
model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])
```

보조 손실의 가중치로 0.2를 할당하고 모델을 컴파일합니다.
리스트나 딕셔너리를 사용해서 각각의 출력에 서로 다른 `loss_weights` 혹은 `loss`를 지정할 수 있습니다.
여기서는 `loss` 인자<sub>argument</sub>에 하나의 손실 함수를 전달하므로 모든 출력에 동일한 손실 함수가 사용됩니다.

```python
model.compile(optimizer='rmsprop', loss='binary_crossentropy',
              loss_weights=[1., 0.2])
```

입력 배열과 타겟 배열의 리스트를 전달하여 모델을 학습시킬 수 있습니다.

```python
headline_data = np.round(np.abs(np.random.rand(12, 100) * 100))
additional_data = np.random.randn(12, 5)
headline_labels = np.random.randn(12, 1)
additional_labels = np.random.randn(12, 1)
model.fit([headline_data, additional_data], [headline_labels, additional_labels],
          epochs=50, batch_size=32)
```

입력 층 인스턴스와 출력 층 인스턴스에 이름을 부여했으므로(각 층 인스턴스에 "name" 인자를 전달했으므로),
다음과 같은 방식으로도 모델을 컴파일 할 수 있습니다.

```python
model.compile(optimizer='rmsprop',
              loss={'main_output': 'binary_crossentropy', 'aux_output': 'binary_crossentropy'},
              loss_weights={'main_output': 1., 'aux_output': 0.2})

# 모델을 다음과 같은 방식으로도 학습시킬 수 있습니다.
model.fit({'main_input': headline_data, 'aux_input': additional_data},
          {'main_output': headline_labels, 'aux_output': additional_labels},
          epochs=50, batch_size=32)
```

예측을 하기 위해서 모델을 사용할 때, 다음과 같이 사용하거나
```python
model.predict({'main_input': headline_data, 'aux_input': additional_data})
```
다음과 같은 방법으로도 사용할 수 있습니다.
```python
pred = model.predict([headline_data, additional_data])
```

-----

## 공유 층<sub>shared layers</sub>

함수형 API의 다른 유용한 사례는 공유 층을 사용하는 모델입니다. 공유 층에 대해서 알아봅시다.

트윗 데이터셋을 고려해 봅시다. 서로 다른 두 개의 트윗을 두고 동일한 사람이 작성했는지 여부를 가려내는 모델을 만들고자 합니다 (예를 들어, 트윗의 유사성을 기준으로 사용자를 비교할 수 있습니다).

이를 구현하는 한 가지 방법은 두 개의 트윗을 두 벡터로 인코딩하고 그 두 벡터를 연결한 후 로지스틱 회귀를 수행하는 모델을 만드는 것입니다; 이 모델은 서로 다른 두 트윗의 작성자가 동일할 확률을 출력합니다. 그런 다음 모델은 긍정적인 트윗 쌍과 부정적인 트윗 쌍에 대해서 학습됩니다.

문제가 대칭적이므로 긍정적인 트윗 쌍을 인코딩하는 메커니즘을 (가중치 등을 포함해) 전부 재사용하여 부정적인 트윗 쌍을 인코딩해야 합니다. 이 예시에서는 공유된 LSTM 층을 사용해 트윗을 인코딩 합니다.

함수형 API로 모델을 만들어 봅시다. `(280, 256)` 형태의 이진 행렬, 다시 말해 256 크기의 벡터 280개로 이루어진 시퀀스를 트윗에 대한 입력으로 받습니다 (여기서 256 차원 벡터의 각 차원은 자모에서 가장 빈번하게 사용되는 256 문자의 유무를 인코딩합니다).

```python
import keras
from keras.layers import Input, LSTM, Dense
from keras.models import Model

tweet_a = Input(shape=(280, 256))
tweet_b = Input(shape=(280, 256))
```

층을 재사용하려면, 간단히 층을 한 번만 인스턴스화하고 서로 다른 입력마다 층 인스턴스를 호출하면 됩니다.

```python
# 이 층 인스턴스는 행렬을 입력으로 전달받고
# 크기가 64인 벡터를 반환합니다.
shared_lstm = LSTM(64)

# 동일한 층 인스턴스를
# 여러 번 재사용하는 경우, layer의
# 가중치 또한 재사용됩니다.
# (그렇기에 이는 *동일한* 레이어입니다)
encoded_a = shared_lstm(tweet_a)
encoded_b = shared_lstm(tweet_b)

# 이제 두 벡터를 연결할 수 있습니다.
merged_vector = keras.layers.concatenate([encoded_a, encoded_b], axis=-1)

# 그리고 로지스틱 회귀를 상층에 추가합니다.
predictions = Dense(1, activation='sigmoid')(merged_vector)

# 트윗 입력을 예측에 연결하는
# 학습 가능한 모델을 정의합니다.
model = Model(inputs=[tweet_a, tweet_b], outputs=predictions)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit([data_a, data_b], labels, epochs=10)
```

잠시 멈춰서 공유 층 인스턴스의 출력 또는 출력 형태<sub>shape</sub>가 어떠한지 살펴봅시다.

-----

## "노드"의 개념

입력에 대해서 층 인스턴스를 호출하면, 새로운 텐서(층 인스턴스의 출력)가 생성됩니다. 또한 "노드"가 층 인스턴스에 추가되어 입력 텐서와 출력 텐서를 연결합니다. 동일한 층 인스턴스를 여러 번 호출하면, 층 인스턴스는 0, 1, 2…와 같은 인덱스가 달린 여러 개의 노드를 소유하게 됩니다.

케라스 이전 버전에서는 `layer.get_output()`를 통해 층 인스턴스의 출력 텐서를 얻을 수 있고, `layer.output_shape`를 통해 출력 형태를 얻을 수 있었습니다. 이러한 방식이 여전히 가능하기는 하지만 (단, `get_output()`은 `output`이라는 속성으로 대체되었습니다), 층 인스턴스가 여러 입력에 연결된 경우는 어떻게 하시겠습니까?

층 인스턴스가 하나의 입력에만 연결되어 있으면 `.output`은 층 인스턴스의 단일 출력을 반환할 것입니다.

```python
a = Input(shape=(280, 256))

lstm = LSTM(32)
encoded_a = lstm(a)

assert lstm.output == encoded_a
```

층 인스턴스가 여러 입력에 대해서 호출이 된다면 상황이 달라집니다.
```python
a = Input(shape=(280, 256))
b = Input(shape=(280, 256))

lstm = LSTM(32)
encoded_a = lstm(a)
encoded_b = lstm(b)

lstm.output
```
```
>> AttributeError: Layer lstm_1 has multiple inbound nodes,
hence the notion of "layer output" is ill-defined.
Use `get_output_at(node_index)` instead.
```

다음은 제대로 작동합니다.

```python
assert lstm.get_output_at(0) == encoded_a
assert lstm.get_output_at(1) == encoded_b
```

간단하지 않습니까?

`input_shape`와 `output_shape`의 경우도 마찬가지 입니다: 층 인스턴스가 하나의 노드만 보유하거나 모든 노드가 동일한 입력/출력 형태를 갖는 한, "layer output/input shape"에 대한 개념이 정의되며 `layer.output_shape`/`layer.input_shape`는 동일한 형태를 반환합니다. 하지만 만약 동일한 `Conv2D` layer를 `(32, 32, 3)` 형태의 입력에 대해서 호출한 다음 `(64, 64, 3)` 형태의 입력에 대해서도 호출한다면, layer는 여러 입력/출력 형태를 가지게 됩니다. 이러한 경우 각 입력/출력 형태가 속한 노드의 인덱스를 지정해야만 형태를 얻을 수 있습니다.

```python
a = Input(shape=(32, 32, 3))
b = Input(shape=(64, 64, 3))

conv = Conv2D(16, (3, 3), padding='same')
conved_a = conv(a)

# 지금까지 하나의 입력만 존재하므로 다음은 제대로 동작합니다.
assert conv.input_shape == (None, 32, 32, 3)

conved_b = conv(b)
# 이제 `.input_shape`은 제대로 작동하지 않지만, 다음은 동작합니다.
assert conv.get_input_shape_at(0) == (None, 32, 32, 3)
assert conv.get_input_shape_at(1) == (None, 64, 64, 3)
```

-----

## 추가 예시

시작하기에 코드 예시보다 좋은 방법은 없으므로, 몇 가지 예시를 더 살펴봅시다.

### Inception 모듈

Inception 구조에 대해서 더 알고 싶다면, [Going Deeper with Convolutions](http://arxiv.org/abs/1409.4842)를 참고하십시오.

```python
from keras.layers import Conv2D, MaxPooling2D, Input

input_img = Input(shape=(256, 256, 3))

tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_img)
tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)

tower_2 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_img)
tower_2 = Conv2D(64, (5, 5), padding='same', activation='relu')(tower_2)

tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_img)
tower_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(tower_3)

output = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)
```

### 합성곱 층<sub>Convolution layer</sub>에 대한 residual connection

Residual network에 대해 더 알고 싶다면, [Deep Residual Learning for Image Recognition](http://arxiv.org/abs/1512.03385)를 참고하십시오.

```python
from keras.layers import Conv2D, Input

# 3개의 채널을 가진 256x256 이미지에 대한 입력 텐서
x = Input(shape=(256, 256, 3))
# 입력 채널과 같은 3개의 출력 채널을 가진 3x3 합성곱
y = Conv2D(3, (3, 3), padding='same')(x)
# x + y를 반환합니다
z = keras.layers.add([x, y])
```

### 공유 시각 모델

이 모델은 동일한 이미지 처리 모듈을 두 입력에 대해서 재사용하여, 두 MNIST 숫자가 같은 숫자인지 다른 숫자인지 여부를 분류합니다.

```python
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten
from keras.models import Model

# 우선, 시각 모듈을 정의합니다.
digit_input = Input(shape=(27, 27, 1))
x = Conv2D(64, (3, 3))(digit_input)
x = Conv2D(64, (3, 3))(x)
x = MaxPooling2D((2, 2))(x)
out = Flatten()(x)

vision_model = Model(digit_input, out)

# 그리고 숫자 분류 모델을 정의합니다.
digit_a = Input(shape=(27, 27, 1))
digit_b = Input(shape=(27, 27, 1))

# 시각 모델은 가중치 등을 포함한 모든것이 공유됩니다.
out_a = vision_model(digit_a)
out_b = vision_model(digit_b)

concatenated = keras.layers.concatenate([out_a, out_b])
out = Dense(1, activation='sigmoid')(concatenated)

classification_model = Model([digit_a, digit_b], out)
```

### 시각적 문답 모델

이 모델은 사진에 대한 질문을 받았을 때 올바른 한 단어로 답변을 선택할 수 있습니다.

질문과 이미지를 벡터로 인코딩하여 두 벡터를 연결한 후, 잠재적 답변의 어휘 목록에 대해서 상층의 로지스틱 회귀를 학습시키는 방식으로 작동합니다.

```python
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model, Sequential

# 우선 Sequential 모델을 사용해서 시각 모델을 정의합시다.
# 다음 모델은 이미지를 벡터로 인코딩합니다.
vision_model = Sequential()
vision_model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)))
vision_model.add(Conv2D(64, (3, 3), activation='relu'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
vision_model.add(Conv2D(128, (3, 3), activation='relu'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
vision_model.add(Conv2D(256, (3, 3), activation='relu'))
vision_model.add(Conv2D(256, (3, 3), activation='relu'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Flatten())

# 시각 모델의 출력으로 텐서를 얻어 봅시다.
image_input = Input(shape=(224, 224, 3))
encoded_image = vision_model(image_input)

# 다음은 질문을 벡터로 인코딩할 언어 모델을 정의합니다
# 각 질문의 최대 길이는 100 단어입니다.
# 그리고 각 단어에 1에서 9999까지의 `int` 인덱스를 부여합니다.
question_input = Input(shape=(100,), dtype='int32')
embedded_question = Embedding(input_dim=10000, output_dim=256, input_length=100)(question_input)
encoded_question = LSTM(256)(embedded_question)

# 질문 벡터와 이미지 벡터를 연결해 봅시다.
merged = keras.layers.concatenate([encoded_question, encoded_image])

# 그리고 상층에 1000개의 잠재적 답변에 대한 로지스틱 회귀를 학습시킵시다.
output = Dense(1000, activation='softmax')(merged)

# 다음은 최종 모델입니다.
vqa_model = Model(inputs=[image_input, question_input], outputs=output)

# 다음 단계는 실제 데이터에 대해 이 모델을 학습시키는 것입니다.
```

### 비디오 문답 모델

이미지 문답 모델을 학습했으니, 이를 간단하게 비디오 문답 모델로 바꿀 수 있습니다. 적절한 학습을 통해서 모델에 짧은 비디오(예. 100-프레임 사람 행동)를 보여주고 비디오에 대한 질문을 할 수 있습니다(예: "남자 아이는 무슨 스포츠를 하고 있니?" -> "축구").

```python
from keras.layers import TimeDistributed

video_input = Input(shape=(100, 224, 224, 3))
# 기존에 학습된 (가중치가 재사용된) vision_model을 통해서 인코딩된 비디오입니다.
encoded_frame_sequence = TimeDistributed(vision_model)(video_input)  # 출력은 벡터의 시퀀스가 됩니다.
encoded_video = LSTM(256)(encoded_frame_sequence)  # 출력은 벡터입니다.

# 다음은 이전 예시와 동일한 가중치를 재사용한 질문 인코더의 모델 수준 표현입니다.
question_encoder = Model(inputs=question_input, outputs=encoded_question)

# 이를 사용해서 질문을 인코딩해 봅시다.
video_question_input = Input(shape=(100,), dtype='int32')
encoded_video_question = question_encoder(video_question_input)

# 다음은 비디오 문답 모델입니다.
merged = keras.layers.concatenate([encoded_video, encoded_video_question])
output = Dense(1000, activation='softmax')(merged)
video_qa_model = Model(inputs=[video_input, video_question_input], outputs=output)
```

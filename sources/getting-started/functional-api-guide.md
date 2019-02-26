# 케라스 기능적 API 첫걸음

케라스 기능적 API는 다중-아웃풋 모델, 비순환 유향 그래프, 혹은 레이어 공유 모델과 같이 복잡한 모델을 정의하는데 최적의 방법입니다.

이 가이드는 독자가 이미 `Sequential` 모델에 대한 지식이 있다고 가정합니다.

간단한 예시로 시작합시다.

-----

## 첫 예시: 밀집 연결 네트워크:

밀집 연결 네트워크를 구현하기에는 `Sequential` 모델이 더 적합한 선택이겠지만, 아주 간단한 예시를 위해서 케라스 기능적 API로 구현해 보겠습니다.

- 레이어 인스턴스는 (텐서에 대해) 호출 가능하고, 텐서를 반환합니다
- 인풋 텐서와 아웃풋 텐서는 `Model`을 정의하는데 사용됩니다.
- 이러한 모델은 케라스 `Sequential` 모델과 완전히 동일한 방식으로 학습됩니다.

```python
from keras.layers import Input, Dense
from keras.models import Model

# 이는 텐서를 반환합니다
inputs = Input(shape=(784,))

# 레이어 인스턴스는 텐서에 대해 호출 가능하고, 텐서를 반환합니다
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

# 이는 Input 레이어와 3개의 Dense 레이어를
# 포함하는 모델을 만들어 냅니다
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data, labels)  # starts training
```

-----

## 레이어처럼, 모든 모델이 호출 가능합니다

기능적 API를 사용하면 학습된 모델을 재사용하기 편리합니다: 어느 모델이건 텐서에 대해 호출하여 레이어처럼 사용할 수 있습니다. 모델을 호출하면 모델의 *구조*만 재사용하는 것이 아니라 가중치까지 재사용되는 것임을 참고하십시오.

```python
x = Input(shape=(784,))
# 다음이 실행되어 위에서 정의한 10 방향 소프트맥스를 반환합니다.
y = model(x)
```

이를 사용하면, 예를 들어 인풋의 *시퀀스*를 처리할 수 있는 모델을 빠르게 만들 수 있습니다. 코드 한 줄로 이미지 분류 모델을 비디오 분류 모델로 바꿀 수 있는 것입니다.

```python
from keras.layers import TimeDistributed

# 20 시간 단계의 시퀀스에 대한 인풋 텐서로,
# 각각 784 차원의 벡터를 담고 있습니다.
input_sequences = Input(shape=(20, 784))

# 인풋 시퀀스의 모든 시간 단계에 이전 모델을 적용합니다.
# 이전 모델의 아웃풋이 10 방향 소프트맥스였으므로,
# 아래 레이어의 아웃풋은 크기 10의 벡터 20개로 이루어진 시퀀스입니다.
processed_sequences = TimeDistributed(model)(input_sequences)
```

-----

## 다중-인풋과 다중-아웃풋 모델

다중 인풋과 아웃풋을 다루는 모델은 기능적 API를 사용하기 특히 적합한 사례입니다. 기능적 API를 사용해서 복잡하게 얽힌 많은 수의 데이터 줄기를 간편하게 관리할 수 있습니다.

다음의 모델을 고려해 봅시다. 얼마나 많은 트위터에서 특정 뉴스 헤드라인이 얼마나 낳은 리트윗과 좋아요를 받을지 예측하려고 합니다. 모델에 대한 주요 인풋은 단어 시퀀스로 표현된 헤드라인 자체이지만, 문제를 조금 더 흥미롭게 풀기 위해, 보조 인풋으로 헤드라인이 올려진 시간 등과 같은 추가 데이터를 받도록 합시다.
두 개의 손실 함수를 통해 모델을 감독합니다. 주요 손실 함수를 모델의 초기 단계에 사용하는 것이 심화 모델에 있어 적절한 정규화 메커니즘입니다.

이 모델은 다음과 같이 구성됩니다:

<img src="https://s3.amazonaws.com/keras.io/img/multi-input-multi-output-graph.png" alt="multi-input-multi-output-graph" style="width: 400px;"/>

기능적 API로 모델을 구현해 봅시다.

주요 인풋은 헤드라인을 (각 정수가 단어 하나를 인코딩하는) 정수 시퀀스의 형태로 전달받습니다.
정수는 1에서 10,000사이의 값이며 (10,000 단어의 어휘목록), 시퀀스의 길이는 100 단어입니다.

```python
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model

# 헤드라인 인풋: 1에서 10000사이의 100개 정수로 이루어진 시퀀스를 전달받습니다
# "name"인수를 전달하여 레이어를 명명할 수 있음을 참고하십시오.
main_input = Input(shape=(100,), dtype='int32', name='main_input')

# 이 임베딩 레이어는 인풋 시퀀스를
# 밀집 512 차원 벡터의 시퀀스로 인코딩합니다
x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)

# 장단기 메모리는 벡터 시퀀스를 전체 시퀀스에 대한
# 정보를 포함하는 단일 벡터로 변형합니다
lstm_out = LSTM(32)(x)
```

다음에서는 보조 손실을 추가하여, 메인 손실이 높아지더라도 장단기 메모리와 임베딩 레이어가 매끄럽게 학습될 수 있도록 합니다.

```python
auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)
```

이 시점에서 보조 인풋을 장단기 메모리 아웃풋에 연결하여 모델에 전달합니다:

```python
auxiliary_input = Input(shape=(5,), name='aux_input')
x = keras.layers.concatenate([lstm_out, auxiliary_input])

# 심화 밀집 연결 네트워크를 상층에 쌓습니다
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)

# 끝으로 주요 로지스틱 회귀 레이어를 추가합니다
main_output = Dense(1, activation='sigmoid', name='main_output')(x)
```

이는 두 개의 인풋과 두 개의 아웃풋을 갖는 모델을 정의합니다:

```python
model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])
```

이제 모델을 컴파일하고 0.2의 가중치를 보조 손실에 배당합니다.
리스트나 딕셔너리를 사용해서 각기 다른 아웃풋에 별도의 `loss_weights` 혹은 `loss`를 특정할 수 있습니다.
다음에서는 `loss`인수로 단일 손실을 전달하여 모든 아웃풋에 대해 동일한 손실이 사용되도록 합니다.

```python
model.compile(optimizer='rmsprop', loss='binary_crossentropy',
              loss_weights=[1., 0.2])
```

인풋 배열과 표적 배열의 리스트를 전달하여 모델을 학습시킬 수 있습니다:

```python
model.fit([headline_data, additional_data], [labels, labels],
          epochs=50, batch_size=32)
```

인풋과 아웃풋이 명명되었기에 (이름은 "name" 인수를 통해 전달합니다),
다음과 같이 모델을 컴파일 할 수 있습니다:

```python
model.compile(optimizer='rmsprop',
              loss={'main_output': 'binary_crossentropy', 'aux_output': 'binary_crossentropy'},
              loss_weights={'main_output': 1., 'aux_output': 0.2})

또한 모델을 다음과 같이도 학습시킬 수 있습니다:
model.fit({'main_input': headline_data, 'aux_input': additional_data},
          {'main_output': labels, 'aux_output': labels},
          epochs=50, batch_size=32)
```

-----

## 공유 레이어

기능적 API의 또 다른 유용한 사용처는 공유 레이어를 사용하는 모델입니다. 공유 레이어에 대해서 알아봅시다.

트윗 데이터셋을 고려해 봅시다. 두 개의 다른 트윗을 두고 동일한 사람이 작성했는지 여부를 가려내는 모델을 만들고자 합니다 (이는, 예를 들어 트윗의 유사성을 기준으로 사용자를 비교할 수 있도록 합니다).

이를 구현하는 방법의 하나는 두 개의 트윗을 두 벡터로 인코딩하고 그 두 벡터를 연결한 후 로지스틱 회귀를 더하는 모델을 만드는 것입니다; 이는 두 트윗이 동일한 저자를 공유할 확률을 출력합니다. 그에 따라 모델은 양성 트윗 쌍과 음성 트윗 쌍에 대해서 학습됩니다.

문제가 대칭적이므로, 첫 번째 트윗을 인코딩하는 메커니즘을 (가중치 등을 포함해) 전부 재사용하여 두 번째 트윗을 인코딩해야 합니다. 이 예시에서는 공유된 장단기 메모리 레이어를 사용해 트윗을 인코딩 합니다.

기능적 API로 이 모델을 만들어 봅시다. `(280, 256)` 형태의 이진 행렬, 다시 말해 256 크기의 벡터 280개로 이루어진 시퀀스를 트윗에 대한 인풋으로 받습니다 (여기서 256 차원 벡터의 각 차원은 가장 빈번한 256 부호의 자모 중 해당 부호의 유/무를 인코딩합니다).

```python
import keras
from keras.layers import Input, LSTM, Dense
from keras.models import Model

tweet_a = Input(shape=(280, 256))
tweet_b = Input(shape=(280, 256))
```

제 각기 다른 인풋에 걸쳐 레이어를 공유하려면, 간단히 레이어를 한 번만 인스턴스화하고 인풋마다 그 레이어를 호출하면 됩니다:

```python
# 이 레이어는 행렬을 인풋으로 전달받고
# 64 크기의 벡터를 반환합니다
shared_lstm = LSTM(64)

# 동일한 레이어 인스턴스를
# 여러 번 재사용하는 경우, 레이어의
# 가중치 또한 재사용됩니다
# (그렇기에 이는 실질적으로 *동일한* 레이어입니다)
encoded_a = shared_lstm(tweet_a)
encoded_b = shared_lstm(tweet_b)

# 이제 두 벡터를 연결할 수 있습니다:
merged_vector = keras.layers.concatenate([encoded_a, encoded_b], axis=-1)

# 그리고 로지스틱 회귀를 상층에 추가합니다
predictions = Dense(1, activation='sigmoid')(merged_vector)

# 트윗 인풋을 예측에 연결하는
# 학습 가능한 모델을 정의합니다
model = Model(inputs=[tweet_a, tweet_b], outputs=predictions)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit([data_a, data_b], labels, epochs=10)
```

잠시 멈추고 공유 레이어의 아웃풋 혹은 아웃풋 형태를 어떻게 해석해야 할지 알아봅시다.

-----

## 레이어 "node"의 개념

어느 인풋에 대해서 레이어를 호출하면, 새로운 텐서(레이어의 아웃풋)가 생성됩니다. 또한 "node"가 레이어에 추가되어 인풋 텐서를 아웃풋 텐서에 연결합니다. 동일한 레이어를 여러 번 호출하면, 레이어는 0, 1, 2…로 색인이 달린 여러 개의 노드를 소유하게 됩니다.

케라스의 이전 버전에서는, `layer.get_output()`를 통해 레이어 인스턴스의 아웃풋 텐서를 얻거나, 혹은 `layer.output_shape`를 통해 아웃풋 형태를 얻을 수 있었습니다. 이러한 방식이 여전히 가능하기는 하지만 (`output`으로 대체된 `get_output()`은 제외), 레이어가 여러 인풋에 연결된 경우는 어떻게 하시겠습니까?

하나의 레이어가 하나의 인풋에만 연결되는 한, `.output`은 혼동없이 레이어의 단일 아웃풋을 반환할 것입니다:

```python
a = Input(shape=(280, 256))

lstm = LSTM(32)
encoded_a = lstm(a)

assert lstm.output == encoded_a
```

레이어가 여러 인풋을 가지면 상황이 달라집니다:
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

좋습니다. 다음은 제대로 작동합니다:

```python
assert lstm.get_output_at(0) == encoded_a
assert lstm.get_output_at(1) == encoded_b
```

간단하지 않습니까?

`input_shape`와 `output_shape`의 경우도 마찬가지 입니다: 레이어가 하나의 노드만 보유하는 한, 혹은 모든 노드가 동일한 인풋/아웃풋 형태를 갖는 한, "layer output/input shape"의 개념이 명확히 정의되며 `layer.output_shape`/`layer.input_shape`은 동일한 형태를 반환합니다. 하지만 만약 동일한 `Conv2D` 레이어를 `(32, 32, 3)` 형태의 인풋에 적용한 후 `(64, 64, 3)` 형태의 인풋에도 적용했다면, 레이어는 여러 종류의 인풋/아웃풋 형태를 보유하게 됩니다. 이러한 경우 각 인풋/아웃풋 형태가 속한 노드의 색인을 특정해서 불러와야 합니다:

```python
a = Input(shape=(32, 32, 3))
b = Input(shape=(64, 64, 3))

conv = Conv2D(16, (3, 3), padding='same')
conved_a = conv(a)

# 지금까진 하나의 인풋만 존재하므로 다음은 제대로 동작합니다:
assert conv.input_shape == (None, 32, 32, 3)

conved_b = conv(b)
# 이제 `.input_shape`은 제대로 작동하지 않지만, 다음은 동작합니다:
assert conv.get_input_shape_at(0) == (None, 32, 32, 3)
assert conv.get_input_shape_at(1) == (None, 64, 64, 3)
```

-----

## 추가 예시

시작하기에 코드 예시보다 좋은 방법은 없기에, 몇 가지 예시를 더 확인하겠습니다.

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

### 컨볼루션 레이어에 대한 잔여 연결

잔여 네트워크에 대해 더 알고 싶다면, [Deep Residual Learning for Image Recognition](http://arxiv.org/abs/1512.03385)를 참고하십시오.

```python
from keras.layers import Conv2D, Input

# 3-채널 256x256 이미지에 대한 인풋 텐서
x = Input(shape=(256, 256, 3))
# 3개의 아웃풋 채널(인풋 채널과 동일)을 가진 3x3 컨볼루션
y = Conv2D(3, (3, 3), padding='same')(x)
# 이는 x + y를 반환합니다
z = keras.layers.add([x, y])
```

### 시각 공유 모델

이 모델은 동일한 이미지 처리 모듈을 두 인풋에 대해서 재사용하여, 두 MNIST 숫자가 같은 숫자인지 다른 숫자인지 여부를 분류합니다.

```python
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten
from keras.models import Model

# 우선, 시각 모듈을 정의합니다
digit_input = Input(shape=(27, 27, 1))
x = Conv2D(64, (3, 3))(digit_input)
x = Conv2D(64, (3, 3))(x)
x = MaxPooling2D((2, 2))(x)
out = Flatten()(x)

vision_model = Model(digit_input, out)

# 그리고 숫자 분류 모델을 정의합니다
digit_a = Input(shape=(27, 27, 1))
digit_b = Input(shape=(27, 27, 1))

# 가중치 등을 포함한 시각 모델은 공유됩니다
out_a = vision_model(digit_a)
out_b = vision_model(digit_b)

concatenated = keras.layers.concatenate([out_a, out_b])
out = Dense(1, activation='sigmoid')(concatenated)

classification_model = Model([digit_a, digit_b], out)
```

### 시각적 문답 모델

이 모델은 사진에 대한 자연어 질문을 받았을 때 올바른 한 단어 대답을 고를 수 있습니다.

이 모델은 질문을 벡터로 인코딩하고 이미지를 벡터를 인코딩하여 둘을 연결시킨 후, 잠재적 대답의 어휘 목록에 대해서 상층의 로지스틱 회귀를 학습시키는 방식으로 작동합니다.

```python
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model, Sequential

# 우선 Sequential 모델을 사용해서 시각 모델을 정의합시다
# 다음 모델은 이미지를 벡터로 인코딩합니다
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

# 이제 시각 모델의 아웃풋으로 텐서를 얻어 봅시다:
image_input = Input(shape=(224, 224, 3))
encoded_image = vision_model(image_input)

# 다음은 문제를 벡터로 인코딩할 언어 모델을 정의합니다
# 각 질문의 최대 길이는 100 단어입니다
# 그리고 단어는 1에서 9999까지의 정수로 색인을 부여받습니다
question_input = Input(shape=(100,), dtype='int32')
embedded_question = Embedding(input_dim=10000, output_dim=256, input_length=100)(question_input)
encoded_question = LSTM(256)(embedded_question)

# 질문 벡터와 이미지 벡터를 연결해 봅시다:
merged = keras.layers.concatenate([encoded_question, encoded_image])

# 그리고 상층의 로지스틱 회귀를 1000 단어에 대해 학습시킵니다:
output = Dense(1000, activation='softmax')(merged)

# 다음은 최종 모델입니다:
vqa_model = Model(inputs=[image_input, question_input], outputs=output)

# 다음 단계는 실제 데이터에 대해 이 모델을 학습시키는 것입니다
```

### 비디오 문답 모델

이미 이미지 문답 모델을 학습시켰으니, 이를 간단하게 비디오 문답 모델로 바꿀 수 있습니다. 적절한 학습을 통해서, 모델에 짧은 비디오(예. 100-프레임 사람 행동)를 보여주고 비디오에 대한 자연어 질문을 할 수 있습니다 (예. "남자 아이는 무슨 스포츠를 하고 있니?" -> "축구").

```python
from keras.layers import TimeDistributed

video_input = Input(shape=(100, 224, 224, 3))
# 기존에 학습된 (가중치가 재사용된) vision_model을 통해서 인코딩된 비디오입니다
encoded_frame_sequence = TimeDistributed(vision_model)(video_input)  # 아웃풋은 벡터의 시퀀스가 됩니다
encoded_video = LSTM(256)(encoded_frame_sequence)  # 아웃풋은 벡터입니다

# 다음은 전과 동일한 가중치를 재사용한 질문 인코더의 모델-수준 표현입니다:
question_encoder = Model(inputs=question_input, outputs=encoded_question)

# 이를 사용해서 질문을 인코딩해 봅시다:
video_question_input = Input(shape=(100,), dtype='int32')
encoded_video_question = question_encoder(video_question_input)

# 그리고 다음은 비디오 문답 모델입니다:
merged = keras.layers.concatenate([encoded_video, encoded_video_question])
output = Dense(1000, activation='softmax')(merged)
video_qa_model = Model(inputs=[video_input, video_question_input], outputs=output)
```

# 케라스: 파이썬 딥러닝 라이브러리

<img src='https://s3.amazonaws.com/keras.io/img/keras-logo-2018-large-1200.png' style='max-width: 600px; width: 90%;' />



## 케라스에 오신걸 환영합니다.

케라스는 파이썬으로 작성된 고수준 신경망 API로 [TensorFlow](https://github.com/tensorflow/tensorflow), [CNTK](https://github.com/Microsoft/cntk), 혹은 [Theano](https://github.com/Theano/Theano)와 함께 사용하실 수 있습니다. 빠른 실험에 특히 중점을 두고 있습니다. *아이디어를 결과물로 최대한 빠르게 구현하는 것은 훌륭한 연구의 핵심입니다.*

케라스를 사용하면:

- (사용자 친화성, 모듈성, 확장성을 통해) 빠르고 간편한 프로토타이핑을 할 수 있습니다.
- 컨볼루션 신경망, 순환 신경망, 그리고 둘의 조합까지 모두 지원됩니다.
- CPU와 GPU에서 매끄럽게 실행됩니다.

[Keras.io](https://keras.io)의 사용설명서를 참고하십시오.

케라스는 다음의 파이썬 버전과 호환됩니다: __Python 2.7-3.6__.


------------------

## Multi-backend Keras and tf.keras:

**At this time, we recommend that Keras users who use multi-backend Keras with the TensorFlow backend switch to `tf.keras` in TensorFlow 2.0**. `tf.keras` is better maintained and has better integration with TensorFlow features (eager execution, distribution support and other).
Keras 2.2.5 was the last release of Keras implementing the 2.2.* API. It was the last release to only support TensorFlow 1 (as well as Theano and CNTK).
The current release is Keras 2.3.0, which makes significant API changes and add support for TensorFlow 2.0. The 2.3.0 release will be the last major release of multi-backend Keras. Multi-backend Keras is superseded by `tf.keras`.
Bugs present in multi-backend Keras will only be fixed until April 2020 (as part of minor releases).
For more information about the future of Keras, see [the Keras meeting notes](http://bit.ly/keras-meeting-notes).


------------------


## 이념

- __사용자 친화성.__ 케라스는 기계가 아닌 사람을 위해 디자인된 API입니다. 무엇보다 사용자 경험을 우선으로 두는 것입니다. 케라스는 사용자가 쉽게 접근할 수 있도록 다음의 원칙을 지킵니다: 일관성있고 간단한 API를 제공하고, 자주 쓰는 기능에 대해서 사용자의 액션을 최소화 하며, 사용자 에러가 발생하는 경우 명료하고 실행가능한 피드백을 줍니다.

- __모듈성.__ 모델은, 최소한의 제한으로 다양한 조합이 가능한 독립적이며 완전히 변경가능한 모듈의 시퀀스 혹은 그래프로 이해할 수 있습니다. 특히 신경망 레이어, 손실 함수, 옵티마이저, 최초값 설정 규칙, 활성화 함수, 정규화 규칙은 모두 독립적인 모듈로, 다양하게 조합하여 새로운 모델을 만들어 낼 수 있습니다.

- __쉬운 확장성.__ 새로운 모듈은 (새로운 클래스와 함수의 형태로) 간단하게 추가할 수 있으며, 기존의 모듈이 풍부한 범례를 제공합니다. 새로운 모듈을 쉽게 만들어낼 수 있다는 점은 표현력을 풍부하게 해주기에, 케라스는 고급 연구를 하기에 적합합니다.

- __파이썬과의 호환.__ 모델 구성 파일을 선언형 포맷으로 따로 두거나 하지 않습니다. 모델은 파이썬 코드로 작성되어 간결하고, 오류를 수정하기 쉬우며, 확장성이 뛰어납니다.


------------------


## 시작하기: 30초 케라스

케라스의 주요 데이터 구조는 __model__,로 레이어를 조직하는 방식입니다. 가장 간단한 종류의 모델인 [`Sequential`](https://keras.io/getting-started/sequential-model-guide) 모델은 레이어를 선형적으로 쌓습니다. 보다 복잡한 구조를 만드려면, [Keras functional API](https://keras.io/getting-started/functional-api-guide)를 사용하여 레이어로 임의의 그래프를 구축하면 됩니다.

`Sequential` 모델입니다:

```python
from keras.models import Sequential

model = Sequential()
```

`.add()`를 통해 레이어를 간단하게 쌓을 수 있습니다 :

```python
from keras.layers import Dense

model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))
```

모델이 마음에 드신다면, `.compile()`로 학습 과정을 조정하십시오:

```python
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
```

필요한 경우 한 걸음 나아가 옵티마이저를 조정할 수 있습니다. 케라스의 주요 철학중 하나는 사용자가 필요한 작업에 대해서는 완전한 제어권을 가질 수 있도록 하되 간결성을 유지하는 것입니다 (제어권의 궁극적인 형태로는 소스코드의 간편한 확장성이 있습니다).
```python
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))
```

배치의 형태로 트레이닝 데이터에 대한 반복작업을 수행할 수 있습니다:

```python
# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
model.fit(x_train, y_train, epochs=5, batch_size=32)
```

혹은, 모델에 배치를 수동으로 전달할 수도 있습니다:

```python
model.train_on_batch(x_batch, y_batch)
```

코드 한 줄로 모델의 성능을 평가해 보십시오:

```python
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
```

혹은 새로운 데이터에 대해서 예측 결과를 생성해 보십시오:

```python
classes = model.predict(x_test, batch_size=128)
```

질문에 대답하는 시스템, 이미지 분류 모델, 신경망 튜링 기계나 그 외의 다른 모델도 이처럼 빠르게 만들 수 있습니다. 딥러닝의 기본이 되는 아이디어가 간단한데 그 실행이 복잡할 이유가 어디 있겠습니까?

조금 더 심화된 케라스 튜토리얼을 원하신다면 다음을 참고하십시오:

- [Getting started with the Sequential model](https://keras.io/getting-started/sequential-model-guide)
- [Getting started with the functional API](https://keras.io/getting-started/functional-api-guide)

리포지토리의 [examples folder](https://github.com/keras-team/keras/tree/master/examples)에서는, 보다 고급 모델을 확인할 수 있습니다: 질문에 대답하는 메모리 신경망, 복수의 LSTM을 이용한 문서 생성 등.


------------------


## 설치

케라스를 설치하기 전에, 다음의 백엔드 엔진 중 하나를 설치하십시오: TensorFlow, Theano, 혹은 CNTK. TensorFlow 백엔드를 사용할 것을 추천드립니다.

- [TensorFlow 설치 설명서](https://www.tensorflow.org/install/).
- [Theano 설치 설명서](http://deeplearning.net/software/theano/install.html#install).
- [CNTK 설치 설명서](https://docs.microsoft.com/en-us/cognitive-toolkit/setup-cntk-on-your-machine).

다음의 **선택적 종속**을 설치하는 것도 고려해 보십시오:

- [cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/) (GPU에 케라스를 실행하실 경우 추천드립니다).
- HDF5 and [h5py](http://docs.h5py.org/en/latest/build.html) (디스크에 케라스 모델을 저장하실 경우 필요합니다).
- [graphviz](https://graphviz.gitlab.io/download/)와 [pydot](https://github.com/erocarrera/pydot) (모델 그래프를 시각화하는 [visualization utilities](https://keras.io/visualization/)에 사용됩니다).

이제 케라스를 설치하시면 됩니다. 케라스 설치에는 두가지 방법이 있습니다:

- **PyPI에서 케라스 설치하기 (추천):**

Note: These installation steps assume that you are on a Linux or Mac environment.
If you are on Windows, you will need to remove `sudo` to run the commands below.

```sh
sudo pip install keras
```

만약 virtualenv를 사용하신다면 sudo를 사용하는 것은 피하는 편이 좋습니다:

```sh
pip install keras
```

- **혹은: 깃허브 소스에서 케라스 설치하기:**

우선 `git`을 사용하여 케라스를 클론하십시오:

```sh
git clone https://github.com/keras-team/keras.git
```

 그리고는 `cd`를 통해 케라스 폴더로 이동한 후 설치명령을 실행하십시오:
```sh
cd keras
sudo python setup.py install
```

------------------


## 케라스 백엔드 조정

케라스는 텐서 처리 라이브러리로 텐서플로우를 사용하도록 디폴트 설정되어 있습니다. 케라스의 백엔드를 조정하려면 [이 설명서를 따라주십시오](https://keras.io/backend/).

------------------


## 지원

다음을 통해서 개발 논의에 참여하거나 질문을 주실 수 있습니다:

- [케라스 구글 그룹](https://groups.google.com/forum/#!forum/keras-users).
- [케라스 슬랙 채널](https://kerasteam.slack.com). [이 링크](https://keras-slack-autojoin.herokuapp.com/)를 사용해서 케라스 슬랙 채널에의 초대를 신청하시면 됩니다.

또한 [GitHub issues](https://github.com/keras-team/keras/issues)에 (유일한 창구입니다) **버그 리포트와 기능 요청**을 올리실 수 있습니다. 먼저 [가이드라인](https://github.com/keras-team/keras/blob/master/CONTRIBUTING.md)을 꼭 참고해주십시오.


------------------


## 왜 케라스라는 이름인가요?

케라스(κέρας)는 그리스어로 _뿔_ 이라는 뜻입니다. _Odyssey_ 에서 최초로 언급된, 고대 그리스와 라틴 문학의 신화적 존재에 대한 이야기로, 두 가지 꿈의 정령(_Oneiroi_, 단수 _Oneiros_) 중 하나는 상아문을 통해 땅으로 내려와 거짓된 환상으로 사람을 속이며, 다른 하나는 뿔을 통해 내려와 앞으로 벌어질 미래를 예언합니다. 이는 κέρας (뿔) / κραίνω (이뤄지다), 그리고 ἐλέφας (상아) / ἐλεφαίρομαι (속이다)에 대한 언어유희이기도 합니다.

케라스는 초기에 ONEIROS(Open-ended Neuro-Electronic Intelligent Robot Operating System)라는 프로젝트의 일환으로 개발되었습니다.

>_"Oneiroi are beyond our unravelling --who can be sure what tale they tell? Not all that men look for comes to pass. Two gates there are that give passage to fleeting Oneiroi; one is made of horn, one of ivory. The Oneiroi that pass through sawn ivory are deceitful, bearing a message that will not be fulfilled; those that come out through polished horn have truth behind them, to be accomplished for men who see them."_ Homer, Odyssey 19. 562 ff (Shewring translation).

------------------

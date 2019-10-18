# 케라스: 파이썬 딥러닝 라이브러리

<img src='https://s3.amazonaws.com/keras.io/img/keras-logo-2018-large-1200.png' style='max-width: 600px; width: 90%;' />



## 케라스에 오신걸 환영합니다.

케라스는 파이썬으로 작성된 고수준 신경망 API로 [TensorFlow](https://github.com/tensorflow/tensorflow), [CNTK](https://github.com/Microsoft/cntk), 혹은 [Theano](https://github.com/Theano/Theano)와 함께 사용하실 수 있습니다. 빠른 실험에 특히 중점을 두고 있습니다. *아이디어를 결과물로 최대한 빠르게 구현하는 것은 훌륭한 연구의 핵심입니다.*

다음과 같은 딥러닝 라이브러리가 필요한 경우에 케라스를 사용하면 좋습니다. 

- (사용자 친화성, 모듈성, 확장성을 통해)빠르고 간편한 프로토타이핑을 할 수 있습니다.
- 합성곱 신경망<sub>convolutional networks</sub>, 순환 신경망<sub>recurrent networks</sub>, 그리고 둘의 조합까지 모두 지원됩니다.
- CPU와 GPU에서 매끄럽게 실행됩니다.

[Keras.io](https://keras.io)의 사용설명서를 참고하십시오.

케라스는 다음의 파이썬 버전과 호환됩니다: __Python 2.7-3.6__.


------------------

## 다중 백엔드 Keras 및 tf.keras
**TensorFlow 백엔드와 함께 다중 백엔드 케라스<sub>multi-backend Keras</sub>를 사용하고 있는 사용자는 TensorFlow 2.0의 `tf.keras`로 전환하기를 권장합니다.** `tf.keras`는 TensorFlow의 기능들 (즉시실행, 배포 지원 등)과 잘 호환되도록 유지, 관리되고 있습니다.
Keras 2.2.5는 2.2.* API를 구현 한 Keras의 마지막 릴리스입니다. TensorFlow 1(그리고 Theano 및 CNTK)을 지원하는 마지막 릴리스입니다. 
현재 릴리스는 Keras 2.3.0입니다. API가 크게 변경되었고 TensorFlow 2.0에 대한 지원이 추가되었습니다. 2.3.0 릴리스는 멀티 백엔드 Keras의 마지막 major release입니다. 다중 백엔드 케라스는 `tf.keras`로 대체되었습니다. 
다중 백엔드 케라스에 존재하는 버그는 2020 년 4 월까지만 minor release로 수정될 예정입니다. 
케라스의 미래에 대한 자세한 내용은 [케라스 회의 노트](http://bit.ly/keras-meeting-notes)를 참조하십시오.

------------------


## 이념

- __사용자 친화성.__ 케라스는 기계가 아닌 사람을 위해 디자인된 API입니다. 무엇보다 사용자 경험을 우선으로 두는 것입니다. 케라스는 사용자가 쉽게 접근할 수 있도록 다음의 원칙을 지킵니다. 일관성있고 간단한 API를 제공하고, 자주 쓰는 기능에 대해서 사용자의 액션을 최소화하며, 사용자 에러가 발생하는 경우 명료하고 실행가능한 피드백을 줍니다.

- __모듈성.__ 모델은, 최소한의 제한으로 다양한 조합이 가능한 독립적이며 완전히 변경가능한 모듈의 시퀀스 혹은 그래프로 이해할 수 있습니다. 특히 신경망 층<sub>neural layers</sub>, 손실 함수<sub>cost functions</sub>, 최적화 함수<sub>optimizer</sub>, 최초값 설정 규칙<sub>initialization schemes</sub>, 활성화 함수<sub>activation functions</sub>, 정규화 규칙<sub>regularization schemes</sub>은 모두 독립적인 모듈로, 다양하게 조합하여 새로운 모델을 만들어 낼 수 있습니다.

- __쉬운 확장성.__ 새로운 모듈은 (새로운 클래스와 함수의 형태로) 간단하게 추가할 수 있으며, 기존의 모듈이 풍부한 범례를 제공합니다. 새로운 모듈을 쉽게 만들어낼 수 있다는 점은 표현력을 풍부하게 해주기에, 케라스는 고급 연구를 하기에 적합합니다.

- __파이썬과의 호환.__ 모델 구성 파일을 선언형 포맷으로 따로 두거나 하지 않습니다. 모델은 파이썬 코드로 작성되어 간결하고, 오류를 수정하기 쉬우며, 확장성이 뛰어납니다.


------------------


## 시작하기: 30초 케라스

케라스의 주요 데이터 구조는 __model__,로 층<sub>layer</sub>을 조직하는 방식입니다. 가장 간단한 종류의 모델인 [`Sequential`](https://keras.io/getting-started/sequential-model-guide) 모델은 층을 선형적으로 쌓습니다. 보다 복잡한 구조를 만드려면, [Keras functional API](https://keras.io/getting-started/functional-api-guide)를 사용하여 층으로 임의의 그래프를 구축하면 됩니다.

`Sequential` 모델입니다.

```python
from keras.models import Sequential

model = Sequential()
```

`.add()`를 통해 층을 간단하게 쌓을 수 있습니다.

```python
from keras.layers import Dense

model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))
```

모델이 마음에 드신다면, `.compile()`로 학습 과정에 대한 설정을 하십시오.

```python
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
```

필요한 경우에 최적화 함수에 대한 설정도 할 수 있습니다. 케라스의 주요 철학중 하나는 사용자가 필요한 작업에 대해서는 완전한 제어권을 가질 수 있도록 하되 간결성을 유지하는 것입니다 (제어권의 궁극적인 형태로는 소스코드의 간편한 확장성이 있습니다).
```python
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))
```

배치<sub>batch</sub>의 형태로 학습 데이터에 대한 반복작업을 수행할 수 있습니다.

```python
# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
model.fit(x_train, y_train, epochs=5, batch_size=32)
```

혹은, 모델에 배치를 수동으로 전달할 수도 있습니다.

```python
model.train_on_batch(x_batch, y_batch)
```

코드 한 줄로 모델의 성능을 평가해 보십시오.

```python
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
```

혹은 새로운 데이터에 대해서 예측 결과를 생성해 보십시오.

```python
classes = model.predict(x_test, batch_size=128)
```

질문에 대답하는 시스템, 이미지 분류 모델, 신경망 튜링 기계나 그 외의 다른 모델도 이처럼 빠르게 만들 수 있습니다. 딥러닝의 기본이 되는 아이디어가 간단한데 그 실행이 복잡할 이유가 어디 있겠습니까?

좀더 심화된 케라스 튜토리얼을 원하신다면 다음을 참고하십시오.

- [Getting started with the Sequential model](https://keras.io/getting-started/sequential-model-guide)
- [Getting started with the functional API](https://keras.io/getting-started/functional-api-guide)

저장소<sub>repository</sub>의 [examples 폴더](https://github.com/keras-team/keras/tree/master/examples)에서는, 보다 고급 모델을 확인할 수 있습니다. 질문에 대답하는 메모리 신경망<sub>memory networks</sub>, 복수의 LSTM을 이용한 문서 생성 등이 있습니다.


------------------


## 설치

케라스를 설치하기 전에, 다음의 백엔드 엔진 중 하나를 설치하십시오: TensorFlow, Theano, 혹은 CNTK. TensorFlow 백엔드를 사용할 것을 추천드립니다.

- [TensorFlow 설치 설명서](https://www.tensorflow.org/install/).
- [Theano 설치 설명서](http://deeplearning.net/software/theano/install.html#install).
- [CNTK 설치 설명서](https://docs.microsoft.com/en-us/cognitive-toolkit/setup-cntk-on-your-machine).

다음의 **선택적 종속**을 설치하는 것도 고려해 보십시오.

- [cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/) (GPU에 케라스를 실행하실 경우 추천드립니다).
- HDF5 and [h5py](http://docs.h5py.org/en/latest/build.html) (디스크에 케라스 모델을 저장하실 경우 필요합니다).
- [graphviz](https://graphviz.gitlab.io/download/)와 [pydot](https://github.com/erocarrera/pydot) (모델 그래프를 시각화하는 [visualization utilities](https://keras.io/visualization/)에 사용됩니다).

이제 케라스를 설치하시면 됩니다. 케라스 설치에는 두가지 방법이 있습니다.

- **PyPI에서 케라스 설치하기(추천)**

참고: 이 설치 단계는 사용자가 Linux 또는 Mac 환경에 있다고 가정합니다. Windows 사용자는 sudo를 빼고 아래의 명령을 실행해야 합니다.

```sh
sudo pip install keras
```

만약 virtualenv를 사용하신다면 sudo를 사용하는 것은 피하는 편이 좋습니다.

```sh
pip install keras
```

- **대안: 깃허브 소스를 통해 케라스 설치하는 방법입니다.**

먼저 `git`명령어를 사용하여 케라스를 clone하십시오.

```sh
git clone https://github.com/keras-team/keras.git
```

그리고 `cd`명령어를 사용하여 케라스 폴더로 이동한 후 install 명령을 실행하십시오.
```sh
cd keras
sudo python setup.py install
```

------------------


## 케라스 백엔드 설정

케라스는 텐서를 처리하는 라이브러리로 Tensorflow를 기본으로 사용합니다. 케라스 백엔드의 설정에 대해서는 [이 설명서](https://keras.io/backend/)를 따라주십시오.

------------------


## 지원

다음을 통해서 개발 논의에 참여하거나 질문을 주실 수 있습니다.

- [케라스 구글 그룹](https://groups.google.com/forum/#!forum/keras-users).
- [케라스 슬랙 채널](https://kerasteam.slack.com). [이 링크](https://keras-slack-autojoin.herokuapp.com/)를 사용해서 케라스 슬랙 채널에의 초대를 신청하시면 됩니다.

또한 [GitHub issues](https://github.com/keras-team/keras/issues)에 (유일한 창구입니다) **버그 리포트와 기능 요청**을 올리실 수 있습니다. 먼저 [가이드라인](https://github.com/keras-team/keras/blob/master/CONTRIBUTING.md)을 꼭 참고해주십시오.


------------------


## 왜 케라스라는 이름인가요?

케라스(κέρας)는 그리스어로 _뿔_ 이라는 뜻입니다. _Odyssey_ 에서 최초로 언급된, 고대 그리스와 라틴 문학의 신화적 존재에 대한 이야기로, 두 가지 꿈의 정령(_Oneiroi_, 단수 _Oneiros_) 중 하나는 상아문을 통해 땅으로 내려와 거짓된 환상으로 사람을 속이며, 다른 하나는 뿔을 통해 내려와 앞으로 벌어질 미래를 예언합니다. 이는 κέρας (뿔) / κραίνω (이뤄지다), 그리고 ἐλέφας (상아) / ἐλεφαίρομαι (속이다)에 대한 언어유희이기도 합니다.

케라스는 초기에 ONEIROS(Open-ended Neuro-Electronic Intelligent Robot Operating System)라는 프로젝트의 일환으로 개발되었습니다.

>_"Oneiroi are beyond our unravelling --who can be sure what tale they tell? Not all that men look for comes to pass. Two gates there are that give passage to fleeting Oneiroi; one is made of horn, one of ivory. The Oneiroi that pass through sawn ivory are deceitful, bearing a message that will not be fulfilled; those that come out through polished horn have truth behind them, to be accomplished for men who see them."_ Homer, Odyssey 19. 562 ff (Shewring translation).

------------------

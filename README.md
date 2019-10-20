# Korean translation of the Keras documentation

This is the repository for the Korean-language `.md` sources files of [keras.io](https://keras.io).

Existing files in `sources/` should be edited in-line.

---

# 케라스 공식 문서 한국어판
케라스 공식 문서의 한국어판입니다. 이미 딥러닝에 익숙한 연구자 및 개발자 외에도 처음 딥러닝을 접하는 사용자들이 최대한 쉽게 이해하고 사용할 수 있도록 그 의미와 용법, 용례가 정확하고 명료하게 그리고 최대한 자연스러운 문장으로 나타나도록 작성되었습니다. :open_book::writing_hand::full_moon_with_face:

## 번역 가이드라인

- 모든 번역문은 **한국어 정서법**을 준수합니다.
- 번역은 문서화 내에 있는 본문 내용과 코드 주석들을 대상으로 합니다.
- 번역시 문장 끝에 붙는 격식체는 '-ㅂ니다'체를 따르며 비속어나 반말은 쓰지 않습니다.
- 큰 따옴표나 작은 따옴표는('，") 특수문자를 사용하지 않고 기본적으로 제공된 것을 사용합니다.
- 코드 강조(syntax highlight) 뒤에 조사가 붙는 경우, 공백을 넣지 않습니다(e.g. `model.fit()`을 실행하면).
- 키워드를 번역할 때 아래에 있는 **작성 규칙** 및 **용어 통일안**을 최우선으로 사용합니다.
- 과한 복문의 경우 단문으로 나누어서 씁니다.
- 원문 내용이 불충분한 경우 원문이 전달하고자 하는 내용을 충실히 전달하는 범위 내에서 사용자가 이해할 수 있도록 간략한 설명을 보충합니다.
- 번역은 다른 언어로 된 문서의 의미를 이해하고 한국어로 다시 표현하는 것이니 번역체는 자제해 주시기 바랍니다(~~우리는 한다 번역을~~).

---

## 작성 규칙

- 용어 번역의 경우 문서 내에서 처음 나온 경우에 한해 subscript로 원어를 병행표기합니다. (예: 층<sub>layer</sub>)
  - 발음만 한글로 옮긴 경우 subscript는 생략합니다. (예: 스트라이드)
  - 특수한 경우를 제외하면 subscript는 소문자로 작성합니다. (특수한 경우: 1. 대문자 고유명사 및 대문자 약칭, 2. 제목의 경우 관사와 접속사, 전치사를 제외한 단어와 제목 첫 단어의 첫글자는 대문자로 작성)
- list, dict 등 파이썬 기본 자료형의 경우 발음대로 표기하고 원어는 병기하지 않습니다.
- int, float, integer 등 자료형 키워드/단어의 경우 
  - 문장 내에 등장하는 경우 한국어로 번역합니다. (예: "~ is tuple of integers" → "~는 정수형 튜플입니다.")
  - argument등 변수 설명에서 입력값의 자료형을 나타내는 경우 highlight로 표시하고 파이썬 자료형 표기대로 적습니다. (예: __X__: Integer, → `int`.)
- 문장 끝의 colon(:)은 마침표로 대체합니다.
  - 문장 끝의 semicolon(;)은 문장을 두 개로 분리하고 필요한 경우 적합한 접속사를 추가합니다.
- Keras를 제외한 모든 API 및 서비스 등의 이름(TensorFlow, NumPy, CNTK, Amazon, Google 등)은 원문 그대로 사용합니다
- 함수 인자 설명시 [__인자__: `data type`, 설명 내용, 기본값 ]의 형식을 따릅니다. (예: __batch_size__: `int` 혹은 `None`. 손실로부터 그래디언트를 구하고 가중치를 업데이트하는 과정 한 번에 사용할 표본의 개수입니다. 기본값은 `32`입니다.)
- **Raises**란의 경우 **오류**로 번역하며, 본문은 "(~하는 경우, ~하면, ~가) 발생합니다."로 정리합니다.  

---

## 용어 통일안

- 이하 통일안은 케라스 코리아 번역팀의 논의를 거쳐 합의된 표현의 목록입니다.
- 통일안 선정은 다음과 같은 리소스를 참고하였습니다.
  - [국립국어원 언어정보 나눔터 언어 정보화 용어](https://ithub.korean.go.kr/user/word/wordTermManager.do)
  - [국립국어원 표준국어대사전](https://stdict.korean.go.kr/main/main.do) / [우리말샘](https://opendict.korean.go.kr/main)
  - [한국정보통신기술협회 정보통신용어사전](http://word.tta.or.kr/main.do)
  - [대한수학회 수학용어집](http://www.kms.or.kr/mathdict/list.html?key=ename&keyword=norm)
  - [한국어 위키백과](https://ko.wikipedia.org)
- 참조대상이 없는 어휘의 경우 의미를 살리되 보편적으로 사용되는 표현을 우선 선정하였습니다. 서로 다른 대상을 가리키는 번역 어휘가 중복되어 쓰이던 경우 최대한 가까운 새로운 어휘로 대체하였습니다.  
- 용어집은 새로 도출한 합의안과 함께 개정됩니다.


| English            | 한국어                 |
|:-------------------|:-----------------------|
| -er| ~화 함수 / 함수|
| 1--9| 1--9|
| accuracy| 정확도|
| argument| 인자|
| (artificial) neural network| (인공) 신경망|
| augmenter| 증강 함수|
| Average Pooling| 평균 풀링|
| axis| 축|
| batch| 배치|
| bias| 편향|
| binary classification| 이진 분류|
| cache| 캐시|
| callback| 콜백|
| cell state| 셀 상태|
| channel| 채널|
| checkpoint| 체크포인트|
| class| 클래스|
| classification| 분류|
| compile| 컴파일|
| constraint| 제약|
| convolutional neural network (CNN)| 합성곱 신경망|
| corpus| 말뭉치|
| dense layer| 완전연결층|
| dimension| 차원|
| dot product| 내적|
| dropout| 드롭아웃|
| element-wise | 원소별|
| embedding| 임베딩|
| encoding| 인코딩|
| epoch| 에폭 (서술적으로 쓸 때는 'n회 반복')|
| factor| 값/요인/요소|
| fully-connected, densely connected| 완전 연결|
| global| 전역|
| generator| 제너레이터|
| gradient| 그래디언트|
| gradient ascent| 경사상승법|
| gradient descent| 경사하강법|
| hidden unit| 은닉 유닛|
| hidden layer| 은닉 층|
| hidden state| 은닉 상태|
| hyperparameter| 하이퍼파라미터|
| identity matrix| 단위 행렬|
| index| 인덱스 (개별 index의 묶음 전체를 가리킬 때는 '목록')|
| input| 입력/입력값|
| instance| 인스턴스|
| initialization| 초기값 생성|
| initializer| 초기화 함수|
| keras| 케라스|
| kernel| 커널|
| label| 레이블|
| layer| 층|
| learning rate| 학습률|
| learning rate decay| 학습률 감소|
| locally| 부분 연결|
| loss function| 손실 함수|
| LSTM| LSTM|
| MaxPooling| 최댓값 풀링|
| mean squared error (MSE)| 평균 제곱 오차(법)|
| metric| (평가) 지표 (문맥에 따라 유연하게 사용)|
| mini-batch| 미니 배치|
| model| 모델|
| momentum| 모멘텀|
| multi-class classification| 다중 분류|
| multilayer perceptron (MLP)| 다층 퍼셉트론|
| neuron| 뉴런|
| node| 노드|
| noise| 노이즈|
| non-negativity| 음이 아닌 ~|
| norm| 노름|
| normalization| 정규화|
| normalize | 정규화하다|
| note| 참고|
| objective function| 목적 함수|
| one-hot encoding| 원-핫 인코딩|
| optimizer| 최적화 함수|
| output| 출력(값)|
| padding| 패딩|
| parameter| (함수의)매개변수|
| parameter| (모델의)파라미터 (가중치와 편향을 함께 이르는 말)|
| placeholder| 플레이스홀더|
| penalty| 페널티|
| pooling| 풀링|
| precision| 정밀도|
| queue| 대기열|
| recurrent neural network (RNN)| 순환 신경망|
| reference| 참고|
| regression| 회귀 분석|
| regression(-ive) model| 회귀 모델|
| regularize(-er)| 규제화/규제 함수|
| repository| 저장소|
| reshape| 형태바꾸기|
| return| 반환값|
| root mean squared error (RMSE)| 평균 제곱근 오차(법)|
| sample| 표본|
| sequence (-tial)| 순서형|
| set| 세트|
| shape| 형태|
| stack| 층을 쌓다|
| stateful| 상태 저장|
| stochastic gradient descent| 확률적 경사하강법|
| stride| 스트라이드|
| target| 목표(값)|
| temporal| 시계열|
| tensor| 텐서|
| test| 시험|
| text| 텍스트|
| timestep| 시간 단계/순서|
| token| 토큰|
| train| (데이터의 경우) 훈련 세트 / (동작의 경우) 학습시키다|
| utility| 도구|
| validation| 검증|
| weight| 가중치|
| wrapper| 래퍼|

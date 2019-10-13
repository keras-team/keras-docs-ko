# 순서형 전처리<sub>Sequence Preprocessing</sub> 
순서형 전처리 도구 모듈입니다.  

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/preprocessing/sequence.py#L16)</span>
### TimeseriesGenerator
```python
keras.preprocessing.sequence.TimeseriesGenerator(data, targets, length, sampling_rate=1, stride=1, start_index=0, end_index=None, shuffle=False, reverse=False, batch_size=128)
```
시간 순서가 있는 데이터의 배치<sub>Batch</sub>를 생성하는 도구 클래스<sub>Class</sub>입니다.  

이 클래스는 일정한 간격으로 수집된 시계열 데이터와 전체 길이, 스트라이드<sub>Stride</sub>와 같이 시계열 특성을 나타내는 매개변수<sub>parameter</sub>를 입력받아 훈련/검증에 사용할 배치 데이터를 생성합니다.

__인자__
- __data__: 리스트 또는 NumPy 배열과 같이 인덱싱 가능한 2D 데이터로 0번째 축<sub>Axis</sub>은 시간차원을 나타냅니다.
- __targets__: `data`의 시간 단계와 상응하는 목표값으로 0번째 축의 길이가 `data`와 서로 같아야 합니다.
- __length__: 생성할 배치의 시계열 길이를 지정합니다. 해당 인자를 통해 지정되는 길이는 최대 길이로서, 각 표본<sub>Sample</sub>의 실제 길이는 `length`를 `sampling_rate`로 나눈 몫만큼이 됩니다.
- __sampling_rate__: `length`를 통해 지정된 시계열 범위 가운데 `sampling_rate` 시점마다 입력값을 추출해서 배치에 포함시킬 것인지를 정합니다. 예를 들어 표본이 `i`번째 데이터에서 시작할 때 `sampling_rate`를 `r`로 설정할 경우 생성되는 표본은 `data[i]`, `data[i+r]`, `data[i+2r]`... 의 형태가 되며, 표본의 최종 길이는 `length`를 `sampling_rate`로 나눈 몫이 됩니다. 기본값은 `1`이며, 이 경우 배치의 길이는 `length`와 같아집니다. 
- __stride__: 입력값 가운데 `stride`로 지정한 순서마다 표본을 생성합니다. 예를 들어 첫번째 시계열 표본이 `i`번째 입력값에서 시작할 때 `stride`가 `s`면 다음 표본은 `data[i+s]`부터, 그 다음 표본은 `data[i+2s]`부터 생성됩니다. 표본 사이에 데이터가 중복되지 않게 하려면 `stride`값을 `length`보다 같거나 크게 지정하면 됩니다. 기본값은 `1`입니다.
- __start_index__: 입력값 가운데 배치 생성에 사용할 최초 시점을 지정합니다. `start_index`이전의 데이터는 사용되지 않기 때문에 별도의 시험/검증 세트를 만드는 데 활용할 수 있습니다. 기본값은 `0`입니다.
- __end_index__: 입력값 가운데 배치 생성에 사용할 마지막 시점을 지정합니다. `end_index`이후의 데이터는 사용되지 않기 때문에 별도의 시험/검증 세트를 만드는 데 활용할 수 있습니다. 기본값은 `None`으로, 이 경우 입력 데이터의 가장 마지막 인덱스가 자동으로 지정됩니다.
- __shuffle__: `bool`. `True`인 경우, 생성한 표본의 순서를 뒤섞습니다. 기본값은 `False`입니다. 
- __reverse__: `bool`. `True`인 경우, 생성한 표본의 순서는 입력된 데이터의 역순입니다. 기본값은 `False`입니다. 
- __batch_size__: 하나의 배치 안에 포함될 표본의 개수입니다. 기본값은 `128`입니다.

__반환값__
[Sequence](/utils/#sequence) 인스턴스.

__예시__
```python
from keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np

data = np.array([[i] for i in range(50)])
targets = np.array([[i] for i in range(50)])

data_gen = TimeseriesGenerator(data, targets,
                               length=10, sampling_rate=2,
                               batch_size=2)
assert len(data_gen) == 20

batch_0 = data_gen[0]
x, y = batch_0
assert np.array_equal(x,
                      np.array([[[0], [2], [4], [6], [8]],
                                [[1], [3], [5], [7], [9]]]))
assert np.array_equal(y,
                      np.array([[10], [11]]))
```
    
----
### pad_sequences
```python
keras.preprocessing.sequence.pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.0)
```
입력값의 길이를 패딩<sub>Padding</sub>하여 동일하게 만듭니다.

이 함수<sub>Function</sub>는 서로 다른 길이를 가진 `num_samples`개의 리스트에 패딩을 더하여 전체 길이가 `num_timesteps`로 동일한 `(num_samples, num_timesteps)`형태의 2D NumPy 배열로 변형합니다. 패딩을 포함한 `num_timesteps`의 길이는 `maxlen`인자에 의해 결정되며, `maxlen`인자를 지정하지 않을 경우에는 전체 입력 리스트 가운데 가장 긴 리스트를 기준으로 맞춰집니다. 길이가 `num_timesteps`보다 긴 경우는 잘라냅니다. 패딩에 사용되는 값은 `value` 인자로 정할 수 있으며 기본값은 `0 (float)`입니다.  
  
`padding`과 `truncating`인자는 각각 개별 리스트의 앞/뒤 가운데 어느 부분을 패딩하고 잘라낼지를 결정합니다. 기본값은 `pre`(앞)입니다.

__인자__
- __sequences__: 리스트들로 이루어진 리스트로, 각각의 하위 리스트가 순서형 데이터입니다.
- __maxlen__: `int`. 전체 리스트의 최대 길이를 결정합니다.
- __dtype__: 출력값의 자료형을 결정합니다. 패딩에 문자열을 사용할 경우 `object`타입을 지정합니다. 기본값은 `int32`입니다.
- __padding__: `string`. `'pre'` 또는 `'post'`를 입력받아 각 리스트의 앞 또는 뒤를 패딩할 위치로 지정합니다.
- __truncating__: `string`. `'pre'` 또는 `'post'`를 입력받아 `maxlen`보다 길이가 긴 리스트를 해당 위치에서 잘라냅니다.    
- __value__: 패딩에 사용할 값으로 `float`또는 `string` 형식의 입력을 받습니다. 기본값은 `0 (float)`입니다.

__반환값__
- __x__: `(len(sequences), maxlen)`형태의 NumPy 배열.

__오류__
- __ValueError__: `truncating` 혹은 `padding`에 잘못된 값을 전달한 경우, 또는 `sequences`입력 형식이 잘못된 경우 오류메시지를 출력합니다.
    
----
### skipgrams
```python
keras.preprocessing.sequence.skipgrams(sequence, vocabulary_size, window_size=4, negative_samples=1.0, shuffle=True, categorical=False, sampling_table=None, seed=None)
```
Skipgram 단어 쌍을 생성합니다.

Skipgram은 어떤 문장을 구성하는 각각의 단어들을 '중심 단어'로 지정하고 특정 중심 단어가 등장했을 때 '주변 단어'가 등장할 확률을 말뭉치<sub>Corpus</sub>로부터 학습하는 모델입니다. Skipgram에 대한 보다 자세한 설명은 [Mikolov et al.의 탁월한 논문](http://arxiv.org/pdf/1301.3781v3.pdf)을 참고하십시오.  
  
케라스의 `skipgrams`함수는 단어 인덱스로 이루어진 리스트를 입력받아 중심 단어와 주변 단어의 쌍으로 이루어진 학습용 데이터 튜플을 생성합니다. 튜플은 다음의 두 리스트로 이루어집니다.

- 리스트`0`: 문장 내의 중심 단어와 주변 단어로 이루어진 리스트들의 리스트. 중심 단어의 인덱스를 `i`, `windows_size`인자에서 지정한 '창의 크기'를 `n`이라고 하면 `[i-n]`, `[i-n+1]`, ..., `[i-1]`, `[i+1]`, ..., `[i+n-1]`, `[i+n]`인덱스의 단어들이 각각 중심 단어 `[i]`와 단어 쌍을 만들게 됩니다. 따라서 각 중심 단어마다 `2n`개의 단어쌍을 만들며, 해당 중심 단어가 문장의 시작 또는 끝에 가까워서 어느 한쪽의 주변 단어 개수가 `n`보다 작을 경우 그 방향에 존재하는 주변 단어의 개수만큼 단어쌍이 생성됩니다. 또한 `negative_samples` 인자에서 지정한 비율에 따라 중심 단어와 '가짜 주변 단어'로 이루어진 '거짓 표본' 리스트들이 함께 생성됩니다.
- 리스트`1`: 리스트`0`에 포함된 각 단어쌍이 실제 중심 단어와 주변 단어로 이루어진 '참 표본'인지, 무작위로 선택한 가짜 주변 단어로 이루어진 '거짓 표본'인지를 나타내는 레이블의 리스트입니다. 참인 경우 `1`, 거짓인 경우 `0`값을 갖습니다.

__인자__
- __sequence__: 단어의 인덱스로 이루어진 정수값의 리스트입니다. `skipgrams`함수는 한 번에 한 문장, 즉 하나의 리스트를 입력받습니다. `sampling_table` 인자를 사용하는 경우 각 단어가 말뭉치 내에서 등장하는 빈도의 순위를 해당 단어의 인덱스로 지정해야 합니다. 예를 들어 인덱스 `10`은 열 번째로 많이 등장하는 단어를 나타냅니다. 인덱스 `0`은 단어 배정되지 않는 예비 인덱스이기 때문에 사용해서는 안된다는 점에 유의하십시오.
- __vocabulary_size__: `int`. 학습에 사용하고자 하는 단어 목록의 크기로, 존재하는 단어 인덱스 최댓값 + 1을 배정합니다.
- __window_size__: `int`. 하나의 중심 단어로부터 학습하고자 하는 주변 단어의 범위를 지정합니다. `window_size = n`일 때, 중심단어 `[i]`로부터 최대 `[i-n]`에서 `[i+n]`까지 최대 `2n`개의 주변 단어가 학습 범위에 포함됩니다. 해당 중심 단어가 문장의 시작 또는 끝에 가까워서 어느 한쪽의 주변 단어 개수가 `n`보다 작을 경우 그 방향에 존재하는 주변 단어의 개수만큼 단어쌍이 생성됩니다.
- __negative_samples__: `0`보다 같거나 큰 `float`값으로 실제 주변 단어로부터 생성될 단어쌍 표본의 개수 대비 생성할 거짓 표본의 비율을 나타냅니다. 0일 경우 거짓 표본을 생성하지 않으며 1일 경우 참인 표본과 동일한 개수의 무작위 거짓 표본을 생성합니다.
- __shuffle__: `boolean`. 생성한 단어쌍의 순서를 무작위로 섞을 것인지를 결정합니다. `skipgrams`함수는 입력된 리스트로부터 차례대로 단어쌍을 생성합니다. `False`일 경우 뒤섞지 않고 생성한 순서대로 결과를 반환합니다. 기본값은 `True`입니다. 
- __categorical__: `boolean`. `False`인 경우, 출력된 튜플의 레이블(인덱스`1`)은 정수값을(예: `[0, 1, 1 ...]`), `True`인 경우 범주형 값을(예: `[[1,0], [0,1], [0,1] ...]`) 갖습니다.
- __sampling_table__: `vocabulary_size`만큼의 길이를 갖는 1D 배열로, 배열의 `i`번째 값은 인덱스 값이 `i`인 단어가 단어쌍으로 추출될 확률을 나타냅니다. 자세한 부분은 `make_sampling_table`함수를 참조하십시오.
- __seed__: 난수 생성에 사용할 시드입니다.

__반환값__
단어 인덱스의 쌍(리스트)으로 이루어진 리스트와 각 단어쌍의 참/거짓을 나타내는 레이블 리스트로 이루어진 튜플.

__유의사항__
상례에 따라, 인덱스 `0`은 특정 단어에 배정되지 않는 값으로 단어쌍 생성에 사용되지 않습니다. 
    
----
### make_sampling_table
```python
keras.preprocessing.sequence.make_sampling_table(size, sampling_factor=1e-05)
```
각 단어가 등장할 확률을 높은 순서대로 정렬한 배열을 생성합니다.  
  
이 함수는 `skipgrams`함수의 `sampling_table`인자에 입력할 확률 배열을 생성합니다. `sampling_table[i]`는 데이터에서 `i`번째로 빈번하게 나타나는 단어를 학습 표본으로 추출할 확률입니다. 학습에 사용할 데이터는 각 항목의 가짓수가 비슷한 편이 좋기 때문에 균형을 위해 자주 등장하는 단어일수록 추출될 확률을 낮게 설정합니다. 해당 확률은 word2vec에서 사용하는 표본추출 분포 공식을 따릅니다.
```
p(word) = (min(1, sqrt(word_frequency / sampling_factor) /
          (word_frequency / sampling_factor)))
```
이때 `word_frequency`값은 각 단어가 실제 말뭉치 안에 등장하는 빈도가 아니라 [지프의 법칙](https://ko.wikipedia.org/wiki/%EC%A7%80%ED%94%84%EC%9D%98_%EB%B2%95%EC%B9%99)(s=1)의 가정을 따라 단어의 등장 빈도 순위별로 자동으로 생성되는 근사값을 사용합니다.  
```
frequency(rank) ≈ 1/(rank * (log(rank) + gamma) + 1/2 - 1/(12*rank))
```
여기서 `gamma`는 오일러-마스케로니 상수입니다.

__인자__
- __size__: `int`. 표본 추출에 사용할 전체 단어의 수.
- __sampling_factor__: word2vec 공식에 사용할 표본 추출 인수. 기본값은 `1e-5`입니다.

__반환값__
`size`의 길이를 가진 1D 형태의 NumPy 배열로, `i`번째 값은 `i`번째로 빈번하게 나타나는 단어를 학습 표본으로 추출할 확률입니다.

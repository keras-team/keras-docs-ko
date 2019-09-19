<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/preprocessing/sequence.py#L16)</span>
### TimeseriesGenerator

```python
keras.preprocessing.sequence.TimeseriesGenerator(data, targets, length, sampling_rate=1, stride=1, start_index=0, end_index=None, shuffle=False, reverse=False, batch_size=128)
```

시계열 데이터의 배치<sub>Batch</sub>를 생성하는 유틸리티 클래스.

이 클래스는 일정한 간격으로 수집된 연속된 데이터포인트에 보폭<sub>Stride</sub>이나 길이<sub>Length</sub>와 같이 시간을 나타내는 매개변수<sub>Parameter</sub>를 적용하여 학습/검증에 사용할 배치를 만들어냅니다.

__인수__

- __data__: 연속적인 데이터 포인트(시간 단계)로 
    구성된 (리스트 혹은 Numpy배열과 같은) 색인처리 가증한 생성자.
    데이터는 2D의 형태를 취해야 하며, 0축에는
    시간 차원을 두어야 합니다.
- __targets__: `data` 내 시간 단계에 호응하는 표적.
    `data`와 길이가 동일해야 합니다.
- __length__: 출력 시퀀스의 길이 (단위는 시간 단계의 수).
- __sampling_rate__: 시퀀스 내 연속되는 개별 시간 단계
    사이의 기간. 속도가 `r`이면,
    `data[i]`, `data[i-r]`, ... `data[i - length]`의
    시간 단계로 샘플 시퀀스를 생성합니다.
- __stride__: 연속되는 출력 시퀀스 사이의 기간.
    보폭이 `s`이면, 연속되는 출력 샘플은
    `data[i]`를 중심으로, `data[i+s]`, `data[i+2*s]` 등이 됩니다.
- __start_index__: `start_index` 이전의 데이터 포인트는 출력 시퀀스에서
    사용되지 않습니다. 이는 데이터의 일부를 보존하여
    테스트나 검증에 사용하는데 유용합니다.
- __end_index__: `end_index` 이후의 데이터 포인트는 출력 시퀀스에서
    사용되지 않습니다. 이는 데이터의 일부를 보존하여
    테스트나 검증에 사용하는데 유용합니다.
- __shuffle__: 출력 샘플을 뒤섞을지,
    혹은 시간순으로 뽑을지 여부.
- __reverse__: 불리언: `true`인 경우, 각 출력 샘플의 시간 단계는
    역시간 순이 됩니다.
- __batch_size__: 각 배치의 (마지막을 제외한)
    시간 단계 샘플의 수.

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


시퀀스를 패딩하여 동일한 길이로 만듭니다.

이 함수는 `num_samples`개의 
시퀀스(정수 리스트)의 리스트를
`(num_samples, num_timesteps)` 형태의 2D Numpy 배열로 변형합니다.
`num_timesteps`는 `maxlen` 인수를 전달받는 경우는 `maxlen` 인수이고,
그 외의 경우 가장 긴 시퀀스의 길이가 됩니다.

`num_timesteps` 보다 짧은 시퀀스는
그 끝에 `value` 값이 패딩됩니다.

`num_timesteps`보다 긴 시퀀스는 원하는
길이에 맞도록 절단됩니다.
`padding`과 `truncating` 인수에 따라 각각
패딩과 절단이 어디서 일어나는지 결정됩니다.

디폴트 모드는 프리-패딩입니다.

__인수__

- __sequences__: 리스트의 리스트로, 각 성분이 시퀀스입니다.
- __maxlen__: 정수, 모든 시퀀스의 최대 길이.
- __dtype__: 출력 시퀀스의 자료형.
    가변적 길이의 문자열로 시퀀스를 패딩하려면 `object`를 사용하시면 됩니다.
- __padding__: 문자열, 'pre' 혹은 'post':
    각 시퀀스의 처음 혹은 끝을 패딩합니다.
- __truncating__: 문자열, 'pre' 혹은 'post':
    `maxlen`보다 큰 시퀀스의
    처음 혹은 끝의 값들을 제거합니다.
- __value__: 부동소수점 혹은 문자열, 패딩할 값.

__반환값__

- __x__: `(len(sequences), maxlen)`형태의 Numpy 배열

__오류__

- __ValueError__: `truncating` 혹은 `padding`에 유효하지 않은 값을 전달한 경우,
    혹은 `sequences` 기입 내용이 유효하지 않은 형태일 경우 오류메시지를 전달합니다.
    
----

### skipgrams


```python
keras.preprocessing.sequence.skipgrams(sequence, vocabulary_size, window_size=4, negative_samples=1.0, shuffle=True, categorical=False, sampling_table=None, seed=None)
```


skipgram 단어 쌍을 생성합니다.

이 함수는 단어 색인 시퀀스(정수 리스트)를 다음과 같은 형태의
단어 튜플로 변형합니다:

- (단어, 동일 윈도우의 단어), 라벨은 1 (양성 샘플).
- (단어, 어휘목록에서 뽑은 무작위 단어), 라벨은 0 (음성 샘플).

Skipgram에 대해서 더 자세한 사항은 Mikolov et al.의 탁월한 논문을 참고하십시오:
[Efficient Estimation of Word Representations in
Vector Space](http://arxiv.org/pdf/1301.3781v3.pdf)

__인수__

- __sequence__: 단어색인(정수) 리스트로 인코딩된 
    단어 시퀀스(문장). `sampling_table`을 사용하는 경우,
    단어 색인이 참조 데이터셋의
    단어 순위에 호응합니다 (예. 10이라는 색인은 
    10 번째로 빈번하게 나타나는 토큰을 의미합니다).
    색인 0 은 단어가 아니면 고려되지 않는다는 점을 유의하십시오.
- __vocabulary_size__: 정수, 가능한 단어 색인의 최댓값 + 1
- __window_size__: 정수, 샘플링 윈도우의 크기 (엄밀히 말하자면 절반-윈도우).
    `w_i`라는 단어의 윈도우는
    `[i - window_size, i + window_size+1]`가 됩니다.
- __negative_samples__: 부동소수점 >= 0. 0은 음성 샘플이 없다는 것(다시 말해 무작위)을,
    1은 양성 샘플과 같은 수의 음성샘플이 존재한다는 것을 의미합니다.
- __shuffle__: 단어 커플을 반환하기 전에 뒤섞을지 여부.
- __categorical__: 불리언. 거짓인 경우, 라벨은
    정수(예. `[0, 1, 1 .. ]`)이고,
    `True`인 경우, 라벨은 범주
    (예. `[[1,0],[0,1],[0,1] .. ]`)입니다.
- __sampling_table__: `vocabulary_size` 크기의 1D 배열로, 기입값 i는
    순위가 i인 단어를 샘플링할 확률을 나타냅니다.
- __seed__: 난수 시드.

__반환값__

커플, 라벨: `couples`은 정수 짝이고
    `labels`은 0 혹은 1입니다.

__유의사항__

상례에 따라, 어휘목록의 색인 0 는 단어가 아니며
고려되지 않습니다.
    
----

### make_sampling_table


```python
keras.preprocessing.sequence.make_sampling_table(size, sampling_factor=1e-05)
```


단서 순위 기반의 확률적 샘플링 표를 생성합니다.

`skipgrams`에 사용할 `sampling_table` 인수를 생성하는데
사용됩니다. `sampling_table[i]`는 데이터셋에서 
i 번째로 빈번하게 나타나는 단어를 샘플링할 확률입니다.
(균형을 위해서 빈번한 단어일수록 드물게 샘플링해야 합니다).

샘플링 확률은 word2vec에서 사용되는
샘플링 분포에 따라 생성됩니다:

```
p(word) = (min(1, sqrt(word_frequency / sampling_factor) /
    (word_frequency / sampling_factor)))
```

단어 빈도가 지프의 법칙(s=1)을 따른다는 가정하에
빈도(순위)의 수치 근사값을 유도합니다:

`frequency(rank) ~ 1/(rank * (log(rank) + gamma) + 1/2 - 1/(12*rank))`
여기서 `gamma`는 오일러-마스케로니 상수입니다.

__Arguments__

- __size__: 정수, 샘플링 가능한 단어의 수.
- __sampling_factor__: word2vec 공식의 샘플링 인수.

__반환값__

`size`의 길이를 가진 1D 형태의 Numpy 배열로, i 번째 기입값이
순위 i의 단어가 샘플되어야 할 확률을 나타냅니다.
    

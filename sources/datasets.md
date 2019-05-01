# 데이터 셋

## CIFAR10 소형 이미지 분류

50,000개의 32x32 컬러 학습 이미지, 10개 범주의 라벨, 10,000개의 테스트 이미지로 구성된 데이터셋.

### 사용법:

```python
from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
```

- __반환값:__
    - 2개의 튜플:
        - __x_train, x_test__: RGB 이미지 데이터의 uint8 배열. `channels_first` 이나 `channels_last`의 `image_data_format` 백엔드 세팅에 따라 각각 (num_samples, 3, 32, 32) 혹은 (num_samples, 32, 32, 3)의 형태를 취합니다.
        - __y_train, y_test__: 범주 라벨의 uint8 배열 (0-9 범위의 정수). (num_samples,)의 형태를 취합니다. 


---

## CIFAR100 소형 이미지 분류:

50,000개의 32x32 컬러 학습 이미지, 10개 범주의 라벨, 10,000개의 테스트 이미지로 구성된 데이터셋.

### 사용법:

```python
from keras.datasets import cifar100

(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
```

- __반환값:__
    - 2개의 튜플:
        - __x_train, x_test__: RGB 이미지 데이터의 uint8 배열. `channels_first` 이나 `channels_last`의 `image_data_format` 백엔드 세팅에 따라 각각 (num_samples, 3, 32, 32) 혹은 (num_samples, 32, 32, 3)의 형태를 취합니다.
        - __y_train, y_test__: 범주 라벨의 uint8 배열 (0-9 범위의 정수). (num_samples,)의 형태를 취합니다.

- __인수:__

    - __label_mode__: "fine" 혹은 "coarse".


---

## IMDB 영화 리뷰 감정 분류:

감정에 따라 (긍정적/부정적)으로 라벨된 25,000개의 IMDB 영화 리뷰로 구성된 데이터셋. 리뷰는 선행처리되었으며, 각 리뷰는 단어 인덱스(정수)로 구성된 [sequence](preprocessing/sequence.md)로 인코딩 되었습니다. 편의를 위해 단어는 데이터내 전체적 사용빈도에 따라 인덱스화 되었습니다. 예를 들어, 정수 "3"은 데이터 내에서 세 번째로 빈번하게 사용된 단어를 나타냅니다. 이는 "가장 빈번하게 사용된 10,000 단어만을 고려하되 가장 많이 쓰인 20 단어는 제외"와 같은 빠른 필터링 작업을 가능케 합니다.

관습에 따라 "0"은 특정 단어를 나타내는 것이 아니라 미확인 단어를 통칭합니다. 

### 사용법:

```python
from keras.datasets import imdb

(x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz",
                                                      num_words=None,
                                                      skip_top=0,
                                                      maxlen=None,
                                                      seed=113,
                                                      start_char=1,
                                                      oov_char=2,
                                                      index_from=3)
```
- __반환값:__
    - 2개의 튜플:
        - __x_train, x_test__: 인덱스(정수)의 리스트인 시퀀스로 이루어진 리스트. 만약 num_words 인수를 특정지으면, 인덱스의 최대값은 num_words-1 입니다. 만약 maxlen 인수를 특정지으면, 시퀀스 길이의 최대값은 maxlen입니다.
        - __y_train, y_test__: 정수 라벨(1 or 0)로 이루어진 리스트. 

- __인수:__

    - __path__: (`'~/.keras/datasets/' + path`)의 위치에 데이터가 없다면, 이 위치로 데이터가 다운로드됩니다.
    - __num_words__: 정수 혹은 None. 고려할 가장 빈번한 단어. 그보다 드물게 사용된 단어는 시퀸스 데이터에 `oov_char` 값으로 나타납니다.
    - __skip_top__: 정수. 고려하지 않을 가장 빈번한 단어. 이러한 단어는 시퀀스 데이터에 `oov_char` 값으로 나타납니다.
    - __maxlen__: 정수. 시퀀스 길의의 최대값. 더 긴 시퀀스는 잘라냅니다.
    - __seed__: 정수. 재현 가능한 데이터 셔플링을 위한 시드입니다.
    - __start_char__: 정수. 시퀀스의 첫 시작이 이 문자로 마킹됩니다.
        0은 통상 패딩 문자이므로 1으로 조정하십시오.
    - __oov_char__: 정수. `num_words` 혹은 `skip_top`으로 인하여 제외된 단어는 이 문자로 대체됩니다.
    - __index_from__: 정수. 단어를 이 인덱스 이상의 수로 인덱스화 시킵니다.


---

## 로이터 뉴스 토픽 분류

46가지 토픽으로 라벨이 달린 11,228개의 로이터 뉴스로 이루어진 데이터셋. IMDB 데이터셋과 마찬가지로, 각 뉴스는 (같은 방식을 사용한) 단어 인덱스의 시퀀스로 인코딩되어 있습니다.

### 사용법:

```python
from keras.datasets import reuters

(x_train, y_train), (x_test, y_test) = reuters.load_data(path="reuters.npz",
                                                         num_words=None,
                                                         skip_top=0,
                                                         maxlen=None,
                                                         test_split=0.2,
                                                         seed=113,
                                                         start_char=1,
                                                         oov_char=2,
                                                         index_from=3)
```

세부사항은 IMDB 데이터셋과 동일하나, 다음의 추가사항이 있습니다:

- __test_split__: float. 테스트 데이터로 사용할 데이터셋의 비율.

또한 이 데이터셋은 시퀀스를 인코딩하는데 사용할 단어 인덱스를 제공합니다:

```python
word_index = reuters.get_word_index(path="reuters_word_index.json")
```

- __반환값:__ 키가 단어(str)이고 값이 인덱스(integer)인 하나의 딕셔너리. 예시. `word_index["giraffe"]`는 `1234`라는 값을 반환할 수 있습니다. 

- __인수:__

    - __path__: (`'~/.keras/datasets/' + path`)의 위치에 인덱스 파일이 없다면, 이 위치로 다운로드 됩니다.
    

---

## 손으로 쓴 숫자들로 이루어진 MNIST 데이터베이스

10가지 숫자에 대한 60,000개의 28x28 그레이 스케일 이미지 데이터셋과, 그에 더해 10,000개의 이미지로 이루어진 테스트셋.

### 사용법:

```python
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

- __반환값:__
    - 2개의 튜플:
        - __x_train, x_test__: 그레이 스케일 이미지 데이터의 uint8 배열. (num_samples, 28, 28)의 형태를 취합니다.
        - __y_train, y_test__: 숫자 라벨의 uint8 배열 (0-9 범위의 정수). (num_samples,)의 형태를 취합니다.

- __인수:__

    - __path__: ('~/.keras/datasets/' + path`)의 위치에 인덱스 파일이 없다면, 이 위치로 다운로드 됩니다.


---

## 패션 이미지로 이루어진 패션-MNIST 데이터베이스

10가지 패션 범주에 대한 60,000개의 28x28 그레일 스케일 이미지로 이루어진 데이터셋과, 그에 더해 10,000개의 이미지로 이루어진 테스트셋. 이 데이터 셋은 MNIST를 간편하게 대체하는 용도로 사용할 수 있습니다. 클래스 라벨은 다음과 같습니다:

| 라벨 | 설명 |
| --- | --- |
| 0 | 티셔츠/상의 |
| 1 | 바지 |
| 2 | 점퍼 |
| 3 | 드레스 |
| 4 | 코트 |
| 5 | 샌들 |
| 6 | 셔츠 |
| 7 | 운동화 |
| 8 | 가방 |
| 9 | 앵클 부츠 |

### 사용법:

```python
from keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
```

- __반환값:__
    - 2개의 튜플:
        - __x_train, x_test__: 그레이 스케일 이미지 데이터의 uint8 배열. (num_samples, 28, 28)의 형태를 취합니다.
        - __y_train, y_test__: 숫자 라벨의 uint8 배열 (0-9 범위의 정수). (num_samples,)의 형태를 취합니다.


---

## 보스턴 주택 가격 회귀 데이터셋


카네기 멜론 대학이 관리하는 StatLib 도서관의 데이터셋. 

각 샘플은 1970년대 보스턴 근교 여러지역에 위치한 주택의 13가지 속성으로 이루어져 있습니다.
타겟은 한 지역의 주택들의 (1,000$ 단위) 중앙값입니다.


### 사용법:

```python
from keras.datasets import boston_housing

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
```

- __인수:__
    - __path__: 데이터셋을 로컬로 캐싱할 경로
        (~/.keras/datasets를 기준으로).
    - __seed__: 테스트 데이터를 분할하기 전 데이터 셔플링을 위한 시드.
    - __test_split__: 테스트셋으로 남겨둘 데이터의 비율.

- __반환값:__
    Numpy 배열들로 이루어진 튜플: `(x_train, y_train), (x_test, y_test)`.

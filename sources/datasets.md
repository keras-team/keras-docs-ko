# Datasets

## CIFAR10 small image classification

Dataset of 50,000 32x32 color training images, labeled over 10 categories, and 10,000 test images.

### Usage:

```python
from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
```

- __Returns:__
    - 2 tuples:
        - __x_train, x_test__: uint8 array of RGB image data with shape (num_samples, 3, 32, 32) or (num_samples, 32, 32, 3) based on the `image_data_format` backend setting of either `channels_first` or `channels_last` respectively.
        - __y_train, y_test__: uint8 array of category labels (integers in range 0-9) with shape (num_samples,).


---

## CIFAR100 small image classification

Dataset of 50,000 32x32 color training images, labeled over 100 categories, and 10,000 test images.

### Usage:

```python
from keras.datasets import cifar100

(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
```

- __Returns:__
    - 2 tuples:
        - __x_train, x_test__: uint8 array of RGB image data with shape (num_samples, 3, 32, 32) or (num_samples, 32, 32, 3) based on the `image_data_format` backend setting of either `channels_first` or `channels_last` respectively.
        - __y_train, y_test__: uint8 array of category labels with shape (num_samples,).

- __Arguments:__

    - __label_mode__: "fine" or "coarse".


---

## IMDB Movie reviews sentiment classification

Dataset of 25,000 movies reviews from IMDB, labeled by sentiment (positive/negative). Reviews have been preprocessed, and each review is encoded as a [sequence](preprocessing/sequence.md) of word indexes (integers). For convenience, words are indexed by overall frequency in the dataset, so that for instance the integer "3" encodes the 3rd most frequent word in the data. This allows for quick filtering operations such as: "only consider the top 10,000 most common words, but eliminate the top 20 most common words".

As a convention, "0" does not stand for a specific word, but instead is used to encode any unknown word.

### Usage:

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
- __Returns:__
    - 2 tuples:
        - __x_train, x_test__: list of sequences, which are lists of indexes (integers). If the num_words argument was specific, the maximum possible index value is num_words-1. If the maxlen argument was specified, the largest possible sequence length is maxlen.
        - __y_train, y_test__: list of integer labels (1 or 0). 

- __Arguments:__

    - __path__: if you do not have the data locally (at `'~/.keras/datasets/' + path`), it will be downloaded to this location.
    - __num_words__: integer or None. Top most frequent words to consider. Any less frequent word will appear as `oov_char` value in the sequence data.
    - __skip_top__: integer. Top most frequent words to ignore (they will appear as `oov_char` value in the sequence data).
    - __maxlen__: int. Maximum sequence length. Any longer sequence will be truncated.
    - __seed__: int. Seed for reproducible data shuffling.
    - __start_char__: int. The start of a sequence will be marked with this character.
        Set to 1 because 0 is usually the padding character.
    - __oov_char__: int. words that were cut out because of the `num_words`
        or `skip_top` limit will be replaced with this character.
    - __index_from__: int. Index actual words with this index and higher.


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

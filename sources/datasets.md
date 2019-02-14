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

## Reuters newswire topics classification

Dataset of 11,228 newswires from Reuters, labeled over 46 topics. As with the IMDB dataset, each wire is encoded as a sequence of word indexes (same conventions).

### Usage:

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

The specifications are the same as that of the IMDB dataset, with the addition of:

- __test_split__: float. Fraction of the dataset to be used as test data.

This dataset also makes available the word index used for encoding the sequences:

```python
word_index = reuters.get_word_index(path="reuters_word_index.json")
```

- __Returns:__ A dictionary where key are words (str) and values are indexes (integer). eg. `word_index["giraffe"]` might return `1234`. 

- __Arguments:__

    - __path__: if you do not have the index file locally (at `'~/.keras/datasets/' + path`), it will be downloaded to this location.
    

---

## MNIST database of handwritten digits

Dataset of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images.

### Usage:

```python
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

- __Returns:__
    - 2 tuples:
        - __x_train, x_test__: uint8 array of grayscale image data with shape (num_samples, 28, 28).
        - __y_train, y_test__: uint8 array of digit labels (integers in range 0-9) with shape (num_samples,).

- __Arguments:__

    - __path__: if you do not have the index file locally (at `'~/.keras/datasets/' + path`), it will be downloaded to this location.


---

## Fashion-MNIST database of fashion articles

Dataset of 60,000 28x28 grayscale images of 10 fashion categories, along with a test set of 10,000 images. This dataset can be used as a drop-in replacement for MNIST. The class labels are:

| Label | Description |
| --- | --- |
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

### Usage:

```python
from keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
```

- __Returns:__
    - 2 tuples:
        - __x_train, x_test__: uint8 array of grayscale image data with shape (num_samples, 28, 28).
        - __y_train, y_test__: uint8 array of labels (integers in range 0-9) with shape (num_samples,).


---

## Boston housing price regression dataset


Dataset taken from the StatLib library which is maintained at Carnegie Mellon University. 

Samples contain 13 attributes of houses at different locations around the Boston suburbs in the late 1970s.
Targets are the median values of the houses at a location (in k$).


### Usage:

```python
from keras.datasets import boston_housing

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
```

- __Arguments:__
    - __path__: path where to cache the dataset locally
        (relative to ~/.keras/datasets).
    - __seed__: Random seed for shuffling the data
        before computing the test split.
    - __test_split__: fraction of the data to reserve as test set.

- __Returns:__
    Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

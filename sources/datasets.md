# 데이터 세트

## CIFAR10 소형 이미지 분류

50,000개의 32x32 컬러 학습 이미지, 10개 범주의 레이블, 10,000개의 테스트 이미지로 구성된 데이터 세트.

### 사용법:

```python
from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
```

- __반환값:__
    - 2개의 튜플:
        - __x_train, x_test__: RGB 이미지 데이터의 `uint8` 배열. 케라스 백엔드의 `image_data_format` 설정이 `channels_first`인지 `channels_last`인지에 따라 각각 `(num_samples, 3, 32, 32)` 혹은 `(num_samples, 32, 32, 3)`의 형태를 취합니다.
        - __y_train, y_test__: 범주형 레이블의 `uint8` 배열<sub>array</sub>로 0-9 범위의 정수값을 갖습니다. (num_samples, )의 형태를 취합니다. 


---

## CIFAR100 소형 이미지 분류:

50,000개의 32x32 컬러 학습 이미지, 100개 범주의 레이블, 10,000개의 테스트 이미지로 구성된 데이터 세트.

### 사용법:

```python
from keras.datasets import cifar100

(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
```

- __반환값:__
    - 2개의 튜플:
        - __x_train, x_test__: RGB 이미지 데이터의 `uint8` 배열. 케라스 백엔드의 `image_data_format` 설정이 `channels_first`인지 `channels_last`인지에 따라 각각 `(num_samples, 3, 32, 32)` 혹은 `(num_samples, 32, 32, 3)`의 형태를 취합니다.
        - __y_train, y_test__: 범주형 레이블의 `uint8` 배열<sub>array</sub>로 0-9 범위의 정수값을 갖습니다. (num_samples, )의 형태를 취합니다.

- __인자:__

    - __label_mode__: `'fine'` 혹은 `'coarse'`. 기본값은 `'fine'`입니다.


---

## IMDB 영화 리뷰 감정 분류:

감정에 따라 긍정적/부정적으로 분류한 25,000개의 IMDB 영화 리뷰로 구성된 데이터 세트. 각 리뷰는 전처리 과정을 통해 단어 인덱스(정수)로 구성된 [순서형](preprocessing/sequence.md)<sub>sequence</sub> 데이터로 인코딩 되었습니다. 편의를 위해 단어는 전체 데이터 내의 등장빈도에 따라 인덱스화 되었습니다. 예를 들어, 정수 "3"은 데이터 내에서 세 번째로 빈번하게 사용된 단어를 나타냅니다. 이는 "가장 빈번하게 사용된 10,000 단어만을 고려하되 가장 많이 쓰인 20 단어는 제외"와 같은 빠른 필터링 작업을 가능케 합니다.

관습에 따라 `0`은 특정 단어에 할당되지 않는 인덱스로 목록에 없는 미확인 단어를 통칭하는 데 사용됩니다.

### 사용법:

```python
from keras.datasets import imdb

(x_train, y_train), (x_test, y_test) = imdb.load_data(path='imdb.npz',
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
        - __x_train, x_test__: 글에 등장하는 단어 순서에 따라 정수 인덱스가 나열된 리스트들로 이루어진 리스트. `num_words`인자를 지정하면 사용 가능한 인덱스의 최댓값은 `num_words-1`이 됩니다. `maxlen`인자를 지정하면 개별 리스트 길이의 최댓값은 maxlen이 됩니다.
        - __y_train, y_test__: `1` 또는 `0`의 정수 레이블로 이루어진 리스트. 

- __인자:__

    - __path__: (`'~/.keras/datasets/' + path`)의 위치에 데이터가 없다면, 이 위치로 데이터가 다운로드됩니다.
    - __num_words__: `int` 혹은 `None`. 등장 횟수가 큰 순서대로 상위 몇 번째까지의 단어를 인덱스로 사용할지 결정합니다. 범위 바깥의 단어는 데이터에서 `oov_char`값으로 나타납니다. 기본값은 `None`입니다.
    - __skip_top__: `int`. 등장 횟수가 큰 순서대로 상위 몇 번째까지의 단어를 무시할지 결정합니다. 지정된 단어는 데이터에 `oov_char` 값으로 나타납니다. 기본값은 `0`입니다.
    - __maxlen__: `int`. 개별 텍스트를 나타내는 순서형 인덱스 리스트의 최대 길이. 초과하는 부분은 잘라냅니다. 기본값은 `None`입니다.
    - __seed__: `int`. 데이터 뒤섞기 과정을 재현 가능하게 하기 위한 시드입니다.
    - __start_char__: `int`. 텍스트의 시작 지점을 나타내기 위한 인덱스를 지정합니다. 대개 `0`은 패딩에 사용되는 인덱스이므로 `1`을 사용합니다.
    - __oov_char__: `int`. `num_words` 혹은 `skip_top`으로 인하여 제외된 단어는 여기서 지정한 인덱스로 대체됩니다. 기본값은 `2`입니다.
    - __index_from__: `int`. 단어에 부여할 인덱스의 최솟값을 지정합니다. 기본값은 `3`입니다.


---

## 로이터 뉴스 토픽 분류

11,228개의 로이터 뉴스를 46가지 주제에 따라 분류한 데이터 세트. IMDB 데이터 세트와 마찬가지로 동일한 전처리 과정을 거친 단어 인덱스의 순서형 데이터로 인코딩되어 있습니다.

### 사용법:

```python
from keras.datasets import reuters

(x_train, y_train), (x_test, y_test) = reuters.load_data(path='reuters.npz',
                                                         num_words=None,
                                                         skip_top=0,
                                                         maxlen=None,
                                                         test_split=0.2,
                                                         seed=113,
                                                         start_char=1,
                                                         oov_char=2,
                                                         index_from=3)
```

인자 및 세부사항은 IMDB 데이터 세트와 동일하나, 다음과 같은 추가사항이 있습니다.

- __test_split__: `float`. 전체 데이터 세트 가운데 테스트용으로 분리할 데이터의 비율. 기본값은 `0.2`입니다.

또한 뉴스 데이터를 인코딩하는데 사용된 단어 인덱스도 함께 제공됩니다.

```python
word_index = reuters.get_word_index(path='reuters_word_index.json')
```

- __반환값:__ 키가 단어(`str`)이고 값이 인덱스(`int`)인 하나의 딕셔너리. (예: `word_index['giraffe']`는 `1234`라는 값을 반환할 수 있습니다.) 

- __인자:__

    - __path__: (`'~/.keras/datasets/' + path`)의 위치에 인덱스 파일이 없다면, 이 위치로 다운로드 됩니다.
    

---

## 손으로 쓴 숫자들로 이루어진 MNIST 데이터베이스

10가지 숫자에 대한 60,000개의 28x28 그레이 스케일 이미지 데이터 세트와 10,000개의 이미지로 이루어진 테스트 세트.

### 사용법:

```python
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

- __반환값:__
    - 2개의 튜플:
        - __x_train, x_test__: 그레이 스케일 이미지 데이터의 `uint8` 배열. `(num_samples, 28, 28)`의 형태를 취합니다.
        - __y_train, y_test__: 숫자 레이블의 `uint8` 배열로 0-9 범위의 정수값을 갖습니다. `(num_samples, )`의 형태를 취합니다.

- __인자:__

    - __path__: (`'~/.keras/datasets/' + path`)의 위치에 인덱스 파일이 없다면, 이 위치로 다운로드 됩니다.


---

## 패션 이미지로 이루어진 패션-MNIST 데이터베이스

10가지 패션 범주에 대한 60,000개의 28x28 그레일 스케일 이미지로 이루어진 데이터 세트와 10,000개의 이미지로 이루어진 테스트 세트. 이 데이터 세트는 MNIST를 간편하게 대체하는 용도로 사용할 수 있습니다. 클래스 레이블은 다음과 같습니다:

| 레이블 | 설명 |
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
        - __x_train, x_test__: 그레이 스케일 이미지 데이터의 `uint8` 배열. `(num_samples, 28, 28)`의 형태를 취합니다.
        - __y_train, y_test__: 숫자 레이블의 `uint8` 배열로 0-9 범위의 정수값을 갖습니다. `(num_samples, )`의 형태를 취합니다.


---

## 보스턴 주택 가격 회귀 데이터 세트


카네기 멜론 대학이 관리하는 StatLib 도서관의 데이터 세트.  
  
각 샘플은 1970년대 보스턴 근교 여러지역에 위치한 주택의 13가지 속성값으로 이루어져 있습니다. 
타겟은 한 지역의 주택들의 (1,000$ 단위) 중앙값<sub>median</sub>입니다.


### 사용법:

```python
from keras.datasets import boston_housing

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
```

- __인자:__
    - __path__: 데이터 세트를 다운로드할 경로. (`'~/.keras/datasets/' + path`)의 형태로 지정됩니다.
    - __seed__: 데이터 뒤섞기 과정을 재현 가능하게 하기 위한 시드입니다. 뒤섞기 과정은 데이터 세트를 분할하기 전에 이루어집니다.
    - __test_split__: 테스트용으로 분리할 데이터의 비율.

- __반환값:__
    NumPy 배열들로 이루어진 튜플: `(x_train, y_train), (x_test, y_test)`.

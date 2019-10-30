# 텍스트 전처리 모듈<sub>Text Preprocessing Modules</sub> 
텍스트 전처리 도구 모듈입니다.  

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/preprocessing/text.py#L138)</span>
## Tokenizer 클래스

### Tokenizer
```python
keras.preprocessing.text.Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~	
', lower=True, split=' ', char_level=False, oov_token=None, document_count=0)
```
텍스트 토큰화 도구 클래스입니다.

`Tokenizer` 클래스는 텍스트 말뭉치<sub>corpus</sub>로부터 단어를 추출하고 정수 인덱스를 부여하여 딕셔너리 형식의 목록을 생성합니다. 이 목록을 바탕으로 문장 입력을 각 단어의 인덱스 숫자로 이루어진 리스트로 변환하며, 반대로 인덱스 리스트를 입력할 경우 문장으로 변환합니다. 또한 문장별 단어의 등장 여부나 횟수, 비율, TF-IDF 등을 나타내는 행렬을 생성할 수 있습니다. 이와 같은 처리는 `Tokenizer`의 하위 메소드들에 의해 이루어지며, `Tokenizer` 클래스는 말뭉치의 토큰화에 필요한 각종 설정값을 지정합니다. 

__인자__

- __num_words__: `int`. 사용할 단어 개수의 최대값. 가장 빈번하게 사용되는 `num_words`개의 단어만 보존합니다. `0`은 어떤 단어에도 배정되지 않는 예비 인덱스 값이기 때문에, 입력된 말뭉치 가운데 실제로 보존되는 단어의 개수는 `num_words-1`개가 됩니다.
- __filters__: `str`. 입력된 텍스트로부터 제외할 문자를 지정합니다. 기본값은 작은따옴표 `'`를 제외한 모든 문장부호 및 탭과 줄바꿈 문자입니다.
- __lower__: `bool`. 텍스트를 소문자로 변환할지의 여부를 지정합니다. 기본값은 `True`입니다.
- __split__: `str`. 문장 내 각 단어를 분리하는 단위 문자를 지정합니다. 기본값은 `' '`공백문자입니다.
- __char_level__: `bool`. 참인 경우 단어 대신 `a, z, 0, 9`와 같은 각각의 문자를 별개의 토큰으로 처리합니다.
- __oov_token__: `str`. 생성한 단어 인덱스 `word_index`에 없는 단어를 대체하기 위한 문자열을 지정합니다(예: `<UNK>`). 기본값은 `None`이며, 지정된 경우 목록 생성시 단어 인덱스에 추가되어 이후 `text_to_sequence` 처리를 하는 경우에 등록되지 않은 단어를 대체합니다.  
  
기본 설정을 따를 경우, 입력된 텍스트에서 `'`를 제외한 모든 문장부호를 삭제하여 단어의 나열로 만든 다음 공백으로 분리하여 토큰의 리스트를 만듭니다. 이 토큰으로 인덱스를 생성하거나 필요한 행렬을 만듭니다.  
  
다시 강조하지만, `0`은 어떤 단어에도 배정되지 않는 예비 인덱스입니다.


## Tokenizer 메소드
### fit_on_texts
```python
fit_on_texts(texts)
```
입력된 텍스트를 바탕으로 단어를 추출, 정수 인덱스를 부여하여 `Tokenizer` 내부에 딕셔너리 형태의 단어 목록을 생성합니다. 이 목록은 `word_index` 메소드로 불러올 수 있습니다. 목록은 전체 단어의 종류 만큼 생성되지만 이 가운데 다른 메소드에서 사용되는 단어 개수는 최대 `Tokenizer`에서 지정한 `num_words-1`개입니다(`num_words-1`개인 까닭은 `0`번 인덱스는 단어에 배정되지 않기 때문입니다). 기본적으로 문자열로 이루어진 리스트를 텍스트로 입력받으며, 텍스트 리스트가 하위 리스트들로 구성된 경우<sub>list of lists</sub> 각 하위 리스트들에 포함된 문자열들을 각각 하나의 토큰으로 취급합니다. 이후 `texts_to_sequences` 메소드와 `texts_to_matrix` 메소드를 사용하기 위해서는 먼저 `fit_on_texts` 메소드로 단어 목록을 생성해야 합니다.

__인자__
- __texts__: 문자열의 리스트, (메모리 절약을 위한)문자열의 제너레이터, 또는 문자열 리스트로 이루어진 리스트.

---
### fit_on_sequences
```python
fit_on_sequences(sequences)
```
단어 인덱스로 이루어진 리스트를 입력받아 `Tokenizer`내부에 인덱스 목록을 생성합니다. `fit_on_text`를 따로 사용하지 않는 경우, 이후 `sequences_to_matrix` 메소드를 사용하려면 먼저 `fit_on_sequences` 메소드를 실행해야 합니다.

__인자__
- __sequences__: 순서형 데이터의 리스트. 여기서 리스트 내의 각 순서형 데이터는 해당 문장의 단어 순서에 따라 각 단어의 인덱스를 나열한 리스트입니다.

---
### texts_to_sequences
```python
texts_to_sequences(texts)
```
입력된 문장을 각 단어의 인덱스로 이루어진 순서형 데이터로 변환합니다. 변환에는 `fit_on_texts` 메소드를 통해 `Tokenizer`에 입력된 단어만이 사용되며, 단어의 종류가 `Tokenizer`에 지정된 `num_words-1`개를 초과할 경우 등장 횟수가 큰 순서대로 상위 `num_words-1`개의 단어를 사용합니다. 

__인자__
- __texts__: 문자열의 리스트, (메모리 절약을 위한)문자열의 제너레이터, 또는 문자열 리스트로 이루어진 리스트.

__반환값__  

단어 인덱스로 이루어진 리스트.

---
### texts_to_sequences_generator
```python
texts_to_sequences_generator(texts)
```
입력된 문장을 각 단어의 인덱스의 리스트로 변환하여 순차적으로 반환하는 제너레이터를 생성합니다. 기본적으로 문자열로 이루어진 리스트를 텍스트로 입력받으며, 텍스트 리스트가 하위 리스트들로 구성된 경우 각 하위 리스트들에 포함된 문자열들을 각각 하나의 토큰으로 취급합니다. 변환에는 `fit_on_texts`메소드를 통해 `Tokenizer`에 입력된 단어만이 사용되며, 단어의 종류가 `Tokenizer`에 지정된 `num_words-1`개를 초과할 경우 등장 횟수가 큰 순서대로 상위 `num_words-1`개의 단어를 사용합니다.

__인자__
- __texts__: 문자열의 리스트, (메모리 절약을 위한)문자열의 제너레이터, 또는 문자열 리스트로 이루어진 리스트.

__반환값__  

단어 인덱스로 이루어진 리스트를 반환하는 제너레이터.

---
### sequences_to_texts
```python
sequences_to_texts(sequences)
```
단어 인덱스로 이루어진 리스트를 텍스트(문장)로 변환합니다. 변환에는 `fit_on_texts` 메소드를 통해 `Tokenizer`에 입력된 단어만이 사용되며, 단어의 종류가 `Tokenizer`에 지정된 `num_words-1`개를 초과할 경우 등장 횟수가 큰 순서대로 상위 `num_words-1`개의 단어를 사용합니다. 

__인자__
- __sequences__: 단어 인덱스로 이루어진 리스트들의 리스트, 또는 단어 인덱스로 이루어진 리스트를 생성하는 제너레이터.

__반환값__  

문자열의 리스트.

---
### sequences_to_texts_generator
```python
sequences_to_texts_generator(sequences)
```
단어 인덱스로 이루어진 리스트를 텍스트(문장)로 변환하여 순차적으로 반환하는 제너레이터를 생성합니다. 변환에는 `fit_on_texts` 메소드를 통해 `Tokenizer`에 입력된 단어만이 사용되며, 단어의 종류가 `Tokenizer`에 지정된 `num_words-1`개를 초과할 경우 등장 횟수가 큰 순서대로 상위 `num_words-1`개의 단어를 사용합니다. 

__인자__
- __sequences__: 단어 인덱스로 이루어진 리스트들의 리스트, 또는 단어 인덱스로 이루어진 리스트를 생성하는 제너레이터.

__반환값__  

문자열의 리스트를 반환하는 제너레이터.
            
---            
### texts_to_matrix
```python
texts_to_matrix(texts, mode)
```
입력된 문장을 NumPy 행렬로 변환합니다. 각 행렬은 (문장의 개수 × `num_words`)의 형태를 가지며 n번째 열은 n의 인덱스를 갖는 단어를 나타냅니다. 변환된 행렬의 값은 `mode`인자에 따라 다음과 같이 달라집니다.
- `'binary'`: 문장별로 존재하는 단어는 1, 아닌 단어는 0의 값을 갖는 행렬.
- `'count'`: 각 단어가 문장 내에서 등장하는 횟수만큼의 값을 갖는 행렬.
- `'tfidf'`: 단어별로 Term Frequency-Inverse Document Frequency 값을 갖는 행렬.
- `'freq'`: 해당 문장의 전체 단어 개수 가운데 각 단어의 등장 횟수 비율을 단어별 값으로 갖는 행렬.

__인자__
- __texts__: 문자열의 리스트, (메모리 절약을 위한) 문자열의 제너레이터, 또는 문자열 리스트로 이루어진 리스트.
- __mode__: `'binary'`, `'count'`, `'tfidf'`, `'freq'`.

__반환값__  

NumPy 행렬.

---
### sequences_to_matrix
```python
sequences_to_matrix(sequences, mode)
```
단어 인덱스로 이루어진 리스트를 NumPy 행렬로 변환합니다. 각 행렬은 (문장의 개수 × `num_words`)의 형태를 가지며 n번째 열은 n의 인덱스를 갖는 단어를 나타냅니다. 변환된 행렬의 값은 `mode`인자에 따라 다음과 같이 달라집니다.
- `'binary'`: 문장별로 존재하는 단어는 1, 아닌 단어는 0의 값을 갖는 행렬.
- `'count'`: 각 단어가 문장 내에서 등장하는 횟수만큼의 값을 갖는 행렬.
- `'tfidf'`: 단어별로 Term Frequency-Inverse Document Frequency 값을 갖는 행렬.
- `'freq'`: 해당 문장의 전체 단어 개수 가운데 각 단어의 등장 횟수 비율을 단어별 값으로 갖는 행렬.

__인자__ 
- __sequences__: 단어 인덱스로 이루어진 리스트들의 리스트.
- __mode__: `'binary'`, `'count'`, `'tfidf'`, `'freq'`.

__반환값__  

NumPy 행렬.

---
### get_config
```python
get_config()
```
`Tokenizer`의 설정값을 파이썬 딕셔너리 형식으로 반환합니다. 

__반환값__  

`Tokenizer` 설정값 딕셔너리.

---
### to_json
```python
to_json(**kwargs)
```
`Tokenizer` 설정값 딕셔너리를 JSON 형식 문자열로 반환합니다. 딕셔너리의 내용은 JSON 형식에 맞게 한 줄로 정렬됩니다. 출력할 JSON 문자열의 특성은 파이썬의 `json.dumps()`함수에 사용되는 인자를 입력함으로써 조정할 수 있습니다. 저장된 JSON 문자열로부터 `Tokenizer` 설정을 불러오기 위해서는 `keras.preprocessing.text.tokenizer_from_json(json_string)` 메소드를 사용합니다. 

__인자__
- **kwargs: `json.dumps()`로 전달되는 인자들.

__반환값__  

`Tokenizer` 설정값 JSON 문자열.

----
<nbsp></nbsp>
<nbsp></nbsp>

## 전처리 함수들
### hashing_trick
```python
keras.preprocessing.text.hashing_trick(text, n, hash_function=None, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~	
', lower=True, split=' ')
```
텍스트에 일반적인 정수 대신 지정한 크기의 해시공간에 맞추어 인덱스를 부여합니다. 이때 해시 함수의 충돌로 인해 서로 다른 단어가 같은 인덱스 값을 갖는 경우가 생길 수 있습니다. 충돌 [확률](https://en.wikipedia.org/wiki/Birthday_problem#Probability_table)은 해시 공간의 차원과 개별 객체의 개수에 따라 달라집니다. `0`은 어떤 단어에도 배정되지 않는 예비 인덱스입니다.

__인자__
- __text__: `str`. 텍스트 입력.
- __n__: `int`. 해시 공간의 차원. 해싱 함수를 통해 부여하는 인덱스의 최댓값을 지정합니다. 전체 어휘 목록의 크기보다 작을 경우 인덱스 중복이 발생합니다.
- __hash_function__: 디폴트 값은 파이썬 `hash` 함수로 MD5 함수를 사용하려면 `'md5'`를 입력합니다. 그밖에도 문자열을 입력받고 정수를 반환하는 모든 함수를 사용할 수 있습니다. 참고로 파이썬의 `hash`는 안정적인 해시 함수가 아니어서 매 작동마다 일관성을 유지하지 못하는 반면, MD5는 안정적인 해시 함수라는 특징이 있습니다.
- __filters__: `str`. 입력된 텍스트로부터 제외할 문자를 지정합니다. 기본값은 작은따옴표 `'`를 제외한 모든 문장부호 및 탭과 줄바꿈 문자입니다.
- __lower__: `bool`. 텍스트를 소문자로 변환할지의 여부를 지정합니다. 기본값은 `True`입니다.
- __split__: `str`. 문장 내 각 단어를 분리하는 단위 문자를 지정합니다. 기본값은 `' '`공백문자입니다.

__반환값__  

정수 인덱스 리스트(단일성이 보장되지 않습니다).

----

### one_hot
```python
keras.preprocessing.text.one_hot(text, n, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~	
', lower=True, split=' ')
```
텍스트를 `n`크기 해쉬공간의 인덱스 리스트로 만든 뒤 원-핫 인코딩합니다. `one_hot`은 파이썬의 `hash`를 해시 함수로 사용하는 `hashing_trick` 함수의 래퍼입니다. 따라서 단어와 인덱스 사이의 단일성은 보장되지 않습니다.

__인자__
- __text__: `str`. 텍스트 입력.
- __n__: `int`. 해시 공간의 차원. 실제로는 어휘 목록의 크기로 기능합니다.
- __filters__: `str`. 입력된 텍스트로부터 제외할 문자를 지정합니다. 기본값은 작은따옴표 `'`를 제외한 모든 문장부호 및 탭과 줄바꿈 문자입니다.
- __lower__: `bool`. 텍스트를 소문자로 변환할지의 여부를 지정합니다. 기본값은 `True`입니다.
- __split__: `str`. 문장 내 각 단어를 분리하는 단위 문자를 지정합니다. 기본값은 `' '`공백문자입니다.

__반환값__  

정수 인덱스 리스트 (단일성이 보장되지 않습니다).
        
----

### text_to_word_sequence
```python
keras.preprocessing.text.text_to_word_sequence(text, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~	
', lower=True, split=' ')
```
텍스트를 단어(혹은 토큰)의 리스트로 변환합니다.

__인자__
- __text__: `str`. 텍스트 입력.
- __filters__: `str`. 입력된 텍스트로부터 제외할 문자를 지정합니다. 기본값은 작은따옴표 `'`를 제외한 모든 문장부호 및 탭과 줄바꿈 문자입니다.
- __lower__: `bool`. 텍스트를 소문자로 변환할지의 여부를 지정합니다. 기본값은 `True`입니다.
- __split__: `str`. 문장 내 각 단어를 분리하는 단위 문자를 지정합니다. 기본값은 `' '`공백문자입니다.

__반환값__  

단어(혹은 토큰)의 리스트.

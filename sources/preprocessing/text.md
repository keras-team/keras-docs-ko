
### Text Preprocessing

<span style="float:right;">[[소스 코드]](https://github.com/keras-team/keras/blob/master/keras/preprocessing/text.py#L138)</span>

### Tokenizer

```python
keras.preprocessing.text.Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~	
', lower=True, split=' ', char_level=False, oov_token=None, document_count=0)
```

텍스트 토큰화 유틸리티 클래스(Text tokenization utility class).

텍스트 말뭉치를 정수 벡터로 변환하는 클래스입니다. 각각의 말뭉치를 하나의 정수로 표현한 숫자 사전으로 변환하거나 
말뭉치 빈도 수에 따라 각 토큰의 계수가 변하는 tf-idf 기반의 정수 벡터로 변환합니다.

__Arguments__

- __num_words__: 보관할 최대 단어 수입니다.  `num_words`만큼 가장 많이 등장하는 순서대로 단어를 보관합니다.
- __filters__: 단어 안에서 제외할 문자들로 이루어진 문자열입니다. 기본 값은 모든 문장 부호와 탭, 줄바꿈에서  `'`문자를 
  뺀 값입니다.
- __lower__: 대소문자 변환 여부입니다(boolean).
- __split__: 단어를 분리하기 위한 구분자입니다(str).
- __char_level__: 토큰 처리 여부입니다. True이면 매 문자마다 토큰으로 변환합니다. (boolean)
- __oov_token__: 값이 주어지면,  word_index에 추가되고 text_to_sequence 호출동안 잘못된 단어를 대체하는 데 
  사용됩니다.

기본적으로 문장 부호가 모두 지워진 다음 공백으로 구분된 텍스트로 변환됩니다. ( `'` 문자가 포함될 수 있음)
이어서 다시 토큰 리스트로 변환된 다음 인덱싱되거나 벡터화 될 것입니다.

`0`은 어떤 단어에도 할당되지않는 고정 인덱스입니다.

----

### hashing_trick


```python
keras.preprocessing.text.hashing_trick(text, n, hash_function=None, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~	
', lower=True, split=' ')
```


텍스트를 고정 크기의 해시 공간의 인덱스로 매핑합니다.

__Arguments__

- __text__: 입력 텍스트입니다. (string)
- __n__: 해시 크기입니다.
- __hash_function__: 기본값은 파이썬 `hash` 함수입니다. 'md5'같이 다른 함수가 들어갈 수 있으며, 문자열을 입력받아 
    정수로 출력할 수 있습니다. 'hash'는 매 실행마다 다른 해시 값을 생성하기 때문에 같은 문자에 대해서 일관되지 
    않습니다.  
    그에 비해 'md5'는 일관되니 참고하십시오.
- __filters__: 필터링할 문자 리스트(또는 문자열)입니다. 기본 값은 ``!"#$%&()*+,-./:;<=>?@[]^_{|}~`` 와 
    기본 문장 부호와 탭, 줄바꿈입니다.

- __lower__: 소문자로 설정할 지 여부입니다. (boolean)
- __split__: 단어를 분리하기 위한 구분자입니다. (str)

__Returns__

단어와 매핑된 정수 리스트입니다. 단일성을 보장하지 않습니다.

해시 함수에 의한 충돌 때문에 두 개 이상의 단어가 하나의 인덱스에 할당될 수 있습니다.  
충돌 확률은 해시 공간의 크기와 객체의 수에 영향을 받습니다.

`0` 은 어떤 단어에도 할당되지않는 고정 인덱스입니다.

----

### one_hot


```python
keras.preprocessing.text.one_hot(text, n, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~	
', lower=True, split=' ')
```

텍스트를 n개의 단어 인덱스 리스트로 원-핫 인코딩합니다.

 `hash` 모듈을 사용하는 `hashing_trick` 함수의 래퍼(wrapper)입니다. 따라서 단어 인덱스 매핑이 
단일성을 보장하지 않습니다.

__Arguments__

- __text__: 입력 텍스트입니다. (string)
- __n__: 단어 사전의 크기입니다. (int)
- __filters__: 필터링할 문자 리스트(또는 문자열)입니다. 기본 값은 ``!"#$%&()*+,-./:;<=>?@[]^_{|}~`` 와 
    기본 문장 부호와 탭, 줄바꿈입니다.

- __lower__: 소문자로 설정할 지 여부입니다. (boolean)
- __split__: 단어를 분리하기 위한 구분자입니다. (str)

__Returns__

각각의 단어가 인코딩된 [1, n] 크기의 정수 리스트입니다. 단일성을 보장하지 않습니다.

----

### text_to_word_sequence


```python
keras.preprocessing.text.text_to_word_sequence(text, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~	
', lower=True, split=' ')
```


텍스트를 일련의 단어 또는 토큰으로 변환합니다.

__Arguments__

- __text__: 입력 텍스트입니다. (string)
- __filters__: 필터링할 문자 리스트(또는 문자열)입니다. 기본 값은 ``!"#$%&()*+,-./:;<=>?@[]^_{|}~`` 와 
    기본 문장 부호와 탭, 줄바꿈입니다.

- __lower__: 소문자로 설정할 지 여부입니다. (boolean)
- __split__: 단어를 분리하기 위한 구분자입니다. (str)

__Returns__

단어 또는 토큰 리스트입니다.


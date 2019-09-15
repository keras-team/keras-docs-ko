<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/utils/generic_utils.py#L21)</span>
### CustomObjectScope

```python
keras.utils.CustomObjectScope()
```

`_GLOBAL_CUSTOM_OBJECTS`로의 변환이 벗어나지 못하는 유효범위를 제공합니다.

`with` 명령문 내의 코드는 이름을 통해 커스텀 객체에 접근할 수 있습니다.
글로벌 커스텀 객체로의 변환은
`with` 명령문의 영역 내에서 유효합니다. `with` 명령문이 끝나면,
글로벌 커스텀 객체는 
`with` 명령문 시작에서의 상태로 되돌아갑니다.

__예시__


`MyObject`(예. 클래스)라는 커스텀 객체를 생각해 봅시다:

```python
with CustomObjectScope({'MyObject':MyObject}):
    layer = Dense(..., kernel_regularizer='MyObject')
    # save, load, 등의 함수가 이름을 통해 커스텀 객체를 인지합니다 
```

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/utils/io_utils.py#L25)</span>
### HDF5Matrix

```python
keras.utils.HDF5Matrix(datapath, dataset, start=0, end=None, normalizer=None)
```

Numpy 배열을 대체하는 HDF5 데이터셋 표현양식.

__예시__


```python
x_data = HDF5Matrix('input/file.hdf5', 'data')
model.predict(x_data)
```

`start`와 `end`를 제공해서 데이터셋을 조각으로 잘라 사용할 수 있도록 합니다.

선택적으로 정규화 함수(혹은 람다)를 사용할 수 있습니다.
이는 회수된 모든 데이터 조각에 대해 호출됩니다.

__인수__

- __datapath__: 문자열, HDF5 파일의 경로입니다
- __dataset__: 문자열, datapath에 명시된 파일 내 HDF5 데이터셋의
    이름입니다.
- __start__: 정수, 명시된 데이터셋의 원하는 조각의 시작부분입니다
- __end__: 정수, 명시된 데이터셋의 원하는 조각의 끝부분입니다
- __normalizer__: 데이터가 회수될 때 데이터에 대해서 호출할 함수

__반환값__

배열과 비슷한 형태의 HDF5 데이터셋
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/utils/data_utils.py#L305)</span>
### Sequence

```python
keras.utils.Sequence()
```

데이터셋 등의 데이터 시퀀스를 학습하기 위한 베이스 객체.

모든 `Sequence`는 `__getitem__`과 `__len__` 메서드를 실행해야 합니다.
`on_epoch_end`를 실행하여 세대와 세대 사이에 데이터셋을 수정할 수 있습니다.
`__getitem__` 메서드는 완전한 배치를 반환해야 합니다.

__안내_


`Sequence`는 멀티프로세싱을 보다 안전하게 실행합니다.
생성기와는 다르게 네트워크가 각 세대에 한 샘플을 한 번만 학습하도록
보장해줍니다.

__예시__


```python
from skimage.io import imread
from skimage.transform import resize
import numpy as np

# 여기서 `x_set`은 이미지 파일 경로의 리스트입니다
# 그리고 `y_set`은 연관 클래스입니다.

class CIFAR10Sequence(Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([
            resize(imread(file_name), (200, 200))
               for file_name in batch_x]), np.array(batch_y)
```
    
----

### to_categorical


```python
keras.utils.to_categorical(y, num_classes=None, dtype='float32')
```


클래스 벡터(정수)를 이진 클래스 행렬로 변환합니다.

예. categorical_crossentropy와 함께 사용할 수 있습니다.

__인수__

- __y__: 행렬로 변환할 클래스 벡터
    (0부터 num_classes까지의 정수).
- __num_classes__: 클래스의 총 개수.
- __dtype__: 문자열로 표현된 인풋의 데이터 자료형
    (`float32`, `float64`, `int32`...)

__반환값__

인풋의 이진행렬 표현.
클래스 축이 마지막에 위치합니다.

__예시__


```python
# 3 클래스 {0, 1, 2}에 대한 5개의 라벨로 이루어진 배열을 생각해봅시다:
> labels
array([0, 2, 1, 2, 0])
# `to_categorical`은 이를 클래스 수 만큼의 열을 가진
# 행렬로 변환합니다. 행의 수는 
# 변하지 않습니다.
> to_categorical(labels)
array([[ 1.,  0.,  0.],
       [ 0.,  0.,  1.],
       [ 0.,  1.,  0.],
       [ 0.,  0.,  1.],
       [ 1.,  0.,  0.]], dtype=float32)
```
    
----

### normalize


```python
keras.utils.normalize(x, axis=-1, order=2)
```


Numpy 배열을 정규화합니다.

__인수__

- __x__: 정규화할 Numpy 배열.
- __axis__: 정규화를 따라 진행할 축.
- __order__: 정규화 계수 (예. L2 노름의 경우 2).

__반환값__

배열을 정규화한 복사본.
    
----

### get_file


```python
keras.utils.get_file(fname, origin, untar=False, md5_hash=None, file_hash=None, cache_subdir='datasets', hash_algorithm='auto', extract=False, archive_format='auto', cache_dir=None)
```


캐시에 파일이 존재하지 않으면 URL에서 파일을 다운로드 합니다.

디폴트로 url `origin`에서 cache_subdir인 `datasets` 내 위치한
cache_dir인 `~/.keras`로 파일이 다운로드되고
`fname`으로 파일이름이 붙습니다. 그러므로
`example.txt` 파일의 최종위치는 `~/.keras/datasets/example.txt`가 됩니다.

tar, tar.gz, tar.bz, 그리고 zip 형식의 파일도 추출가능합니다.
다운로드 후에 해시를 전달하여 파일을 확인합니다.
명령줄 프로그램인 `shasum`과 `sha256sum`으로 해시를 계산할 수 있습니다.

__인수__

- __fname__: 파일의 이름. 절대적 경로 `/path/to/file.txt`가 명시된 경우
    바로 그 위치로 파일이 저장됩니다.
- __origin__: 파일의 본래 URL.
- __untar__: 'untar'대신 'extract'를 권장합니다. 
    파일의 압축을 풀지 여부에 대한 불리언.
- __md5_hash__: 'md5_hash'대신 'file_hash'를 권장합니다.
    파일검사용 md5 해시.
- __file_hash__: 다운로드 후 예산되는 파일의 해시 문자열.
    해시 알고리즘인 sha256과 md5 둘 모두 지원됩니다.
- __cache_subdir__: 파일이 저장되는 케라스 캐시 디렉토리 내 서브디렉토리.
    절대적 경로 `/path/to/folder`가 명시된 경우
    바로 그 위치로 파일이 저장됩니다.
- __hash_algorithm__: 파일을 검사하기 위한 해시 알고리즘을 선택합니다.
    'md5', 'sha256', 그리고 'auto'를 선택할 수 있습니다.
    디폴트 값인 'auto'는 사용중인 해시 알고리즘을 감지합니다.
- __extract__: True 값일 경우 tar 혹은 zip처럼 Archive로 파일을 추출합니다.
- __archive_format__: 파일 추출을 시도할 Archive 형식.
    'auto', 'tar', 'zip', 그리고 None을 선택할 수 있습니다.
    'tar'는 tar, tar.gz, 그리고 tar.bz 파일을 포함합니다.
    디폴트 'auto'는 ['tar', 'zip']입니다.
    None 혹은 빈 리스트는 '발견된 매치가 없음'을 반환합니다.
- __cache_dir__: 캐시된 파일을 저장할 위치.
    None일 경우 디폴트 값은 [케라스 디렉토리](/faq/#where-is-the-keras-configuration-filed-stored)입니다.

__반환값__

다운로드된 파일의 경로
    
----

### print_summary


```python
keras.utils.print_summary(model, line_length=None, positions=None, print_fn=None)
```


모델을 요약하여 프린트합니다.

__인수__

- __model__: 케라스 모델 인스턴스.
- __line_length__: 프린트된 라인의 총 개수
    (예. 터미널 창의 크기에 맞도록
    이 값을 설정합니다).
- __positions__: 각 라인의 로그 요소의 절대적 혹은 상대적 위치.
    값을 특정하지 않으면 디폴트 값인 `[.33, .55, .67, 1.]`로 설정됩니다.
- __print_fn__: 사용할 프린트 함수.
    모델 요약의 각 라인마다 호출됩니다.
    문자열 요약을 캡처하려면
    이 값을 커스텀 함수로 설정할 수 있습니다.
    디폴트 값은 `print`(stdout으로 프린트)입니다.
    
----

### plot_model


```python
keras.utils.plot_model(model, to_file='model.png', show_shapes=False, show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96)
```


케라스 모델을 도트 형식으로 전환하고 파일에 저장합니다.

__인수__

- __model__: 케라스 모델 인스턴스.
- __to_file__: 플롯 이미지의 파일 이름.
- __show_shapes__: 형태 정보를 보여줄지 여부.
- __show_layer_names__: 레이어 이름을 보여줄지 여부.
- __rankdir__: `rankdir` 인수가,
    플롯의 형식을 결정하는 문자열인 PyDot으로 전달됩니다:
    'TB'는 세로 플롯;
    'LR'는 가로 플롯을 생성합니다.
- __expand_nested__: 중첩된 모델을 클러스터로 확장할지 여부.
- __dpi__: 도트 DPI.

Returns:

A Jupyter notebook Image object if Jupyter is installed. This enables in-line display of the model plots in notebooks.
    
----

### multi_gpu_model


```python
keras.utils.multi_gpu_model(model, gpus=None, cpu_merge=True, cpu_relocation=False)
```


다른 GPU에 모델을 복제합니다.

이 함수는 구체적으로 단일기계 다중-GPU 데이터 병렬처리를 실행합니다.
다음과 같은 방식으로 작동합니다:

- 모델의 인풋을 여러 서브 배치로 나눕니다.
- 모델 복사본을 각 서브 배치에 적용합니다.
각 모델 복사본이 특별히 배정된 GPU에서 실행됩니다.
- 결과물을 (CPU에서) 연결하여 하나의 큰 배치로 만듭니다.

예. `batch_size`가 64이고 `gpus=2`라면,
인풋이 각 32개의 샘플로 구성된 2개의 서브 배치로 나뉘고,
한 GPU 당 각각의 서브 배치가 처리된 후,
64개의 처리된 샘플로 구성된 완전한 배치를 반환합니다.

8개의 GPU까지는 유사선형적인 속도증가를 유도합니다.

이 함수는 현재 탠서플로우 백엔드에서만
사용가능합니다.

__인수__

- __model__: 케라스 모델 인스턴스. 
    메모리 부족 오류를 피하기 위해서 이 모델을 CPU에 생성해두는 방법이 있습니다
    (아래의 사용법 예시를 참고하십시오).
- __gpus__: 정수 >= 2 혹은 정수 리스트, number of GPUs or
    생성된 모델 복사본을 위치시킬 GPU의 개수 혹은 GPU ID의 리스트.
- __cpu_merge__: CPU의 유효범위 내에서 모델 가중치를 합치는 것을 강제할지
    여부를 명시하는 불리언 값.
- __cpu_relocation__: CPU의 유효범위 내에서 모델 가중치를 생성할지
    여부를 명시하는 불리언 값.
    만약 이전의 어떤 장치의 유효범위에도 모델이 정의되지 않았다면,
    이 옵션을 활성화시켜 문제를 해결할 수 있습니다.

__반환값__

초기 `model` 인수와 완벽하게 동일하게 사용가능하되,
작업부하를 여러 GPU에 분산시키는 케라스 `Model` 인스턴스.

__예시__


예시 1 - CPU에서 가중치를 병합하는 모델 학습

```python
import tensorflow as tf
from keras.applications import Xception
from keras.utils import multi_gpu_model
import numpy as np

num_samples = 1000
height = 224
width = 224
num_classes = 1000

# 베이스 모델(혹은 "템플릿" 모델)을 인스턴스화합니다.
# 모델의 가중치가 CPU 메모리에 저장될 수 있도록,
# CPU 장치 유효범위 내에서 이 작업을 진행하는 것을 권합니다.
# 그렇지 않은 경우, 가중치가 GPU에 저장되어
# 가중치 공유작업이 원활치 않을 수 있습니다.
with tf.device('/cpu:0'):
    model = Xception(weights=None,
                     input_shape=(height, width, 3),
                     classes=num_classes)

# 모델을 8개의 GPU에 복제합니다.
# 이는 컴퓨터에 8개의 사용가능한 GPU가 있다고 가정하는 것입니다.
parallel_model = multi_gpu_model(model, gpus=8)
parallel_model.compile(loss='categorical_crossentropy',
                       optimizer='rmsprop')

# 가짜 데이터를 생성합니다.
x = np.random.random((num_samples, height, width, 3))
y = np.random.random((num_samples, num_classes))

# 이 `fit` 호출은 8개의 GPU에 분산됩니다.
# 배치 크기가 256이므로, 각 GPU는 32샘플을 처리합니다.
parallel_model.fit(x, y, epochs=20, batch_size=256)

# (같은 가중치를 공유하는) 템플릿 모델을 통해서 모델을 저장합니다:
model.save('my_model.h5')
```

예시 2 - cpu_relocation을 사용해 CPU에서 가중치를 병합하는 모델 학습

```python
..
# 모델 정의를 위해서 장치 유효범위를 바꿀 필요는 없습니다:
model = Xception(weights=None, ..)

try:
    parallel_model = multi_gpu_model(model, cpu_relocation=True)
    print("Training using multiple GPUs..")
except ValueError:
    parallel_model = model
    print("Training using single GPU or CPU..")
parallel_model.compile(..)
..
```

Example 3 - GPU에서 가중치를 병합하는 모델 학습 (NV-link에 권장됩니다)

```python
..
# 모델 정의를 위해서 장치 유효범위를 바꿀 필요는 없습니다:
model = Xception(weights=None, ..)

try:
    parallel_model = multi_gpu_model(model, cpu_merge=False)
    print("Training using multiple GPUs..")
except:
    parallel_model = model
    print("Training using single GPU or CPU..")

parallel_model.compile(..)
..
```

__모델 저장하기__


다중-GPU 모델을 저장하려면, `multi_gpu_model`에 의해서 반환되는 모델보다는 
템플릿 모델(`multi_gpu_model`에 전달되는 인수)과 함께
`.save(fname)` 혹은 `.save_weights(fname)`를 사용하면 됩니다.
    

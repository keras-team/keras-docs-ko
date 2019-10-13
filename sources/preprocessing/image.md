# 이미지 전처리<sub>Image Preprocessing</sub>

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/preprocessing/image.py#L238)</span>

## 이미지 데이터 생성 클래스<sub>ImageDataGenerator Class</sub>

```python
keras.preprocessing.image.ImageDataGenerator(featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False, samplewise_std_normalization=False, zca_whitening=False, zca_epsilon=1e-06, rotation_range=0, width_shift_range=0.0, height_shift_range=0.0, brightness_range=None, shear_range=0.0, zoom_range=0.0, channel_shift_range=0.0, fill_mode='nearest', cval=0.0, horizontal_flip=False, vertical_flip=False, rescale=None, preprocessing_function=None, data_format='channels_last', validation_split=0.0, interpolation_order=1, dtype='float32')
```

실시간 데이터 증강<sub>Data augmentation</sub>을 이용하여 텐서 이미지 데이터의 배치들을 생성합니다.
데이터가 배치 단위로 루프<sub>Loop</sub>를 순환합니다.

__인자__

- __featurewise_center__: `Bool`. 데이터셋의 특성 단위 <sub>Feature-wise</sub>로 입력 값의 평균이 0이 되도록 합니다.

- __samplewise_center__:  `Bool`. 각 표본<sub>Sample</sub>의 평균이 0이 되도록 합니다.

- __featurewise_std_normalization__:  `Bool`.
      입력 값을 데이터셋의 특성 단위로 구한 표준편차로 나눕니다.

- __samplewise_std_normalization__:  `Bool`. 각 입력값을 표본 단위로 구한 표준편차로 나눕니다.

- __zca_whitening__:  `Bool`. ZCA 화이트닝<sub>ZCA whitening</sub>를 적용할지 여부입니다.

- __zca_epsilon__: ZCA 화이트닝의 엡실론<SUB>Epsilon</SUB> 값. 기본값<sub>Defalut</sub>은 1e-6입니다.

- __rotation_range__: `Int`. 임의 회전의 각도<sub>Degree</sub> 범위입니다.

- __width_shift_range__: `Float`, `1-D array-like` 혹은 `Int`

  - `Float`: < 1인 경우 전체 가로넓이의 비율, >= 1인 경우 픽셀의 개수입니다.
  - `1-D array-like`: 배열<sub>Array</sub>에서 가져온 임의 요소<sub>Element</sub>입니다.
  - `Int`: `(-width_shift_range, +width_shift_range)`
    사이 구간의 픽셀 개수입니다.
  - `width_shift_range=2`인 경우 유효값은
    정수형인 `[-1, 0, +1]`로,
    `width_shift_range=[-1, 0, +1]`와 동일한 반면,
    `width_shift_range=1.0`인 경우 유효값은
    `[-1.0, +1.0)`의 구간 사이 부동 소수점입니다.

- __height_shift_range__: `float`, `1-D array-like` 혹은 `int`

  - `Float`: < 1인 경우 전체 세로높이에서의 비율, >= 1인 경우 픽셀의 개수입니다.
  - `1-D array-like`: 배열에서 가져온 임의 요소입니다.
  - `Int`: `(-height_shift_range, +height_shift_range)`
    사이 구간의 픽셀 개수입니다.
  - `height_shift_range=2`인 경우
    유효한 값은 정수형인 `[-1, 0, +1]`으로
    `height_shift_range=[-1, 0, +1]`와 동일한 반면,
    `height_shift_range=1.0`인 경우 유효한 값은 
    `[-1.0, +1.0)`의 구간 사이 부동소수점입니다.

- __brightness_range__:  두 개의 부동소수점 값으로 이루어진 리스트 혹은 튜플.
  밝기 정도를 조절할 값의 범위입니다.

- __shear_range__: `Float`. 층밀리기<sub>Shear</sub>의 강도입니다.
      (도 단위의 반시계 방향 층밀리기 각도)

- __zoom_range__: `Float` 혹은 [하한, 상한]. 임의 확대/축소의 범위입니다.
  `Float`인 경우, `[하한, 상한] = [1-zoom_range, 1+zoom_range]`입니다.

- __channel_shift_range__: `Float`. 채널의 임의 이동 범위입니다.

- __fill_mode__: {"constant", "nearest", "reflect" 혹은 "wrap"} 중 하나.
  기본값은 "nearest"입니다.
  입력 경계의 바깥 공간은 다음의 모드에 따라 다르게 채워집니다.
  - 'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
- 'nearest':  aaaaaaaa|abcd|dddddddd
  - 'reflect':  abcddcba|abcd|dcbaabcd
  - 'wrap':  abcdabcd|abcd|abcdabcd
  
- __cval__: `float` 혹은 `int`.
  `fill_mode = "constant"`인 경우
  경계 밖 공간에 사용하는 값입니다.

- __horizontal_flip__: `Bool`. 입력값을 무작위로 가로로 뒤집습니다.

- __vertical_flip__: `Bool`. 입력값을 무작위로 세로로 뒤집습니다.

- __rescale__: 크기 재조절 인수. 기본값은 None입니다.
  None 혹은 0인 경우 크기 재조절이 적용되지 않습니다,
  그 외의 값이 존재하는 경우,  변형을 적용한 후 값을 데이터와 곱합니다.
  
- __preprocessing_function__: 각 입력에 적용되는 함수.
  이미지의 크기가 재조절되고 증강된 후에 함수가 작동합니다.
  이 함수는 단일 이미지 (랭크가 3인 Numpy 텐서) 인자를 갖습니다.
  그리고 단일 이미지와 동일한 형태의 Numpy 텐서를 출력해야 합니다.
  
- __data_format__: 이미지 데이터 형식,
  "channels_first" 혹은 "channels_last"가 사용가능합니다.
  "channels_last" 모드는 이미지가 `(samples, height, width, channels)` 형태, 
  "channels_first" 모드는 이미지가 `(samples, channels, height, width)`형태임을 의미합니다.
  기본값은 `~/.keras/keras.json`에 위치한 
  케라스 구성 파일의 `image_data_format` 값으로 설정됩니다.
  따로 설정을 바꾸지 않았다면, "channels_last"가 기본값입니다.
  
- __validation_split__: `Float` ( 0과 1사이의 값으로) .검증의 용도로 남겨둘 이미지의 비율입니다.
  
- __interpolation_order__: `int`, 스플라인 보간법<sub>Spline interpolation</sub>을 사용하기 위한 순서입니다. 높을수록 느립니다.   

- __dtype__: 생성된 배열에 사용할 자료형.


__예시__

`.flow(x, y)`사용한 예시

```python
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

# 특성 단위로 정규화에 필요한 수치를 계산합니다
# (ZCA 화이트닝을 적용하는 경우, 표준편차, 평균, 그리고 주성분이 이에 해당합니다)
datagen.fit(x_train)

# 실시간 데이터 증강을 사용해 배치에 대해서 모델을 학습합니다:
model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch=len(x_train) / 32, epochs=epochs)

# 아래는 위 내용를 좀더 풀어 쓴 예시입니다
for e in range(epochs):
    print('Epoch', e)
    batches = 0
    for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=32):
        model.fit(x_batch, y_batch)
        batches += 1
        if batches >= len(x_train) / 32:
            # 제너레이터의 루프가 무한히 돌아가기 때문에
            # 직접 루프를 끊어 주어야 합니다.
            break
```

`.flow_from_directory(directory)`사용한 예시

```python
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

model.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800)
```

이미지와 마스크를 함께 변형하는 예시

```python
# 동일한 인자로 두 인스턴스를 생성합니다
data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=90,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# fit과 flow 메소드에 동일한 시드와 키워드 인자를 제공합니다
seed = 1
image_datagen.fit(images, augment=True, seed=seed)
mask_datagen.fit(masks, augment=True, seed=seed)

image_generator = image_datagen.flow_from_directory(
    'data/images',
    class_mode=None,
    seed=seed)

mask_generator = mask_datagen.flow_from_directory(
    'data/masks',
    class_mode=None,
    seed=seed)

# 제너레이터를 하나로 합쳐 이미지와 마스크를 만들어 냅니다
train_generator = zip(image_generator, mask_generator)

model.fit_generator(
    train_generator,
    steps_per_epoch=2000,
    epochs=50)
```

 ```.flow_from_dataframe(dataframe, directoryx_col, y_col)``` 사용한 예시

```python
train_df = pandas.read_csv("./train.csv")
valid_df = pandas.read_csv("./valid.csv")
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory='data/train',
        x_col="filename",
        y_col="class",
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')
validation_generator = test_datagen.flow_from_dataframe(
        dataframe=valid_df,
        directory='data/validation',
        x_col="filename",
        y_col="class",
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')
model.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800)
```

------

## 이미지 데이터 생성 방법<sub>ImageDataGenerator Method</sub>

### apply_transform

```python
apply_transform(x, transform_parameters)
```

주어진 매개변수<sub>Parameters</sub>에 따라 이미지에 변형을 가합니다.

__인자__

- __x__: 3D 텐서, 단일 이미지.
- __transform_parameters__: 문자열 딕셔너리, 어떤 변형을 할 지를 나타내는 매개변수 쌍.
  현재, 딕셔너리에서 다음과 같은 매개변수가 사용됩니다:
  - `'theta'`: `Float`. 도<sub>Degrees</sub> 단위의 회전 각도.
  
  - `'tx'`: `Float`. x 방향으로의 이동<sub>Shift</sub>.
  
  - `'ty'`: `Float`. y 방향으로의 이동.
  
  - `'shear'`: `Float`. 도 단위의 층밀리기 각도.
  
  - `'zx'`: `Float`. x 방향으로의 확대/축소.
  
  - `'zy'`: `Float`. y 방향으로의 확대/축소.
  
  - `'flip_horizontal'`: `Bool`. 가로로 뒤집기.
  
  - `'flip_vertical'`: `Bool`. 세로로 뒤집기.
  
  - `'channel_shift_intencity'`: `Float`. 채널의 이동 정도.
  
  - `'brightness'`:  `Float`. 밝기의 이동 강도.

__반환값__

* 입력값의 변형된 형태 (입력값과 동일한 형태).
      

------

### fit

```python
fit(x, augment=False, rounds=1, seed=None)
```

표본 데이터를 데이터 제너레이터<sub>Data Generator</sub>에 학습시킵니다.

이 함수는 표본 데이터 배열의 데이터-의존적<sub>Data-dependent</sub> 변형에 관련된 
내적 데이터 통계치를 계산합니다.

`featurewise_center`, `featurewise_std_normalization`,
혹은 `zca_whitening`이 True로 설정되어 있을 때만 필요합니다.

__인자__

- __x__: 표본 데이터. 랭크가 4이어야 합니다.
  흑백<sub>Grayscale</sub> 데이터의 경우 채널이 1개가 되어야 하며,
  RGB 데이터의 경우 채널 3개,
  RGBA 데이터의 경우 채널 4개가 되어야 합니다.
- __augment__: `Bool` (기본값: False).
  무작위로 증강된 표본을 학습할지 여부.
- __rounds__: `Int` (기본값: 1).
  데이터 증강을 사용하는 경우(`augment=True`),
  사용할 데이터를 몇 번이나 증강 시킬지 횟수를 지정합니다.
- __seed__: `Int` (기본값: None). 난수 시드.

------

### flow

```python
flow(x, y=None, batch_size=32, shuffle=True, sample_weight=None, seed=None, save_to_dir=None, save_prefix='', save_format='png', subset=None)
```

데이터와 레이블<sub>Lable</sub> 배열을 받아 증강된 데이터의 배치를 생성합니다.

__인자__

- __x__: 입력 데이터. 랭크 4의 NumPy 배열 혹은 튜플.
  튜플인 경우 첫 번째 원소는 이미지를 담고
  두 번때 원소는 수정 없이 출력으로 전달되는 NumPy 배열의 리스트 
  혹은 NumPy 배열의 리스트를 담습니다.
  이미지와 함께 다양한 종류의 데이터를 모델에 전달할 때 사용할 수 있습니다.
  흑백 데이터의 경우 이미지 배열의 채널이 1이어야 하며,
  RGB 데이터의 경우 채널이 3, RGBA 데이터의 경우 채널이 4 이여야 합니다.

- __y__: 레이블.

- __batch_size__: `Int` (기본값: 32).

- __shuffle__: `Bool` (기본값: True).

- __sample_weight__: 표본 가중치.

- __seed__: `Int` (기본값: None).

- __save_to_dir__: None 혹은 `str` (기본값: None).
  디렉토리를 직접 지정하여 생성되고 있는 증강 사진을 
  저장할 수 있도록 합니다.
  (현재 작업을 시각화하는데 유용합니다).

- __save_prefix__: `str` (기본값: `''`).
  저장된 사진의 파일 이름에 사용할 접두사
  (`save_to_dir`이 지정된 경우에만 유효합니다).

- __save_format__: "png"나 "jpeg" 중 하나
  (`save_to_dir`이 지정된 경우에만 유효합니다). 기본값: "png".

- __subset__: `ImageDataGenerator`에서 `validation_split`이 설정된 경우
   subset의 입력으로 `"training"` 혹은 `"validation"` 을 사용할 수 있습니다.

__반환값__

* `(x, y)` 튜플을 만들어내는 `Iterator`
   여기서 `x`는 하나의 이미지 데이터를 받는 경우 Numpy 배열이며
    여러 개의 이미지 데이터를 받는 경우 NumPy 배열의 리스트입니다.
    `y`는 `x`에 대응되는 레이블로 이루어진 Numpy 배열입니다.
    'sample_weight'이 None이 아니라면,
    생성된 튜플은 `(x, y, sample_weight)`의 형태를 갖습니다.
    `y`가 None인 경우, NumPy 배열 `x`만 반환됩니다.

------

### flow_from_dataframe

```python
flow_from_dataframe(dataframe, directory=None, x_col='filename', y_col='class', weight_col = None, target_size=(256, 256), color_mode='rgb', classes=None, class_mode='categorical', batch_size=32, shuffle=True, seed=None, save_to_dir=None, save_prefix='', save_format='png', subset=None, interpolation='nearest', validate_filenames=True)
```

데이터프레임과 디렉토리의 위치를 전달받아 증강/정규화<sub>normalize</sub>된 데이터 배치를 
생성합니다.

**간단한 튜토리얼은** [여기](http://bit.ly/keras_flow_from_dataframe)**에서 확인하실 수 있습니다.**

__인자__

- __dataframe__: 문자열 칼럼<sub>Column</sub> 이미지의 `directory` 상대 경로를 포함한 Pandas 데이터 프레임,  `directory`가 없는 경우 절대 경로를 포함하는 Pandas 데이터 프레임.  
  

`class_mode`에 따라 다양한 형태의 칼럼을 가집니다.

- `class_mode`가 `"categorical"`(기본값) 인 경우 각 이미지의 클레스(들)은 `y_col`을 포함해야 합니다.
  
    이때, 단일 클레스일 경우,  문자열/리스트/튜플
  다수 클레스일 경우, 리스트/튜플이 칼럼의 값이 될 수 있습니다. 
  
- `class_mode` 가 `"binary"` 또는 `"sparse"`인 경우 클레스 값은 `string` 형태의 `y_col`을 포함해야 합니다. 
  
- `class_mode` 가 `"raw"` 또는 `"multi_output"`인 경우 `y_col` 가 지정된 칼럼을 포함해야 합니다.
  
- `class_mode` 가 `"input"` 또는 `None` 인 경우 추가 칼럼이 필요하지 않습니다.
  
- __directory__: `string`, 이미지를 읽어오기 위한 디렉토리 경로. `None` 인 경우 'x_col'  칼럼의 데이터는 절대 경로로 지정되어 있어야 한다.

- __x_col__: `string`, 파일 이름들을 (또는 디렉토리가 `None`인 경우 절대 경로를) 포함하는 `dataframe`의 칼럼

- __y_col__: `string` 또는 `list`, 목푯값 데이터를 가지고 있는 `dataframe`의 칼럼(들)

- __weight_col__: `string`, 표본의 가중치를 포함하고 있는 `dataframe`의 칼럼. 기본값 : `None`

- __target_size__: 정수 튜플 `(높이, 넓이)`, 기본값: `(256, 256)`. 모든 이미지의 크기를 재조정하는 치수.

- __color_mode__: "grayscale", "rbg", "rgba" 중 하나. 기본값: "rgb". 이미지가 1개 혹은 3개의 색깔 채널을 갖도록 변환할 지 여부.

- __classes__: 클래스값을 지정할 수있는 클래스 리스트 (예. `['dogs', 'cats']`). 

  기본값: `None`.  특별히 값을 지정하지 않으면, 영숫자로 순서로 `y_col`의 레이블 인덱스에 매핑<sub>Map</sub>되는 클래스 리스트를 자동으로 유추할 것이다.
  `class_indices` 속성을 통해서 클래스 이름과 클래스 인덱스 간 매핑을 담은 딕셔너리를 얻을 수 있다.

- __class_mode__: "binary", "categorical", "input", "multi_output", "raw", "sparse" 혹은 `None` 중 하나. 기본값: "categorical".
  생성할 수 있는 목푯값(들)의 형태는 다음과 같다.

  - `"binary"`: 이진 레이블의 1D NumPy 배열,
  - `"categorical"`: 원-핫 인코딩<sub>One-hot encode</sub>의 2D NumPy 배열. 다중 레이블 출력값을 지원한다,
  - `"input"`: 입력 이미지와 동일한 이미지(주로 오토인코더<sub>Autoencoder</sub>에서 사용된다.),
  - `"multi_output"`: 여러 칼럼의 값들을 가지고 있는 리스트,
  - `"raw"`: `y_col`칼럼(들) 값들의 NumPy 배열,
  - `"sparse"`: 정수 레이블의 1D NumPy 배열,
  - `None`, 목푯값들이 반환되지 않습니다.(제너레이터는 `model.predict_generator()`에서 사용하는 데 유용한 이미지 데이터의 배치들만 만들어 낸다.)

- __batch_size__: 데이터 배치의 크기 (기본값: 32).

- __shuffle__: 데이터를 뒤섞을지 여부 (기본값: True)

- __seed__: 데이터 셔플링<sub>Shuffling</sub>과 변형시에 사용되는 직접 지정 할 수 있는 난수 시드.

- __save_to_dir__: `None` 혹은 `str` (기본값: None). 
  생성된 증강 사진을 특정한 디렉토리에서 저장할 수 있게 해줍니다. 이는 현재 작업을 시각화하는데 유용합니다.

- __save_prefix__: `str`. 저장된 사진의 파일 이름에 사용할 접두부호<sub>Prefix</sub> (`save_to_dir`가 설정된 경우에만 해당됩니다).

- __save_format__: "png"와 "jpeg" 중 하나 (`save_to_dir`가 설정된 경우에만 해당됩니다). 기본값: "png".

- __follow_links__: 클래스 하위 디렉토리 내 심볼릭 링크<sub> Symlink</sub>를 따라갈지 여부 (기본값: False).

- __subset__: `ImageDataGenerator`에서 `validation_split`이 설정된 경우
  subset의 입력으로 `"training"` 혹은 `"validation"` 을 사용할 수 있습니다.

- __interpolation__: 보간 방법은 불러온 이미지가 목푯값 크기와 다르면 이미지를 다시 추출할 때 사용하는 방법입니다.  
지원되는 메소드로는 `"nearest"`, `"bilinear"`, 그리고 `"bicubic"`이 있습니다.
  PIL 버전 1.1.3 이상이 설치된 경우, `"lanczos"`도 지원됩니다. PIL 버전 3.4.0 이상이 설치된 경우, `"box"`와 `"hamming"` 또한 지원됩니다. 기본값으로 `"nearest"`가 사용됩니다.
  
- __validate_filenames__: `Bool`, `x_col`의 이미지 파일 이름 검증할 지 여부, `True`면 유효하지 않은 이미지들은 무시될 것입니다. 이 옵션을 비활성화하면 flow_from_dataframe의 실행 속도가 빨라집니다.
기본값: True


__반환값__

* `(x, y)` 튜플을 만들어내는 `DataFrameIterator`에서  `x`는 `(batch_size, *target_size, channels)` 형태의 이미지 배치로 구성된 NumPy 배열이고 `y`는 `x`에 대응되는 레이블로 이루어진 NumPy 배열입니다.
      

------

### flow_from_directory

```python
flow_from_directory(directory, target_size=(256, 256), color_mode='rgb', classes=None, class_mode='categorical', batch_size=32, shuffle=True, seed=None, save_to_dir=None, save_prefix='', save_format='png', follow_links=False, subset=None, interpolation='nearest')
```

디렉토리의 경로를 전달받아 증강된 데이터의 배치를 생성합니다.

__인자__

- __directory__: `string`, 목표 디렉토리의 경로.
  반드시 한 클래스 당 하나의 하위 디렉토리가 있어야 합니다.
  각 하위 디렉토리 구조 내에 있는 PNG, JPG, BMP, PPM 혹은 TIF 이미지가 
  제너레이터에 포함됩니다.
  세부사항은 [이 스크립트](
  https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d)를 참조하십시오.
- __target_size__: 정수 튜플 ,`(높이, 넓이)`,
  기본값: `(256, 256)`.
  모든 이미지의 크기를 재조정 할 때 치수.
- __color_mode__: "grayscale", "rbg", "rgba" 중 하나. 기본값: `"rgb"`.
  변환될 이미지가 1개, 3개, 혹은 4개의 채널을 가질지 여부.
- __classes__:  클래스 값을 지정할 수 있는 클래스 하위 디렉토리
  (예. `['dogs', 'cats']`). 기본값: `None`.
  값을 지정하지 않으면, 클래스의 리스트는  `directory` 의 하위 디렉토리의 이름/구조에서 자동으로 추론 되며, 각 하위 디렉토리는 다른 클래스로 취급됩니다. 
  (그리고 레이블 인덱스에 매핑되는 클래스의 순서는 영숫자 순서를 따릅니다). 
  `class_indices` 속성을 통해서 클래스 이름에서 클래스 인덱스로
  매핑된 딕셔너리를 얻을 수 있습니다.
- __class_mode__: "categorical", "binary", "sparse",
  "input", 혹은 None 중 하나. 기본값: "categorical".
  반환될 레이블 배열의 종류를 결정합니다:
  
  - `"categorical"`은 2D형태의 원-핫 인코딩된 레이블입니다,
  - `"binary"`는 1D 형태의 이진 레이블입니다,
    `"sparse"`는 1D 형태의 정수 레이블입니다,
  - `"input"`은 입력 이미지와
    동일한 이미지입니다 (주로 오토인코더와 함께 사용합니다).
  - `None`의 경우, 어떤 레이블도 반환되지 않습니다
    (제너레이터가 이미지 데이터의 배치만 만들기 때문에,
    `model.predict_generator()`을 사용하는 것이 유용합니다).
    class_mode가 `None`일 경우,
    데이터가 `directory` 내 하위 디렉토리에 위치해야 제대로 작동한다는 점을 유의하십시오. 
- __batch_size__: 데이터 배치의 크기 (기본값: 32).
- __shuffle__: 데이터를 뒤섞을지 여부 (기본값: True)
   `False`로 설정이 된다면, 데이터는 영숫자 순서를 따를 것입니다.
- __seed__: 데이터 셔플링과 변형시에 사용되는 선택적 난수 시드.
- __save_to_dir__: `None` 혹은 `str` (디폴트 값: None).
  생성된 증강 사진을 특정한 디렉토리에서 저장할 수 있게 해줍니다. (현재 작업을 시각화하는데 유용합니다).
- __save_prefix__: `str`. 저장된 사진의 파일 이름에 사용할 접두부호
  (`save_to_dir`이 설정된 경우에만 해당됩니다.).
- __save_format__: "png"와 "jpeg" 중 하나
  (`save_to_dir`이 설정된 경우에만 해당됩니다). 기본값: "png".
- __follow_links__: 클래스 하위 디렉토리 내 심볼릭 링크를
  따라갈지 여부 (기본값: False).
- __subset__: `ImageDataGenerator`에서 `validation_split`이 설정된 경우
  subset의 입력으로 `"training"` 혹은 `"validation"` 을 사용할 수 있습니다.
- __interpolation__: 보간 방법은 불러온 이미지가 목푯값 크기와 다르면 이미지를 다시 추출할 때 사용하는 방법입니다. 지원되는 메소드로는 `"nearest"`, `"bilinear"`, 그리고 `"bicubic"`이 있습니다. PIL 버전 1.1.3 이상이 설치된 경우 `"lanczos"`도 지원됩니다. PIL 버전 3.4.0 이상이 설치된 경우는,`"box"`와 `"hamming"` 또한 지원됩니다. 기본값으로 `"nearest"`가 사용됩니다.


__반환값__

* `(x, y)` 튜플을 만들어내는 `DirectoryIterator` 
  여기서 `x`는 `(batch_size, *target_size, channels)`의 형태의
  이미지 배치로 구성된 NumPy 배열이고 `y`는 그에 대응하는 레이블로 이루어진    NumPy 배열입니다.
      

------

### get_random_transform

```python
get_random_transform(img_shape, seed=None)
```

변형하기 위한 무작위 매개변수를 생성합니다.

__인자__

- __seed__: 난수 시드.

- __img_shape__: 정수 튜플.
  변형할 이미지의 형태입니다.

__반환값__

* 변형을 위해 무작위로 선택된 매개변수를 포함한 딕셔너리.   

------

### random_transform

```python
random_transform(x, seed=None)
```

이미지에 무작위 변형을 적용합니다.

__인자__

- __x__: 3D 텐서, 단일 이미지.
- __seed__: 난수 시드.

__반환값__

* 입력이 무작위로 변형된 형태 (입력과 동일한 형태)

------

### standardize

```python
standardize(x)
```

입력의 배치에 표준화를 적용합니다.

`X` 는 주로 내부에서 이미지를 표준화<sub>Standardize</sub>하고 네트워크에 공급하기 위해 사용되기 때문에 인플레이스<sub>In-place</sub> 방식으로 변경됩니다.  `x`를 복사해 만든다면 성능 비용이 클 것입니다. 
입력 내용을 인플레이스 방식으로 변경하지 않고 이 방법을 그대로 사용하기 위해선 먼저 복사본을 생성하세요 

standarize(np.copy(x))

__인자__

​    __x__: 표준화할 입력의 배치.

__반환값__

​    표준화된 입력값.


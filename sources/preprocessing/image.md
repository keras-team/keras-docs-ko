
# 이미지 전처리

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/preprocessing/image.py#L233)</span>
## ImageDataGenerator 클래스

```python
keras.preprocessing.image.ImageDataGenerator(featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False, samplewise_std_normalization=False, zca_whitening=False, zca_epsilon=1e-06, rotation_range=0, width_shift_range=0.0, height_shift_range=0.0, brightness_range=None, shear_range=0.0, zoom_range=0.0, channel_shift_range=0.0, fill_mode='nearest', cval=0.0, horizontal_flip=False, vertical_flip=False, rescale=None, preprocessing_function=None, data_format=None, validation_split=0.0, dtype=None)
```

실시간 데이터 증강을 사용해서 텐서 이미지 데이터 배치를 생성합니다.
데이터에 대해 (배치 단위로) 루프가 순환됩니다.

__인수__

- __featurewise_center__: 불리언.
    데티터셋에 대해 특성별로 인풋의 평균이 0이 되도록 합니다.
- __samplewise_center__: 불리언. 각 샘플의 평균이 0이 되도록 합니다.
- __featurewise_std_normalization__: 불리언.
    인풋을 각 특성 내에서 데이터셋의 표준편차로 나눕니다.
- __samplewise_std_normalization__: 불리언. 각 인풋을 표준편차로 나눕니다.
- __zca_epsilon__: 영위상 성분분석 백색화의 엡실론 값. 디폴트 값은 1e-6입니다.
- __zca_whitening__: 불리언. 영위상 성분분석 백색화를 적용할지 여부입니다.
- __rotation_range__: 정수. 무작위 회전의 각도 범위입니다.
- __width_shift_range__: 부동소수점, 1D 형태의 유사배열 혹은 정수
    - 부동소수점: < 1인 경우 전체 가로넓이에서의 비율, >= 1인 경우 픽셀의 개수입니다.
    - 1D 형태의 유사배열: 배열에서 가져온 무작위 요소입니다.
    - 정수: `(-width_shift_range, +width_shift_range)`
        사이 구간의 픽셀 개수입니다.
    - `width_shift_range=2`인 경우 유효값은
        정수인 `[-1, 0, +1]`로,
        `width_shift_range=[-1, 0, +1]`와 동일한 반면,
        `width_shift_range=1.0`인 경우 유효값은
        `[-1.0, +1.0[`의 반개구간 사이 부동소수점입니다.
- __height_shift_range__: 부동소수점, 1D 형태의 유사배열 혹은 정수
    - 부동소수점: < 1인 경우 전체 세로높이에서의 비율, >= 1인 경우 픽셀의 개수입니다.
    - 1D 형태의 유사배열: 배열에서 가져온 무작위 요소입니다.
    - 정수: `(-height_shift_range, +height_shift_range)`
        사이 구간의 픽셀 개수입니다.
    - `height_shift_range=2`인 경우
        유효한 값은 정수인 `[-1, 0, +1]`으로
        `height_shift_range=[-1, 0, +1]`와 동일한 반면,
        `height_shift_range=1.0`인 경우 유효한 값은 
        `[-1.0, +1.0[`의 반개구간 사이 부동소수점입니다.
- __brightness_range__: 두 부동소수점 값으로 이루어진 리스트 혹은 튜플.
    밝기 정도를 조절할 값의 범위입니다.
- __shear_range__: 부동소수점. 층밀리기의 강도입니다.
    (도 단위의 반시계 방향 층밀리기 각도)
- __zoom_range__: 부동소수점 혹은 [하한, 상산]. 무작위 줌의 범위입니다.
    부동소수점인 경우, `[하한, 상한] = [1-zoom_range, 1+zoom_range]`입니다.
- __channel_shift_range__: 부동소수점. 무작위 채널 이동의 범위입니다.
- __fill_mode__: {"constant", "nearest", "reflect" 혹은 "wrap"} 중 하나.
    디폴트 값은 'nearest'입니다.
    인풋 경계의 바깥 공간은 다음의 모드에 따라
    다르게 채워집니다:
    - 'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
    - 'nearest':  aaaaaaaa|abcd|dddddddd
    - 'reflect':  abcddcba|abcd|dcbaabcd
    - 'wrap':  abcdabcd|abcd|abcdabcd
- __cval__: 부동소수점 혹은 정수.
    `fill_mode = "constant"`인 경우
    경계 밖 공간에 사용하는 값입니다.
- __horizontal_flip__: 불리언. 인풋을 무작위로 가로로 뒤집습니다.
- __vertical_flip__: 불리언. 인풋을 무작위로 세로로 뒤집습니다.
- __rescale__: 크기 재조절 인수. 디폴트 값은 None입니다.
    None 혹은 0인 경우 크기 재조절이 적용되지 않고,
    그 외의 경우 (다른 변형을 전부 적용한 후에)
    데이터를 주어진 값으로 곱합니다.
- __preprocessing_function__: 각 인풋에 적용되는 함수.
    이미지가 크기 재조절되고 증강된 후에 함수가 작동합니다.
    이 함수는 다음과 같은 하나의 인수를 갖습니다:
    단일 이미지 (계수가 3인 Numpy 텐서),
    그리고 동일한 형태의 Numpy 텐서를 출력해야 합니다.
- __data_format__: 이미지 데이터 형식,
    "channels_first" 혹은 "channels_last"가 사용가능합니다.
    "channels_last" 모드는 이미지의 형태가
    `(샘플, 높이, 넓이, 채널)`이어야 함을,
    "channels_first" 모드는 이미지의 형태가
    `(샘플, 채널, 높이, 넓이)`이어야 함을 의미합니다.
    디폴트 값은 `~/.keras/keras.json`에 위치한  value found in your
    케라스 구성 파일의 `image_data_format` 값으로 설정됩니다.
    따로 설정을 바꾸지 않았다면, "channels_last"가 초기값입니다.
- __validation_split__: 부동소수점. (엄격히 0과 1사이의 값으로) 검증의 용도로 남겨둘
    남겨둘 이미지의 비율입니다.
- __dtype__: 생성된 배열에 사용할 자료형.

__예시__

`.flow(x, y)`사용한 예시:

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

# 특성별 정규화에 필요한 수치를 계산합니다
# (영위상 성분분석 백색화를 적용하는 경우, 표준편차, 평균, 그리고 주성분이 이에 해당합니다)
datagen.fit(x_train)

# 실시간 데이터 증강을 사용해 배치에 대해서 모델을 학습합니다:
model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch=len(x_train) / 32, epochs=epochs)

# 다음은 보다 "수동"인 예시입니다
for e in range(epochs):
    print('Epoch', e)
    batches = 0
    for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=32):
        model.fit(x_batch, y_batch)
        batches += 1
        if batches >= len(x_train) / 32:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break
```
`.flow_from_directory(directory)`사용한 예시:

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

이미지와 마스크를 함께 변형하는 예시.

```python
# 동일한 인수로 두 인스턴스를 생성합니다
data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=90,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# fit과 flow 메서드에 동일한 시드와 키워드 인수를 제공합니다
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

# 생성기를 하나로 합쳐 이미지와 마스크를 만들어 냅니다
train_generator = zip(image_generator, mask_generator)

model.fit_generator(
    train_generator,
    steps_per_epoch=2000,
    epochs=50)
```


---
## ImageDataGenerator 메서드

### apply_transform


```python
apply_transform(x, transform_parameters)
```


주어진 매개변수에 따라 이미지에 변형을 가합니다.

__인수__

- __x__: 3D 텐서, 단일 이미지.
- __transform_parameters__: 문자열을 가진 딕셔너리 - 변형을 묘사하는
    매개변수 쌍.
    현재는 딕셔너리에서
    다음과 같은 매개변수가 사용됩니다:
    - `'theta'`: 부동소수점. 도 단위의 회전 각도.
    - `'tx'`: 부동소수점. x 방향으로의 이동.
    - `'ty'`: 부동소수점. y 방향으로의 이동.
    - `'shear'`: 부동소수점. 도 단위의 층밀리기 각도.
    - `'zx'`: 부동소수점. x 방향으로의 줌.
    - `'zy'`: 부동소수점. y 방향으로의 줌.
    - `'flip_horizontal'`: 불리언. 가로 뒤집기.
    - `'flip_vertical'`: 불리언. 세로 뒤집기.
    - `'channel_shift_intencity'`: 부동소수점. 채널 이동 강도.
    - `'brightness'`: 부동소수점. 밝기 이동 강도.

__반환값__

인풋이 변형된 버전 (인풋과 동일한 형태).
    
---
### fit


```python
fit(x, augment=False, rounds=1, seed=None)
```


샘플 데이터에 데이터 생성기를 학습시킵니다.

이는 샘플 데이터 배열을 기반으로
데이터 의존적인 변형에 관련된 내적 데이터 통걔를 계산합니다.

`featurewise_center`, `featurewise_std_normalization`,
혹은 `zca_whitening`이 참으로 설정되어 있을 때만 필요합니다.

__인수__

- __x__: 샘플 데이터. 계수가 4이어야 합니다.
 흑백 데이터의 경우
 채널 축이 1의 값을 가져야 하며,
 RGB 데이터의 경우 3,
 RGBA 데이터의 경우, 그 값이 4가 되어야 합니다.
- __augment__: 불리언 (디폴트 값: 거짓).
    무작위로 증강된 샘플에 대해서 학습할지 여부.
- __rounds__: 정수 (디폴트 값: 1).
    데이터 증강을 사용하는 경우(`augment=True`),
    사용할 데이터에 몇 번이나 증강이 적용되는지에 대한 값입니다.
- __seed__: 정수 (디폴트 값: None). 난수 시드.
   
---
### flow


```python
flow(x, y=None, batch_size=32, shuffle=True, sample_weight=None, seed=None, save_to_dir=None, save_prefix='', save_format='png', subset=None)
```


데이터와 라벨 배열을 받아 증강된 데이터의 배치를 생성합니다.

__인수__

- __x__: 인풋 데이터. 계수 4의 numpy 배열 혹은 튜플.
    튜플인 경우 첫 번째 성분이
    이미지를 담고
    두 번때 성분이 어떤 수정도 없이
    아웃풋에 전달되는 또 다른 numpy 배열
    혹은 numpy 배열의 리스트를 담습니다.
    이미지와 함께 다양한 종류의 데이터를 모델에
    전달할 때 사용할 수 있습니다.
    흑백 데이터의 경우 이미지 배열의 채널 축의 값이
    1이어야 하며,
    RGB 데이터의 경우 3,
    RGBA 데이터의 경우 4의 값을 가져야 합니다.
- __y__: 라벨.
- __batch_size__: 정수 (디폴트 값: 32).
- __shuffle__: 불리언 (디폴트 값: 참).
- __sample_weight__: 샘플 가중치.
- __seed__: 정수 (디폴트 값: None).
- __save_to_dir__: None 혹은 문자열 (디폴트 값: None).
    이는 디렉토리를 선택적으로 지정해서
    생성된 증강 사진을 저장할 수 있도록 합니다.
    (현재 작업을 시각화하는데 유용합니다).
- __save_prefix__: 문자열 (기본값: `''`).
    저장된 사진의 파일이름에 사용할 접두부호
    (`save_to_dir`이 지정된 경우에만 유의미합니다).
- __save_format__: "png"나 "jpeg" 중 하나
    (`save_to_dir`이 지정된 경우에만 유의미합니다). 디폴트 값: "png".
- __subset__: `ImageDataGenerator`에 `validation_split`이 설정된 경우
    데이터의 부분세트 (`"training"` or `"validation"`).

__반환값__

`(x, y)` 튜플을 만들어내는 `Iterator`
    여기서 `x`는 이미지 데이터로 구성된
    (단일 이미지 인풋의 경우) numpy 배열 혹은
    (추가적 인풋이 존재하는 경우) numpy 배열의 리스트이고 
    `y`는 그에 대응하는 라벨로 이루어진
    numpy 배열입니다. 'sample_weight'이 None이 아니라면,
    생성된 튜플은 `(x, y, sample_weight)`의 형태를 갖습니다.
    `y`가 None인 경우, numpy 배열 `x`만 반환됩니다.
    
---
### flow_from_dataframe


```python
flow_from_dataframe(dataframe, directory=None, x_col='filename', y_col='class', target_size=(256, 256), color_mode='rgb', classes=None, class_mode='categorical', batch_size=32, shuffle=True, seed=None, save_to_dir=None, save_prefix='', save_format='png', subset=None, interpolation='nearest', drop_duplicates=True)
```


dataframe과 디렉토리의 위치를 전달받아 증강/정규화된 데이터의 배치를 생성합니다.

__다음 링크에서 간단한 튜토리얼을 확인하실 수 있습니다: http://bit.ly/keras_flow_from_dataframe__


__인수__

- __dataframe__: Pandas dataframe containing the filepaths relative to 'directory' (or absolute paths  
if `directory` is None) of the images in a string column. It should include other column/s  
depending on the `class_mode`: - if `class_mode` is `"categorical"` (default value) it must include  
the y_col column with the class/es of each image. Values in column can be string/list/tuple if  
a single class or list/tuple if multiple classes. - if 'class_mode' is '"binary"' or '"sparse"' it must   include the given 'y_col' column with class values as strings. - if 'class_mode' is '"other"' it  
should contain the columns specified in 'y_col'. - if 'class_mode' is '"input"' or 'None' no extra  
column is needed.
- __directory__: string, path to the directory to read images from. If 'None', data in 'x_col' column  
should be absolute paths.
- __x_col__: string, column in 'dataframe' that contains the filenames (or absolute paths if 'directory' is 'None').
- __y_col__: string or list, column/s in dataframe that has the target data.
- __target_size__: 정수의 튜플 `(높이, 넓이)`, 디폴트 값: `(256, 256)`. 모든 이미지의 크기를 재조정할 치수.
- __color_mode__: "grayscale"과 "rbg" 중 하나. 디폴트 값: "rgb". 이미지가 1개 혹은 3개의 색깔 채널을 갖도록
  변환할지 여부.
- __classes__: 클래스로 이루어진 선택적 리스트 (예. `['dogs', 'cats']`). 디폴트 값: None. 특별히 값을 지정하지 않으면,   클래스로 이루어진 리스트가 `y_col`에서 자동으로 유추됩니다 (이는 영숫자순으로 라벨 색인에 대응됩니다).    `class_indices` 속성을 통해서 클래스 이름과 클래스 색인 간 매핑을 담은 딕셔너리를 얻을 수 있습니다.
- __class_mode__: "categorical", "binary", "sparse", "input", "other" 혹은 None 중 하나. 디폴트 값: "categorical".
  Mode for yielding the targets:
  - `"binary"`: 1D numpy array of binary labels,
  - `"categorical"`: 2D numpy array of one-hot encoded labels. Supports multi-label output.
  - `"sparse"`: 1D numpy array of integer labels,
  - `"input"`: images identical to input images (mainly used to work with autoencoders),
  - `"other"`: numpy array of y_col data,
  - `None`, no targets are returned (the generator will only yield batches of image data, which is
  useful to use in `model.predict_generator()`).
- __batch_size__: 데이터 배치의 크기 (디폴트 값: 32).
- __shuffle__: 데이터를 뒤섞을지 여부 (디폴트 값: 참)
- __seed__: 데이터 셔플링과 변형에 사용할 선택적 난수 시드.
- __save_to_dir__: None 혹은 문자열 (디폴트 값: None).이는 디렉토리를 선택적으로 지정해서 
  생성된 증강 사진을 저장할 수 있도록 해줍니다. (현재 작업을 시각화하는데 유용합니다).
- __save_prefix__: 문자열. 저장된 사진의 파일 이름에 사용할 접두부호 (`save_to_dir`이 설정된 경우에만 유의미합니다).
- __save_format__: "png"와 "jpeg" 중 하나 (`save_to_dir`이 설정된 경우에만 유의미합니다). 디폴트 값: "png".
- __follow_links__: 클래스 하위 디렉토리 내 심볼릭 링크를 따라갈지 여부 (디폴트 값: 거짓).
- __subset__: `ImageDataGenerator`에 `validation_split`이 설정된 경우 데이터의 부분집합 (`"training"` or `"validation"`).
- __interpolation__: Interpolation method used to resample the image if the target size is different from that of the loaded image. 지원되는 메서드로는 `"nearest"`, `"bilinear"`, 그리고 `"bicubic"`이 있습니다.
PIL 버전 1.1.3 이상이 설치된 경우, `"lanczos"`도 지원됩니다. PIL 버전 3.4.0 이상이 설치된 경우, `"box"`와 
`"hamming"` 또한 지원됩니다. 디폴트 값으로 `"nearest"`가 사용됩니다.
- __drop_duplicates__: Boolean, whether to drop duplicate rows based on filename.

__반환값__

`(x, y)` 튜플을 만들어내는 A `DataFrameIterator` 여기서 `x`는 `(배치 크기, *표적 크기, 채널)` 형태의
이미지 배치로 구성된 numpy 배열이고 `y`는 그에 상응하는 라벨로 이루어진 numpy 배열입니다.
    
---
### flow_from_directory


```python
flow_from_directory(directory, target_size=(256, 256), color_mode='rgb', classes=None, class_mode='categorical', batch_size=32, shuffle=True, seed=None, save_to_dir=None, save_prefix='', save_format='png', follow_links=False, subset=None, interpolation='nearest')
```


디렉토리에의 경로를 전달받아 증강된 데이터의 배치를 생성합니다.

__인수__

- __directory__: string, 표적 디렉토리에의 경로.
    반드시 한 클래스 당 하나의 하위 디렉토리가 있어야 합니다.
    각 하위 디렉토리 내에 위치한 
    어떤 PNG, JPG, BMP, PPM 혹은 TIF 이미지도
    생성자에 포함됩니다.
    세부사항은 [이 스크립트](
    https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d)
    를 참조하십시오.
- __target_size__: 정수 튜플 `(높이, 넓이)`,
    디폴트 값: `(256, 256)`.
    모든 이미지의 크기를 재조정할 치수.
- __color_mode__: "grayscale", "rbg", "rgba" 중 하나. 디폴트 값: "rgb".
    변환될 이미지가
    1개, 3개, 혹은 4개의 채널을 가질지 여부.
- __classes__: 클래스 하위 디렉토리의 선택적 리스트
    (예. `['dogs', 'cats']`). 디폴트 값: None.
    특별히 값을 지정하지 않으면, 각 하위 디렉토리를 각기 다른 클래스로
    대하는 방식으로 클래스의 리스트가
    `directory` 내 하위 디렉토리의 이름/구조에서
    자동으로 유추됩니다
    (그리고 라벨 색인에 대응되는 클래스의 순서는
    영숫자 순서를 따릅니다).
    `class_indices` 속성을 통해서 클래스 이름과 클래스 색인 간
    매핑을 담은 딕셔너리를 얻을 수 있습니다.
- __class_mode__: "categorical", "binary", "sparse",
    "input", 혹은 None 중 하나. 디폴트 값: "categorical".
    반환될 라벨 배열의 종류를 결정합니다:
    - "categorical"은 2D형태의 원-핫 인코딩된 라벨입니다,
    - "binary"는 1D 형태의 이진 라벨입니다,
        "sparse"는 1D 형태의 정수 라벨입니다,
    - "input"은 인풋 이미지와
        동일한 이미지입니다 (주로 자동 인코더와 함께 사용합니다).
    - None의 경우, 어떤 라벨되 반환되지 않습니다
      (생성자가 이미지 데이터의 배치만 만들기 때문에,
      `model.predict_generator()`,
      `model.evaluate_generator()` 등과 함께 사용하는 것이 유용합니다).
      class_mode가 None일 경우,
      제대로 작동하려면 데이터가 `directory` 내 하위 디렉토리에
      위치해야 한다는 점을 유의하십시오.
- __batch_size__: 데이터 배치의 크기 (디폴트 값: 32).
- __shuffle__: 데이터를 뒤섞을지 여부 (디폴트 값: 참)
- __seed__: 데이터 셔플링과 변형에 사용할 선택적 난수 시드.
- __save_to_dir__: None 혹은 문자열 (디폴트 값: None).
    이는 디렉토리를 선택적으로 지정해서
    생성된 증강 사진을 
    저장할 수 있도록 해줍니다
    (현재 작업을 시각화하는데 유용합니다).
- __save_prefix__: 문자열. 저장된 사진의 파일 이름에 사용할 접두부호
    (`save_to_dir`이 설정된 경우에만 유의미합니다).
- __save_format__: "png"와 "jpeg" 중 하나
    (`save_to_dir`이 설정된 경우에만 유의미합니다). 디폴트 값: "png".
- __follow_links__: 클래스 하위 디렉토리 내 심볼릭 링크를
    따라갈지 여부 (디폴트 값: 거짓).
- __subset__: `ImageDataGenerator`에 `validation_split`이 설정된 경우
    데이터의 부분집합 (`"training"` or `"validation"`).
- __interpolation__: 로드된 이미지의 크기와
    표적 크기가 다른 경우
    이미지를 다시 샘플링하는 보간법 메서드.
    지원되는 메서드로는 `"nearest"`, `"bilinear"`,
    그리고 `"bicubic"`이 있습니다.
    PIL 버전 1.1.3 이상이 설치된 경우 `"lanczos"`도
    지원됩니다. PIL 버전 3.4.0 이상이 설치된 경우는,
    `"box"`와 `"hamming"` 또한 지원됩니다.
    디폴트 값으로 `"nearest"`가 사용됩니다.

__반환값__

`(x, y)` 튜플을 만들어내는 `DirectoryIterator` 
    여기서 `x`는 `(배치 크기, *표적 크기, 채널)`의 형태의
    이미지 배치로 구성된 numpy 배열이고 
    `y`는 그에 대응하는 라벨로 이루어진 numpy 배열입니다.
    
---
### get_random_transform


```python
get_random_transform(img_shape, seed=None)
```


변형에 대한 무작위 매개변수를 생성합니다.

__인수__

- __seed__: 난수 시드.
- __img_shape__: 정수 튜플.
    변형할 이미지의 형태입니다.

__반환값__

변형을 묘사하는 무작위 선정된 매개변수를 포함한
딕셔너리.
    
---
### random_transform


```python
random_transform(x, seed=None)
```


이미지에 무작위 변형을 적용합니다.

__인수__

- __x__: 3D 텐서, 단일 이미지.
- __seed__: 난수 시드.

__반환값__

인풋이 무작위로 변형된 버전 (인풋과 동일한 형태 유지).
    
---
### standardize


```python
standardize(x)
```


인풋의 배치에 정규화 구성을 적용합니다.

`x` is changed in-place since the function is mainly used internally to standarize images and feed them to your network. If a copy of `x` would be created instead it would have a significant performance cost. If you want to apply this method without changing the input in-place you can call the method creating a copy before:

standarize(np.copy(x))

__인수__

- __x__: 정규화할 인풋의 배치.

__반환값__

정규화된 인풋.
    

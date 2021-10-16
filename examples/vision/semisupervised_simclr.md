# Semisupervised SimCLR

## 관련 파일

- `examples/vision/`
    - `semisupervised_simclr.ipynb` : 작업된 노트북
    - `semisupervised_simclr.md` : 작업 기록

## 작업 내역

### 2021/10/11

- [Semi-supervised image classification using contrastive pretraining with SimCLR](https://keras.io/examples/vision/semisupervised_simclr/) 을 기반으로 번역
- 설명을 조금 더 다듬음
- 일부 코드를 보기 좋게 정리
- 선행 추천 노트북을 추가함

### 2021/10/17

- [이주혁](https://github.com/Joohyuk-Lee)님의 피드백
- 일부 image 의 번역이 "영상" 이라고 된 부분을 노트북의 통일성을 위해 "이미지" 로 변경
- 오탈자 수정 (constrative -> contrastive, 서로 이미지로부터 -> 서로 다른 이미지로부터)
- `tf.keras.layers.experimental.preprocessing` 에 속해 있던 레이어 `Rescaling`, `RandomFlip`, `RandomTranslation`, `RandomZoom` 을 원본 노트북의 업데이트에 맞추어 `tf.keras.layers` 로 변경함
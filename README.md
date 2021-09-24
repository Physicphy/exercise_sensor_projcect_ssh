# 운동동작 분류

- 데이터 출처 : https://dacon.io/competitions/official/235689/data

### 목표
- 한쪽 팔에 착용한 센서 데이터만을 이용해서, 운동동작 분류(총 61가지로 분류).

### 사용한 방식
- tensorflow를 이용한 딥러닝 모델
- 각각 다른 조합의 feature를 input으로 받는 CNN과 RNN(GRU)를 조합한 모델 7개를 구성
- 7개 모델의 결과를 모두 합쳐서, Resnet을 이용해 최종 분류 학습
- data augment : data loader를 만들어서 3d rotation, permutation, translation을 적용한 뒤 학습.
- Colab-SSH로 GPU연산 사용

### 결과
- Accuracy : 0.8832
- F1-score : 0.7586
- 예측이 틀린 경우 중에서, Non-exercise(24.66%)가 가장 비율이 크다. 
- Non-exercise vs exercise를 먼저 구분할 수 있도록 개선할 필요가 있다.
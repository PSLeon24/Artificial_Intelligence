# Artificial_Intelligence
Yeongmin Ko's learning notes

### 1. K-Nearest Neighbors
- K 최근접 이웃(K-Nearest Neighbors)
  - 새로운 데이터가 들어왔을 때 기존 데이터 중 새로운 데이터와 비슷한 속성의 그룹으로 분류하는 알고리즘(Classifies unlabeled data points by assigning them the class of similar labeled data points)
- 작동 알고리즘
  - Step 1. 주변의 몇 개의 데이터와 비교할지 파라미터 k 결정(Determine parameter k (k > 0))
  - Step 2. 새 데이터와 기존 데이터 간의 거리를 계산해서 두 데이터 간의 유사도 구하기(Determine similarity by calculating the distance between a test point and all other points in the dataset)
  - Step 3. 2단계에서 계산한 거리 값에 따라 데이터 세트를 정렬(Sort the dataset according to the distance values)
  - Step 4. k번째 최근접 이웃의 범주를 결정(Determine the category of the k-th nearest neighbors)
  - Step 5. 새로운 데이터에 대해 k개의 최근접 이웃의 단순 다수결을 통해 범주를 결정(Use simple majority of the category of the k nearest neighbors as the category of a test point)
- 장점(advantages)
  - 간단하고 상대적으로 효과적(Simple and relatively effective)
- 단점(disadvantages)
  - Requires selection of an appropriate k
    - k가 너무 작으면 모델이 복잡해서 과적합(overfitting)이 발생
    - k가 너무 크면 모델이 너무 단순해져서 과소적합(underfitting)이 발생
  - Does not produce a model
    - 별도의 학습 모델을 생성하지 않기 때문에 새로운 데이터에 대해 매번 계산이 필요하므로 계산 비용이 높음
  - Nominal feature and missing data require additional processing
    - KNN은 주로 수치형 데이터에 사용되기 때문에 명목형 변수에 대해서는 라벨 인코딩이나 원핫 인코딩과 같은 방식으로 수치형으로 변환해야 하며 결측값의 경우 별도의 방식으로 전처리해야 하는 추가 비용이 발생

### 2. Naive Bayes
- 베이즈 정리(Bayes' theorem): 사전확률과 사후확률의 관계에 대해서 설명하는 정리
- ![image](https://github.com/PSLeon24/Artificial_Intelligence/assets/59058869/ec97f4bb-f437-45e8-b588-eb5dc8dbb0b2)
- <img width="191" alt="스크린샷 2024-05-30 오후 1 32 03" src="https://github.com/PSLeon24/Artificial_Intelligence/assets/59058869/1d0da83c-5617-42cf-9437-3dea2c5288c1">
- 용어 정리
  - 가설(H, Hypothesis): 가설 혹은 어떤 사건이 발생했다는 주장
  - 증거(E, Evidence): 새로운 정보
  - 우도(가능도, likelihood) = P(E|H): 가설(H)이 주어졌을 때 증거(E)가 관찰될 가능성
    - 확률 vs 우도
      - 확률: 특정 경우에 대한 상대적 비율
        - 모든 경우에 대하여 더하면 1이 됨(Mutually exclusive & exhaustive)
      - 우도: '가설'에 대한 상대적 비율
        - 가설은 얼마든지 세울 수 있고, 심지어 서로간에 포함관계가 될 수도 있음(Not mutually exclusive & Not exhaustive)
  - 사전확률(Prior probaility) = P(H): 어떤 사건이 발생했다는 주장의 신뢰도
  - 사후확률(Posterior probability) = P(H|E): 새로운 정보를 받은 후 갱신된 신뢰도
- 작동 알고리즘
  - Step 1. 주어진 클래스 라벨에 대한 사전 확률(Prior probability)을 계산
  - Step 2. 각 클래스의 각 속성으로 우도 확률(Likelihood probability) 계산
  - Step 3. 이 값을 Bayes Formula에 대입하고 사후 확률(Posterior probability)을 계산
  - Step 4. 1~3의 결과로 어떤 클래스가 높은 사후 확률을 갖게 될 지 알 수 있음(입력 값이 어떤 클래스에 더 높은 확률로 속할 수 있을지)
    
### 3. Association Mining(Apriori Algorithm)
- 연관 규칙 분석
  - 데이터에서 변수 간의 유의미한 규칙을 발견하는 데 쓰이는 알고리즘
  - e.g., 라면을 구매하는 고객이 햇반을 함께 구매할 가능성이 높다.
- 연관성 규칙 생성 과정
  - 1단계: 지지도(Support, 교사건)
    - 데이터에서 항목 집합이 얼마나 빈번하게 등장하는지를 나타내는 척도
    - Support(X) = Count(X) / n
  - 2단계: 신뢰도(Confidence, 조건부 확률)
    - 조건부 아이템(A)을 구매한 경우, 이중에서 얼마나 결론부 아이템(B)을 구매할 것인지를 나타내는 척도
    - Confidence(A → B) = Support(X, Y) / Support(X) = Support(X, Y) / Support(X)
- Apriori Algorithm
  - 연관 규칙(association rule)의 대표적인 알고리즘으로, 특정 사건이 발생했을 때 함께 발생하는 또 다른 사건의 규칙을 찾는 알고리즘
  - 작동 알고리즘
    - 1단계: 모든 항목의 빈도를 계산하여 최소 지지도(minimum support)를 넘는 항목들만 남김
    - 2단계: 남은 항목들을 조합하여 2개의 항목 집합으로 이루어진 후보 항목 집합을 만듦
    - 3단계: 2단계에서 만든 후보 항목 집합으로부터 빈도를 계산하여 최소 지지도를 넘는 항목들만 남김
    - 4단계: 후보 항목 집합이 더이상 나오지 않을 때까지 남은 항목들로부터 2~3단계를 반복 수행
    - 5단계: 각 빈발 항목 집합에 대해 모든 가능한 연관 규칙을 생성하고 각각의 신뢰도(confidence)를 계산함
    - 6단계: 신뢰도가 최소 신뢰도(minimum confidence)를 넘는 규칙들만 남김

### 4. Collaborative Filtering

### 5. Linear Regression

### 6. Logistic Regression
- 로지스틱 회귀(Logistic Regression): 입력 변수들을 선형 함수에 통과시켜 얻은 값을 활성화 함수를 통해 변환시켜 얻은 확률을 임계 함수를 통해 예측을 수행하는 구조
  - 활성화 함수: 선형 함수를 통과시켜 얻은 값을 임계 함수에 보내기 전에 변형시키는데 필요한 함수로, 주로 비선형 함수를 사용
    - Why does Activation function use nonlinear function?
      - 선형 함수(단순한 규칙)의 경우 직선으로 data를 구분하는데, 이는 아무리 층을 깊게 쌓아도 하나의 직선으로 규칙이 표현된다는 것을 뜻함. 즉, 선형 변환을 계속 반복하더라도 결국 선형 함수이므로 별 의미가 없음.
      - 그러나, 비선형 함수의 경우 여러 데이터의 복잡한 패턴을 학습할 수 있고, 계속 비선형을 유지하기 때문에 다층 구조의 유효성을 충족시킬 수 있음. 또한, 비선형 함수는 대부분 미분이 가능하기 때문에 활성화 함수로 적합함.
- Odds(오즈): 성공 확률과 실패 확률의 비율 → 특정 사건이 발생할 확률을 그 사건이 발생하지 않을 확률과 비교한 값
  - 0부터 1까지 증가할 때 오즈 비의 값은 처음에는 천천히 증가하다가 p가 1에 가까워지면 급격히 증가함
  - Odds Ratio(오즈 비): p / (1 - p) (p = 성공 확률)
    - e.g., 어떤 사건이 발생할 확률이 80%일 때의 odds ratio는?
      - 0.8 / (1 - 0.8) = 0.8 / 0.2 = 4
- Logit function(로짓 함수): 오즈의 자연 로그를 취한 값
  - logit(p) = log(p / (1 - p))
  - ![image](https://github.com/PSLeon24/Artificial_Intelligence/assets/59058869/c5c323d6-3c18-4a4c-b467-ef38b47e209d)
  - p가 0.5일 때 0이 되고 가 0과 1일 때 각각 무한대로 음수와 양수가 되는 특징을 가짐
- Sigmoid function(시그모이드 함수)
  - <img src="https://github.com/PSLeon24/Artificial_Intelligence/assets/59058869/44c8677a-74cb-4427-a015-4d3993248337" width="400px">

### 7. Perceptron

### 8. Adaline

### 9. Single Layer Neural Network

### 10. Multi Layer Neural Network

### 11. Convolutional Neural Network

### 12. Recurrent Neural Network

### 13. Long Short-Term Memory

---

## Evaluation Metrics
### 1. Classification
- 혼동행렬(Confusion matrix)
  - 혼동행렬: 예측값이 실제값과 일치하는지 여부에 따라 분류한 표(a table that categorizes predictions according to whether they match the actual value)
  - The most common performance measures consider the model's ability to discern one class versus all others
    - The class of interest is known as the positive
    - All others are known as negative
  - The relationship between the positive class and negative class predictions can be depicted as a 2 x 2 confusion matrix
    - True Positive(TP): Correctly classfied as the class of interest
    - True Negative(TN): Correctly classified as not the class of interest
    - False Positive(FP): Incorrectly classified as the class of interest
    - False Negative(FN): Incorrectly classified as not the class of interest
  - ![image](https://github.com/PSLeon24/Artificial_Intelligence/assets/59058869/11af3389-daf8-42f2-9b4d-62f8db75067c)
    - T와 F의 경우, True(참)와 False(거짓)을 나타내며, 예측값과 실제값이 일치하는 경우 T가 오고 예측값과 실제값이 다른 경우 F가 옴
    - P와 N의 경우, Positive(긍정)와 Negative(부정)을 나타내며, 예측값이 양성 클래스(1)을 나타내는 경우 P가 오고 예측값이 음성 클래스(0)을 나타내는 경우 N이 옴 
    - e.g., 예측값=0, 실제값=0인 경우, TN
    - e.g., 예측값=1, 실제값=0인 경우, FP
- 정확도(Accuracy): 2 x 2 혼동행렬에서, 아래와 같이 정확도를 수식화할 수 있음
  - <img width="326" alt="image" src="https://github.com/PSLeon24/Artificial_Intelligence/assets/59058869/1881977a-9a9a-4107-8a0d-b9b3c0b8bfd0">
- 오분류율(Error rate): 오분류율은 1에서 정확도를 빼면 됨
  - <img width="432" alt="image" src="https://github.com/PSLeon24/Artificial_Intelligence/assets/59058869/a8d70ffa-20f5-4d2e-b63d-7e8aae638e23">
- 정밀도(Precision): 정밀도는 모델의 예측값이 긍정인 것들 중 실제값이 긍정인 비율을 나타냄
  - <img width="210" alt="image" src="https://github.com/PSLeon24/Artificial_Intelligence/assets/59058869/3d1a2bc9-f6f4-4656-add9-5eb290020954">
  - 정밀도는 재현율과 헷갈리기 쉬운데, 예측값이 긍정이라는 키워드를 기억하면 분모의 수식인 TP + FP를 기억하기 쉬움
- 재현율(Recall): 재현율은 실제값이 긍정인 것들 중 예측값이 긍정인 비율을 나타냄
  - <img width="168" alt="image" src="https://github.com/PSLeon24/Artificial_Intelligence/assets/59058869/ebcc335b-6be3-427b-a6e5-6b61702655f6">
  - 재현율은 정밀도와 헷갈리기 쉬운데, 실제값이 긍정이라는 키워드를 기억하면 분모의 수식인 TP + FN을 기억하기 쉬움
- F 점수(F-Score): 정밀도와 재현율의 조화평균
  - <img width="326" src="https://github.com/PSLeon24/Artificial_Intelligence/assets/59058869/e2e410db-b2e6-4cc3-95c5-af5164d27b24">

### 2. Regression
- 평균 제곱 오차(Mean Squared Error, MSE)
- 평균 절대 오차(Mean absolute error, MAE)

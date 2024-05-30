# Artificial_Intelligence
Yeongmin Ko's learning notes

### 1. K-Nearest Neighbor

### 2. Naive Bayes
- 베이즈 정리(Bayes' theorem): 사전확률과 사후확률의 관계에 대해서 설명하는 정리
- ![image](https://github.com/PSLeon24/Artificial_Intelligence/assets/59058869/ec97f4bb-f437-45e8-b588-eb5dc8dbb0b2)
- 용어 정리
  - 가설(H, Hypothesis): 가설 혹은 어떤 사건이 발생했다는 주장
  - 증거(E, Evidence): 새로운 정보
  - 우도(가능도, likelihood) = P(E|H): 가설(H)이 증거(H)를 지지할 가능성
    - 확률 vs 우도
      - 확률: 특정 경우에 대한 상대적 비율
        - 모든 경우에 대하여 더하면 1이 됨(Mutually exclusive & exhaustive)
      - 우도: '가설'에 대한 상대적 비율
        - 가설은 얼마든지 세울 수 있고, 심지어 서로간에 포함관계가 될 수도 있음(Not mutually exclusive & Not exhaustive)
  - 사전확률(Prior probaility) = P(H): 어떤 사건이 발생했다는 주장의 신뢰도
  - 사후확률(Posterior probability) = P(H|E): 새로운 정보를 받은 후 갱신된 신뢰도

### 3. Association Mining(Apriori Algorithm)

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

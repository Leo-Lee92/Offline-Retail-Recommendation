# Offline-Retail-Recommendation
This repository covers the contents of research related to offline retail recommendation

**목표**

- 실제 오프라인 매장 환경을 그리드월드로 구축하여 매장내 고객들에게 원하는 매대를 방문한 후 계산대로 돌아오는 최적의 동선을 추천해주는 서비스 개발을 목표로 함.

- 실제 매장환경을 구축하기에 앞서 단순한 형태의 그리드 월드에서 실험하는 코드를 본 레포지토리에 업로드 함.
 
 ![그리드월드_실험환경](https://user-images.githubusercontent.com/61273017/101512460-5f472680-39be-11eb-83fa-703213b3d344.PNG)

1) 실제 매장은 구조적으로 위 그림과 같이 입구 (출발지점)와 출구 (도착지점)가 같은 열에 있음.
2) 고객은 입구를 통해 입장한 뒤 매장을 돌아다니며 목표 매대들을 방문한 뒤 (상품들을 카트에 담은 뒤) 계산대로 돌아와야 함.
3) 즉, 오프라인 매장내 고객의 동선은 C자형 커브를 그리게 됨.

**실험결과 중간요약**
- 보상을 반환하는 지점은 "방문 매대"와 "계산대"로 설정하였음
- 보상을 반환하는 지점이 한 곳일 경우, 즉 "계산대"에서만 보상을 받을 경우, 또는 보상을 반환하는 지점들이 아래 그림과 같이 서로 경로상에 나열되어 있을 경우 수렴하는 퍼포먼스를 보여주었음.

<p align="center"><img src="https://user-images.githubusercontent.com/61273017/101516731-2cebf800-39c3-11eb-8bef-b80fca9bcd42.PNG"></p>
<p align="center"><img src="https://user-images.githubusercontent.com/61273017/101516727-2bbacb00-39c3-11eb-8514-8f0434e309ef.PNG"></p>
<p align="center"><img src="https://user-images.githubusercontent.com/61273017/101516733-2cebf800-39c3-11eb-92aa-4d6b82e9927d.PNG"></p>

- 실험 세팅은 아래와 같음.
  - **학습 알고리즘:** Deep Q-learning, Deep SARSA 
  - **상태:** (0, 1), (3, 1)과 같이 좌표를 나타내는 2차원 벡터
  - **행위:** {0, 1, 2, 3} (상, 하, 좌, 우)의 4차원 벡터
  - **Agent 네트워크:** Fully Connected Perceptron
  - **옵티마이저:** Adam
  - **학습률:** 1e-20
  - **정책:** Epsilon-greedy
  - **입실론 감소율:** 0.9999

**소결**
- Fully Connected Layer 에이전트가 학습하기에 주어진 상태가 너무 단순하여 Q(a|s)를 학습한다기보다 Q(a)를 학습하게 되는 경향이 있음.

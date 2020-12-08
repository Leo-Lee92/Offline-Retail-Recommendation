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


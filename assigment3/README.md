## assignment3/README.md

````markdown
# 과제 3 – Marabou 신경망 검증

PyTorch로 학습한 간단한 분류 모델을 ONNX로 변환한 후, Marabou 도구를 사용하여 모델이 특정 조건에서 안정적으로 작동하는지를 형식적으로 검증합니다.

---

## 실행 방법

### 1. 저장소 클론

```bash
git clone https://github.com/NeuralNetworkVerification/Marabou
````

### 2. 필요한 패키지 설치

```bash
git clone https://github.com/GeonHeeJo2000/Trustworthy-AI.git
pip install -e .
pip install -r requirements.txt
```
### 2. Marabou 예제 디렉토리로 이동
용량 문제로 데이터랑 코드만 병합하면 해당 경로에 추가하면 됩니다.
```bash
[cd Marabou/maraboupy/examples/7_wineExample.py](https://github.com/GeonHeeJo2000/Trustworthy-AI/blob/main/assigment3/maraboupy/examples/7_wineExample.py)
[cd Marabou/maraboupy/examples/winequality-red.csv](https://github.com/GeonHeeJo2000/Trustworthy-AI/blob/main/assigment3/maraboupy/examples/winequality-red.csv)
```

---

## 모델 설명

* PyTorch 기반의 2층 완전연결 신경망
* 입력: `winequality-red.csv`의 정규화된 11개 피처
* 출력: 와인 품질이 6 이상인지 아닌지를 이진 분류 (1 or 0)
* 학습된 모델은 ONNX로 변환 후 Marabou에 입력됨

---

## 검증 조건

아래와 같은 조건을 Marabou에 설정하여 검증을 수행합니다:

* **입력 조건**: alcohol (10번째 입력 변수) ∈ \[0.0, 0.5]
* **출력 조건**: 출력값 ≥ 0.8

검증 코드는 `7_wineExample.py`에 포함되어 있습니다.

---

## 실행 결과 예시

```text
[INFO] Verifying with Marabou (ONNX)...
Marabou result: UNSAT
```

이는 설정한 입력 조건 내에서는 출력값이 0.8 이상이 될 수 있는 입력이 존재하지 않음을 의미하며,
모델이 해당 입력 조건 하에서 **안정적으로 작동함을 형식적으로 보장**합니다.

---

````

---

이제 이 내용을 `assignment3/README.md` 파일로 저장한 뒤 다음 명령어로 GitHub에 푸시하세요:

```bash
cd ~/geonhee/Trustworthy-AI/assigment3
echo "[README 내용 붙여넣기]" > README.md
git add README.md
git commit -m "과제3 설명용 README.md 추가"
git push origin main
````

원하시면 PDF 버전으로도 변환해드릴 수 있습니다.

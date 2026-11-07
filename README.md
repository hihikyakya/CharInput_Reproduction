# 문자 인코더 논문 재현 자료

원고 제목: 「단어 수준 언어 모델에서 토큰 임베딩과 문자 인코더 비교」

## 한 번에 실행하기

Python 3.12.14, NumPy 2.3.5에서 검증하였다. ZIP을 푼 뒤 이 README가 있는 디렉터리에서 아래 명령을 실행한다. NumPy가 이미 설치되어 있으면 첫 줄을 생략할 수 있다. 보조 실험은 GPU나 외부 코퍼스 다운로드 없이 실행된다.

```bash
python -m pip install -r ce_probe/requirements.txt
python reproduce_all.py
```

기본 결과는 `rerun/`에 저장된다. 다시 실행하려면 새로운 경로를 지정한다(예: `python reproduce_all.py --out rerun_2`). 수치 비교와 기울기 검증이 끝나면 `verification.json`을 생성한다. 표에서 반올림한 값과 비교할 때에는 각 스크립트가 생성하는 CSV의 원래 정밀도를 사용한다.

## 원고의 수치와 파일

| 원고 위치 | 구분 | 파일과 검증 범위 |
|---|---|---|
| 식 (1)–(3), 표 1 | 모델 구조의 직접 계산 | `mse_diagnostic/architecture.py`가 세 입력의 파라미터 수, 문자 수용 영역, 밀집 합성곱 곱셈–누산 수를 계산한다. Transformer의 가중치 공유 기준은 파라미터만 산정했다. |
| 그림 2, 표 2 | 수정 전 원고에 기록된 단일 실행 | `original_ce_reported.csv`는 보고값의 전사본이다. 그림 2의 개별 에포크 손실, 원본 로그, 학습 시드와 체크포인트는 첨부되지 않았다. |
| 표 3 | 수정 전 원고의 회귀 결과와 그로부터 계산한 지표 | `mse_diagnostic/reported_inputs.json`에 집계값을 보관하였다. `aggregate_audit.json`이 32.53 ≈ 22.53 × 1.44 및 정규화 오차를 검산한다. |
| 식 (4), 4.2절 | 새로 실행한 독립 토큰 대조 실험 | `mse_diagnostic/reproduce.py`가 5개 시드에서 인과 조건, 다음 토큰 노출, 목표 척도 변경을 검사한다. 시드별 결과는 `results/runs.csv`에 있다. |
| 표 4, Ⅴ절 | 새로 학습한 영어 다음 단어 예측 실험 | `ce_probe/data/`에 고정 코퍼스와 라이선스가 있다. `ce_probe/run_probe.py`가 비공유 ID, 공유 ID, 문자 CNN을 각 5시드로 학습한다. `ce_probe/results/seeds.csv`에 15개 실행이 기록되어 있다. |
| WikiText-103 후속 실험 | 별도 참고 구현 | `reference_ce/`에 모델, GPU/Colab 실행 명령, 미기재 설정의 선택과 마스크 검사가 있다. 원 코드를 확보할 수 없어 서술된 구조로 재작성하였다. 이 패키지를 작성할 때 그 폴더의 구문·전처리·파라미터 계산을 점검했으며 PyTorch 학습과 LAMBADA 평가는 수행하지 않았다. |

보조 CE 실험의 ±는 고정된 자료 분할에서 얻은 시드 간 표본 표준편차다. 자료 모집단에 대한 신뢰구간과 구분한다. 이 실험은 직전 단어 하나만 보고 다음 단어를 예측하며 원 Transformer의 재학습과 구분된다. 압축 파일에는 원고 자료에 없던 WikiText/LAMBADA 체크포인트나 예측을 포함하지 않는다.

각 폴더의 README에는 실험별 세부 절차와 가정이 있다. CPU와 BLAS가 달라지면 마지막 자릿수가 달라질 수 있으므로 수치 허용오차를 사용한다.

# WikiText-103 인과 CE 재실험용 참고 구현

이 폴더는 원고에 기재된 차원·데이터 규모·최적화 설정에서 출발해 새로 작성한 GPU 실행용 코드다. 기존 WikiText/LAMBADA 표를 산출한 저자의 프로그램·체크포인트·분할 파일은 첨부 자료에 없었다. 새로 실행한 결과는 별도 실험으로 기록한다. 배포 전 `check_protocol.py`, CLI 도움말과 모든 파일의 파이썬 구문을 확인했다. 이 폴더의 PyTorch 학습 및 GPU 검증은 수행하지 않았다. 실행된 보강 실험은 상위 패키지의 별도 폴더에 있다.

## 실행

Python 3.11 이상에서 이 폴더를 작업 디렉터리로 둔다. Colab에서는 GPU 런타임을 선택한다. `requirements.txt`의 범위는 필요한 패키지를 표시하며, 실제 사용 버전은 `config.json`으로 기록된다.

```bash
python -m pip install -r requirements.txt
python check_protocol.py
python train.py --smoke --device cpu
python train.py --wiki /content/wiki.train.tokens --lambada /content/lambada_test_plain_text.txt --seeds 11 --epochs 1 --out pilot
python train.py --wiki /content/wiki.train.tokens --lambada /content/lambada_test_plain_text.txt --seeds 11 23 37 --out paired_ce
```

- WikiText-103 훈련 파일은 한 줄 한 문단의 UTF-8 텍스트를 사용한다. 배포판의 `wiki.train.tokens`와 원문형 `wiki.train.raw`는 서로 다른 입력이다. 선택한 원본 파일명·SHA-256·학습/검증 분할을 보존한다. 데이터는 동봉하지 않았다. WikiText 배포 경로: https://huggingface.co/datasets/Salesforce/wikitext
- LAMBADA는 한 줄 한 지문인 공식 영어 테스트 평문을 사용한다. `text` 필드에 완성된 지문이 있는 JSONL도 지원한다. 마지막 공백 구분 단어를 리터럴 정답으로 보존하고, 앞 문맥만 spaCy로 나누어 마지막 64개 토큰을 사용한다. 정답 끝에 문장부호를 인위적으로 덧붙인 변형 자료는 점수가 달라질 수 있다. 데이터 출처: https://zenodo.org/records/2630551
- `--smoke`는 데이터 없이 두 분기의 총 파라미터, 미래 단어를 바꾸어도 앞 위치 은닉 상태가 일정한지, 1회 역전파 및 갱신이 되는지를 검사한다. 전체 실행보다 먼저 시도한다. `--lambada`를 생략하면 WikiText 학습·검증만 수행한다. 출력 폴더는 새 이름을 사용한다. 시드 3개, 2개 모델, 50 에포크는 6회 학습이다. 고정 슬롯의 문자 합성곱은 파라미터가 적어도 연산량이 클 수 있다. OOM 시 `--batch 8` 등으로 낮추되 두 분기 모두 같은 설정으로 새로 실행한다. 변경된 배치 크기는 원고 기본 설정과 구별한다.

## 비교 조건 및 기록

| 항목 | 이 구현의 규칙 |
|---|---|
| 분할 | 처음 5,000개 비어 있지 않은 줄을 지정된 `--split-seed`(기본 2026)로 3,500/1,500개로 무작위 분할. 이 분할은 모델·학습 시드 전체에서 동일하며, 기사별 분할은 아니다. |
| 단어 처리 | `spacy.blank('en')` 토크나이저, 공백 토큰 제외, 원래 대소문자를 유지한다. 학습 줄 전체의 빈도를 집계하고 빈도 동률은 표면형 Unicode 정렬로 푼다. 예약 기호는 PAD, UNK, EOS, BOS, MASK, SEP의 6개다. BOS/MASK/SEP는 관측열에 추가하지 않는다. EOS를 각 줄 끝에 붙이고 최대 65토큰에서 64개의 이동된 입력·정답 쌍을 만든다. |
| 어휘와 출력 | 20,000개 고정 행에서 6개 예약 행을 사용한다. 학습 단어 종류가 19,994개 미만인 경우 뒤쪽 행은 출력 분모에 포함하되 정답으로 할당하지 않는다. 입력·출력 가중치는 비공유다. 표면형별 캐시 인덱스는 철자 조회에만 쓰며 미등록 단어에 학습 가능한 단어 ID를 부여하지 않는다. |
| 문자 입력 | 영문 소문자 26개, ASCII 숫자 10개, 구두점 32개, 공백 및 0번 패딩. 소문자화 후 128슬롯, 미등록 문자와 채움 문자는 0. 입력이 다른 여러 단어를 같은 출력 UNK로 평가할 수 있다. |
| 모델 | 두 분기 모두 너비 128, 8층, 4헤드, FFN 512, pre-LN, GELU, 사인파 위치 값, 인과·패딩 마스크를 사용한다. 문자 분기는 128→64→32→16 채널, 단계당 잔차 블록 2개, 위치별 선형 스칼라 출력. 최종 LN을 추가하지 않는다. 총 6,726,176 / 4,230,145개. |
| 최적화 | AdamW, lr 0.0002, weight decay 0.01, 다른 세부값은 PyTorch 기본값, clip-norm 1, batch 16, 50 에포크, dropout 0.1, 스케줄러 없음. 같은 시드는 같은 본체·출력 초기 가중치와 에포크별 같은 줄 순서를 사용한다. 장치·라이브러리에 따라 비트 단위 결과는 달라질 수 있다. |
| 모델 선택 | 패딩을 제외한 모든 정답의 토큰 가중 CE(nats). 학습 CE는 에포크 업데이트 도중의 평균, 검증 CE는 해당 에포크 완료 후의 평균. 학습 중 검증 CE가 가장 낮은 최초 에포크를 선택하고 마지막 에포크도 저장한다. 테스트는 학습 후 두 체크포인트 모두 평가한다. |
| LAMBADA 해석 | `mapped_CE_all`은 어휘 밖 단어를 UNK로 바꾼 분류 손실이다. `mapped_CE_OOV`는 실제 OOV 철자의 확률이 아닌 UNK 확률로 계산한다. `literal_top1_IV`는 출력 어휘에 있는 정답만, `literal_top1_all`은 복원 불가능한 OOV 정답을 오답으로 포함한 정답률이다. |

`config.json`, `split.json`, `vocab.json`, 대응 시드별 CE 차이(문자−조회) 및 그 평균·표본 표준편차를 담은 `paired.json`, 각 분기의 `curve.json`, `report.json`, `best.pt`, `last.pt`를 보존한다. 처리량이나 GPU 메모리 결과는 이 코드로 측정하지 않았다. 원고에 명시되지 않았던 분할 시드, 예약 기호, 동률 처리, spaCy 언어 객체, 드롭아웃, weight decay, LAMBADA 경계 처리, 층 초기화 등은 이 구현의 명시적 선택이다. 원고의 단일 실행과 수치 일치를 보장하지 않는다.

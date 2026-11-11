# Row-wise LWE Secret Recovery

관계 보존형 입력 표현을 이용한 희소 이진 LWE 비밀키 복구 실험.

이 저장소는 작은 synthetic LWE 설정에서

```text
b = A s + e (mod q)
```

로 생성된 샘플 `A, b`로부터 희소 binary secret `s`의 support를 복구하는 학습 실험을 담고 있다. 목적은 실제 암호 파라미터를 깨는 공격 구현이 아니라, 소규모 통제 실험에서 어떤 입력 표현이 secret recovery에 중요한지 확인하는 것이다.

메인 평가지표는 `exact_match` / `exact_recovery_rate`와 `support_f1`이다. `coord_acc`는 sparse secret에서 all-zero baseline도 높게 나올 수 있으므로 보조 지표로만 본다.

## 사용권 안내

본 저장소의 소스코드는 연구 내용의 소개와 열람 그리고 재현의 목적으로 공개합니다. 관련 논문과 본 저장소에 제시된 실험·결과의 재현 및 검증을 위한 다운로드·복제·설치·실행과 이에 필요한 최소한의 환경·실험 설정 조정은 별도의 허락 없이 허용합니다.

위 재현 범위를 넘어 코드의 전부 또는 일부를 재사용·수정·재배포하거나 다른 프로젝트에 포함하려면 저자 중 해당 권리자 또는 그 권한을 위임받은 자의 사전 서면 허락을 받아야 합니다. 이 조건은 연구·교육·비상업적 목적과 상업적 목적 모두에 적용됩니다.

GitHub 서비스 약관에 따라 이미 부여되는 플랫폼 내 이용권(열람 및 fork 등)과 관련 법령에서 허용하는 행위는 위 제한의 대상에서 제외됩니다. 논문을 인용하는 것만으로 재현 범위를 넘어서는 사용 허락이 부여되지는 않으며, 본 공지는 위 재현 범위를 넘어서는 특허권이나 출원 중인 특허의 실시를 허락하지 않습니다.

해당 프로젝트는 영남대학교 산학협력단 및 저자들에게 권한이 있습니다.

사용 허락은 저장소 관리자에게 문의해 주십시오. 자세한 조건은 [LICENSE](LICENSE)에 명시되어 있습니다.

## 저장소 구성

```text
idea1/   ViT/Transformer 기반 relation-preserving encoder 탐색
idea2/   1D CNN 기반 입력 표현 ablation 및 최종 비교 실험
references.bib  관련 학술대회 논문 두 편의 BibTeX
```

학습 결과는 각 실험의 `runs/`에, 그림은 스크립트의 `--output-dir`에 생성된다.
`result/`, `runs/`, checkpoint, log, 로컬 환경과 논문 원문은 Git에서 제외하며,
이 저장소에는 과거 실험의 원시 로그와 체크포인트가 포함되어 있지 않다.

## 설치와 검증

아래 설치·실행 예시는 관련 논문과 본 저장소의 실험을 재현하기 위해 별도의 허락 없이 사용할 수 있습니다.

Python 3.10 이상에서 저장소 루트를 기준으로 실행한다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ./idea1 -e './idea2[plots]'

python -m unittest discover -s idea1/tests -v
python -m unittest discover -s idea2/tests -v
```

학습만 실행하려면 `./idea2[plots]` 대신 `./idea2`를 설치해도 된다.
`--device auto`는 CUDA를 사용할 수 있으면 CUDA를, 그 외에는 CPU를 선택한다.

## Idea 1

`idea1`은 LWE 샘플을 이미지/토큰처럼 표현해서 ViT 또는 Transformer 계열 모델로 secret support를 예측할 수 있는지 탐색한 실험 묶음이다.

핵심 질문은 다음과 같다.

```text
LWE row equation의 구조를 보존한 tokenization이 secret recovery에 도움이 되는가?
```

주요 표현은 row-wise relation grid다.

```text
X[i,j] = [ENC(A[i,j]), ENC(b[i])]
```

이 표현을 row/column block token으로 만들고, ViT-style model이 coordinate-wise secret bit를 예측한다. 과거 실험 기록에서는 `row_block`의 복구 성공과 `raw` 및 `phase` 인코딩의 차이를 보고했다. 세부 조건과 수치는 [실험 기록](idea1/docs/lwe_vit_experiment_state.md)에 남겨 두었다.

주요 코드:

- `idea1/scripts/train_lwe_vit.py`: row-block / equation-transformer / row-CNN 학습 entrypoint
- `idea1/src/lwe_vit/`: LWE data, representation, tokenization, model, metric 구현
- `idea1/CNN+TRANS/`: CNN/Transformer 비교 및 embedding/t-SNE 관련 보조 코드
- `idea1/docs/lwe_vit_experiment_state.md`: 성공/실패 실험 상태 기록

빠른 실행 예시는 `idea1/README.md`를 본다.

## Idea 2

`idea2`는 아래 [관련 논문](#관련-논문과-인용)의 관계 보존형 표현을 비교하는 1D CNN 입력 표현 ablation이다. 같은 row의 `A_i`와 `b_i` 대응, 그리고 각 coefficient 옆에 `b_i`를 반복해 배치하는 방식이 복구 성능에 미치는 영향을 확인한다.

네 가지 비교군은 다음과 같다.

```text
Proposed / relation_pair:
  X[i,j] = [phi(A[i,j]), phi(b[i])]

B-shuffle:
  X[i,j] = [phi(A[i,j]), phi(b[pi(i)])]
  row-wise A_i - b_i 대응을 깨뜨린다.

B-once:
  Row i = [phi(A[i,1]), ..., phi(A[i,n]), phi(b[i])]
  b_i를 row 끝에 한 번만 제공한다.

All-zero:
  s_hat = 0
```

`relation_pair`, `b_shuffle`, `b_once`는 같은 종류의 1D CNN row-pooling backbone을 쓰고 `--input-mode`를 바꾼다. `b_once`는 입력 채널 수와 시퀀스 길이가 다르므로 파라미터 수까지 완전히 같지는 않다. `all_zero`는 고정 헤밍 무게에 대한 이론적 baseline metric을 학습 없이 계산한다.

주요 코드:

- `idea2/scripts/train.py`: 네 가지 `input_mode` 학습/평가 entrypoint
- `idea2/src/lwe_idea2/data.py`: synthetic LWE 생성 및 `b_shuffle` 입력 생성
- `idea2/src/lwe_idea2/model.py`: `relation_pair`와 `b_once` 입력 텐서 구성 및 CNN backbone
- `idea2/src/lwe_idea2/metrics.py`: exact recovery, support F1, all-zero baseline metric
- `idea2/scripts/plot_*.py`: t-SNE 및 결과 그림 생성

논문의 대표 비교는 `n=10, m=40`과 `n=16, m=64`에서 seed 0–2의 평균과 표준편차로 정리되어 있다. [idea2 설명](idea2/README.md)에 논문에 보고된 결과와 실행 예시를 정리했다.

자세한 실행 예시는 `idea2/README.md`를 본다.

## 주요 CNN 비교 실험 실행

저장소 루트에서 `n=10, m=40`의 Proposed run을 다음처럼 실행한다.

```bash
cd idea2

python scripts/train.py \
  --input-mode relation_pair \
  --run-name n10_h2_noise3_m40_relation_pair_cnn1d_10k_1k_1k_ep100_seed0 \
  --n 10 --m 40 --q 257 \
  --h-setting fixed_h --fixed-h 2 \
  --noise-dist discrete_gaussian --noise-width 3.0 \
  --residue-encoding raw \
  --num-train 10000 --num-val 1000 --num-test 1000 \
  --epochs 100 --batch-size 512 \
  --hidden-channels 64 --depth 3 \
  --lr 3e-4 --weight-decay 1e-4 \
  --dropout 0.05 --decode top_h \
  --seed 0 --num-workers 0 --device auto
```

같은 설정에서 `--input-mode`를 `b_shuffle`, `b_once`로 바꾸고 seed를 `0,1,2`로 반복한다. 각 실행의 `--run-name`도 모드와 seed에 맞춰 바꿔 결과가 덮어써지지 않게 한다. `n=16` 비교에는 `--n 16 --m 64 --epochs 200`을 사용한다. `all_zero`는 같은 `idea2/` 디렉터리에서 다음처럼 계산한다.

```bash
python scripts/train.py \
  --input-mode all_zero \
  --run-name n10_h2_noise3_m40_all_zero_baseline \
  --n 10 --m 40 --q 257 \
  --h-setting fixed_h --fixed-h 2 \
  --noise-dist discrete_gaussian --noise-width 3.0 \
  --num-test 1000 --seed 0
```

## 결과 해석

`Proposed >> B-shuffle`이면 해당 실험 조건에서 row-wise `A_i, b_i` alignment가 유효한 신호라는 근거다.

`Proposed >> B-once`이면 이 CNN에서 coordinate-level `A_ij, b_i` pairing이 유리하다는 근거다. 입력 채널 수, 시퀀스 길이, 정규화 방식의 영향도 함께 고려해야 한다.

`B-once`가 Proposed에 가까우면 현재 작은 파라미터가 쉬워서 row 끝의 `b_i`만으로도 충분한 신호가 될 수 있음을 뜻한다. 따라서 `n=16`처럼 조금 더 어려운 설정에서 격차를 같이 보는 것이 중요하다.

각 데이터 항목은 독립적으로 생성한 `(A,b,s,e)`이며 train/val/test는 서로 다른 seed offset을 사용한다. 같은 support가 다른 split에 등장하는 것은 허용된다. 고정 헤밍 무게 `h=2`에서 가능한 support는 `n=10`이면 45개, `n=16`이면 120개이므로, 이 결과는 새로운 공개 샘플에 대한 평가이며 보지 못한 support 조합으로의 일반화를 검증한 결과는 아니다.

`--decode top_h`는 헤밍 무게를 알고 있다고 가정한다. 이 조건을 완화하려면 `--decode threshold`를 사용한다. 논문 규모의 장기 실험은 별도로 실행해야 하며, 아래 논문 수치는 현재 코드로 새로 측정한 값이 아니다.

## 관련 논문과 인용

이 저장소의 관계 보존형 CNN 입력 표현과 희소 LWE 비밀키 복구 실험에 관한 학술대회 논문이다. 인용용 BibTeX는 [references.bib](references.bib)에 제공한다.

1. Seungmin-Lee, Jian-Woo, & Sejong-Lee (2026-07-09). A Convolutional Secret Recovery Model Based on Relation-Preserved LWE Samples. Proceedings of the Korean Society of Computer Information Conference, Jeju.

2. Lee, S., Woo, J., Jo, S., Jin, M., Park, I., Jeon, J., & Lee, S. (2026-06-23). Relation-Preserving Convolutional Learning for Sparse LWE Secret Recovery. Summer Annual Conference of IEIE, Institute of Electronics and Information Engineers (IEIE), Jeju.

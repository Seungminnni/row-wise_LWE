# 관계 보존형 CNN을 통한 희소 LWE 비밀키 복구

목표는 모델 경쟁이 아니라 입력 표현 ablation이다.

본 저장소의 소스코드는 연구 내용의 소개와 열람 그리고 재현의 목적으로 공개합니다. 관련 논문과 본 저장소의 실험·결과를 재현하거나 검증하기 위한 다운로드·복제·설치·실행과 필요한 최소한의 환경·실험 설정 조정은 별도의 허락 없이 허용합니다.

위 재현 범위를 넘어서는 재사용·수정·재배포에는 해당 권리자 또는 그 권한을 위임받은 자의 사전 서면 허락이 필요합니다. GitHub 서비스 약관과 관련 법령에 따라 이미 허용되는 행위는 제외됩니다. 자세한 내용은 [LICENSE](../LICENSE)를 참조하십시오.

관련 학술대회 논문 두 편은 [저장소의 인용 안내](../README.md#관련-논문과-인용)에 정리되어 있으며, [references.bib](../references.bib)로 인용할 수 있다. 두 논문 모두 DOI가 없다.

## 설치와 검증

저장소 루트에서 [설치](../README.md#설치와-검증)를 마친 뒤 아래 명령은 `idea2/`에서 실행한다.

```bash
python -m unittest discover -s tests -v
```

## 핵심 주장

소규모 sparse binary LWE에서 secret recovery가 되는 이유가 단순히 `b`를 썼기 때문인지,
아니면 같은 row의 coefficient와 response를 coordinate-level로 정렬해 붙였기 때문인지 검증한다.

```text
Proposed:
  X[i,j] = [phi(A[i,j]), phi(b[i])]

B-shuffle:
  X[i,j] = [phi(A[i,j]), phi(b[pi(i)])]

B-once:
  Row i = [phi(A[i,1]), ..., phi(A[i,n]), phi(b[i])]

All-zero:
  s_hat = 0
```

## 고정 실험 조건

CLI 기본값:

```text
n = 10
m = 256
q = 257
secret = binary
fixed H = 2
noise sigma = 1
train/val/test = 10000/1000/1000
epochs = 30
batch = 512
```

논문에 보고된 대표 비교 실험은 다음 두 설정을 사용했다.

```text
n=10, m=40, noise_width=3.0, epochs=100, seeds=0,1,2
n=16, m=64, noise_width=3.0, epochs=200, seeds=0,1,2
train/val/test = 10000/1000/1000
q = 257
fixed H = 2
residue_encoding = raw
decode = top_h
```

## 모델

동일한 종류의 1D CNN row-pooling backbone을 사용하고 `input_mode`를 바꾼다. `b_once`는 입력 채널 수와 시퀀스 길이가 달라 파라미터 수까지 동일한 비교는 아니다. 각 run의 `config.json`에 실제 파라미터 수를 기록한다.

```text
relation_pair / b_shuffle:
  [B,m,n,C] -> [B*m,C,n] -> Conv1d -> [B,m,d,n]

b_once:
  [B,m,n+1,C0] -> [B*m,C0,n+1] -> Conv1d
  마지막 b 위치는 보조 context로만 쓰고, 앞 n개 coordinate 위치에서 예측

pooling:
  row_mean + row_max concat

head:
  Linear(2d -> 1) per coordinate
  top-H decoding
```

## 실행 예시

Proposed:

```bash
python scripts/train.py \
  --input-mode relation_pair \
  --run-name n10_h2_relation_cnn1d_seed0 \
  --n 10 --m 256 --q 257 \
  --h-setting fixed_h --fixed-h 2 \
  --noise-width 1.0 \
  --num-train 10000 --num-val 1000 --num-test 1000 \
  --epochs 30 --batch-size 512 \
  --hidden-channels 64 --depth 3 \
  --lr 3e-4 --weight-decay 1e-4 \
  --dropout 0.05 \
  --decode top_h --seed 0 \
  --device auto
```

B-shuffle:

```bash
python scripts/train.py \
  --input-mode b_shuffle \
  --run-name n10_h2_bshuffle_cnn1d_seed0 \
  --n 10 --m 256 --q 257 \
  --h-setting fixed_h --fixed-h 2 \
  --noise-width 1.0 \
  --num-train 10000 --num-val 1000 --num-test 1000 \
  --epochs 30 --batch-size 512 \
  --hidden-channels 64 --depth 3 \
  --lr 3e-4 --weight-decay 1e-4 \
  --dropout 0.05 \
  --decode top_h --seed 0 \
  --device auto
```

B-once:

```bash
python scripts/train.py \
  --input-mode b_once \
  --run-name n10_h2_bonce_cnn1d_seed0 \
  --n 10 --m 256 --q 257 \
  --h-setting fixed_h --fixed-h 2 \
  --noise-width 1.0 \
  --num-train 10000 --num-val 1000 --num-test 1000 \
  --epochs 30 --batch-size 512 \
  --hidden-channels 64 --depth 3 \
  --lr 3e-4 --weight-decay 1e-4 \
  --dropout 0.05 \
  --decode top_h --seed 0 \
  --device auto
```

All-zero baseline:

```bash
python scripts/train.py \
  --input-mode all_zero \
  --n 10 --m 256 --q 257 \
  --h-setting fixed_h --fixed-h 2 \
  --num-test 1000
```

## 해석 기준

```text
Proposed >> b_shuffle:
  해당 조건에서 row-wise alignment가 유효한 복구 신호라는 근거

Proposed >> b_once:
  해당 CNN에서 coordinate-level A_ij-b_i pairing이 유리하다는 근거

b_shuffle도 높음:
  shuffle 구현 오류 또는 shortcut 의심

b_once도 Proposed와 비슷함:
  n=10/H=2가 너무 쉬워 b를 한 번만 줘도 충분할 가능성
```

`b_shuffle`은 각 샘플 내부에서 `b`의 행 순서를 무작위로 섞으며, 평가용 원래 `b`는 보존한다. 무작위 순열에는 고정점이 남을 수 있다. `b_once`와 비교할 때는 입력 채널 수, 시퀀스 길이, GroupNorm의 영향도 함께 고려해야 한다.

## 논문에 보고된 결과

아래 값은 관련 논문에 보고된 test set 결과이며, seed 0–2의 평균 +/- 표준편차다. 과거 원시 로그와 체크포인트는 이 저장소에 포함되어 있지 않고, 현재 코드로 새로 측정한 수치도 아니다. 논문에서는 학습된 세 모드(`relation_pair`, `b_shuffle`, `b_once`)를 각 run의 best checkpoint로 평가했으며, `all_zero`는 학습 없는 baseline이다.

### n=10, m=40, 100 epochs

| Method | Exact recovery | Support F1 | Coord acc |
| --- | ---: | ---: | ---: |
| All-zero | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.800 +/- 0.000 |
| B-shuffle | 0.026 +/- 0.005 | 0.212 +/- 0.011 | 0.685 +/- 0.004 |
| B-once | 0.926 +/- 0.031 | 0.962 +/- 0.016 | 0.985 +/- 0.006 |
| Proposed / relation_pair | 0.963 +/- 0.003 | 0.982 +/- 0.002 | 0.993 +/- 0.001 |

해석:

- `B-shuffle`이 거의 실패한다. `b` 값을 보기는 하지만 `A_i`와 `b_i`의 row-wise correspondence가 깨지면 exact recovery가 0.026 수준으로 떨어진다.
- `B-once`도 매우 높다. `n=10, h=2`에서는 문제가 작아서 row 끝에 `b_i`를 한 번만 줘도 상당 부분 복구한다.
- 그래도 `Proposed`가 가장 높고 seed 간 분산도 작다. 같은 row의 `A_ij`와 `b_i`를 coordinate-level로 붙이는 표현이 가장 안정적이다.

### n=16, m=64, 200 epochs

| Method | Exact recovery | Support F1 | Coord acc |
| --- | ---: | ---: | ---: |
| All-zero | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.875 +/- 0.000 |
| B-shuffle | 0.007 +/- 0.006 | 0.123 +/- 0.002 | 0.781 +/- 0.001 |
| B-once | 0.836 +/- 0.033 | 0.917 +/- 0.017 | 0.979 +/- 0.004 |
| Proposed / relation_pair | 0.938 +/- 0.009 | 0.969 +/- 0.005 | 0.992 +/- 0.001 |

해석:

- 차원이 커지면 `B-once`와 `Proposed`의 격차가 더 뚜렷해진다. exact recovery 기준으로 `Proposed`는 0.938, `B-once`는 0.836이다.
- `B-shuffle`은 200 epochs를 돌려도 exact recovery가 0.007이었다. 올바른 row relation이 해당 조건의 복구 성능에 영향을 준다는 근거다.
- `coord_acc`만 보면 all-zero도 0.875라 높아 보인다. 그래서 이 실험의 주 지표는 coordinate accuracy가 아니라 exact recovery와 support F1이어야 한다.

### 결론

보고된 100/200 epoch 실험 모두에서 `Proposed / relation_pair`가 가장 좋은 exact recovery와 support F1을 보였다. `B-shuffle`의 성능 하락은 row-wise `A_i, b_i` alignment의 유효성을 뒷받침하며, `n=16`의 `B-once`와의 차이는 해당 CNN에서 반복 pairing이 유리할 수 있음을 보여준다. 이 관찰을 다른 모델이나 실제 암호 파라미터의 안전성으로 일반화하려면 추가 검증이 필요하다.

### 재현과 평가 범위

- `--seed`는 데이터 생성, 모델 초기화, dropout, 학습 데이터 순서에 적용한다. 하드웨어와 PyTorch 버전이 달라질 때의 비트 단위 동일성까지 보장하지는 않는다.
- 최적 체크포인트는 validation `support_f1`이 가장 높은 epoch로 선택하며, 동률이면 먼저 선택한 epoch를 유지한다.
- `--decode top_h`는 고정 헤밍 무게를 알고 있다는 가정이다. `variable_h`에는 `--decode threshold`를 사용한다. 이론적으로 계산하는 `all_zero` baseline은 `fixed_h`만 지원한다.
- train/val/test의 `(A,b,s,e)`는 서로 다른 seed offset으로 생성하지만 support 조합은 공유할 수 있다. `h=2`에서 가능한 support는 `n=10`이면 45개, `n=16`이면 120개다. 이 평가는 보지 못한 support 조합으로의 일반화를 검증하지 않는다.
- `noise_dist=discrete_gaussian`은 코드에서 정규분포 샘플을 정수로 반올림하여 구현한다.
- 출력은 `idea2/runs/<run-name>/`에 저장된다. 모드와 seed를 바꿀 때는 `--run-name`도 바꾸거나 기본 이름을 사용한다. 논문 설정의 명령은 [주요 CNN 비교 실험 실행](../README.md#주요-cnn-비교-실험-실행)을 참조한다.

## 평가 지표

보고서 기준 메인 지표는 `exact_match`, `support_f1`, `active_recall`이다.
`coord_acc`는 all-zero baseline도 `n=10,H=2`에서 0.8이 나오므로 보조 지표로만 본다.

```text
exact_match:
  s_hat 전체가 s와 완전히 같은 sample 비율

support_f1 / support_precision / support_recall:
  nonzero support 위치를 얼마나 정확히 복구했는지

active_recall:
  support_recall과 동일하게 기록

coord_acc:
  coordinate별 평균 정확도. class imbalance 때문에 단독 해석 금지

pred_residual_std_mean:
  centered_mod(b - A*s_hat, q)의 sample별 std 평균

residual_success_rate:
  pred_residual_std <= residual_success_factor * noise_width 인 sample 비율
```

metric은 batch별 score를 단순 평균하지 않고, 전체 `TP/FP/FN/count`를 누적한 뒤 계산한다.
따라서 마지막 batch 크기가 달라도 결과가 왜곡되지 않는다.

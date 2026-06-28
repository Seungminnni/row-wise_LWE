# Idea 2: LWE 1D CNN 필수 입력 표현 실험

목표는 모델 경쟁이 아니라 입력 표현 ablation이다.

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

보고서 기준 기본값:

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

최신 비교 실험은 더 명확한 ablation을 위해 다음 두 설정으로 돌렸다.

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

동일한 1D CNN row-pooling backbone을 사용하고 `input_mode`만 바꾼다.

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
cd /home/yu_mcc/vit_PQC/idea2

PYTHONPATH=src /home/yu_mcc/miniconda3/envs/lattice_env/bin/python scripts/train.py \
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
PYTHONPATH=src /home/yu_mcc/miniconda3/envs/lattice_env/bin/python scripts/train.py \
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
PYTHONPATH=src /home/yu_mcc/miniconda3/envs/lattice_env/bin/python scripts/train.py \
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
PYTHONPATH=src /home/yu_mcc/miniconda3/envs/lattice_env/bin/python scripts/train.py \
  --input-mode all_zero \
  --n 10 --m 256 --q 257 \
  --h-setting fixed_h --fixed-h 2 \
  --num-test 1000
```

## 해석 기준

```text
Proposed >> b_shuffle:
  row-wise alignment가 핵심

Proposed >> b_once:
  b 사용 자체가 아니라 coordinate-level A_ij-b_i pairing이 핵심

b_shuffle도 높음:
  shuffle 구현 오류 또는 shortcut 의심

b_once도 Proposed와 비슷함:
  n=10/H=2가 너무 쉬워 b를 한 번만 줘도 충분할 가능성
```

## 최신 결과 해석

아래 값은 test set 기준이며, seed 0-2 평균 +/- 표준편차다. 학습된 세 모드(`relation_pair`, `b_shuffle`, `b_once`)는 각 run의 best checkpoint로 test를 평가했고, `all_zero`는 학습 없는 baseline이다.

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
| B-shuffle | 0.007 +/- 0.006 | 0.123 +/- 0.002 | 0.781 +/- 0.000 |
| B-once | 0.836 +/- 0.033 | 0.917 +/- 0.017 | 0.979 +/- 0.004 |
| Proposed / relation_pair | 0.938 +/- 0.009 | 0.969 +/- 0.005 | 0.992 +/- 0.001 |

해석:

- 차원이 커지면 `B-once`와 `Proposed`의 격차가 더 뚜렷해진다. exact recovery 기준으로 `Proposed`는 0.938, `B-once`는 0.836이다.
- `B-shuffle`은 200 epochs를 돌려도 0.007로 거의 all-zero 수준의 exact recovery다. 따라서 성능은 단순히 `b`를 입력에 포함했기 때문이 아니라, 올바른 row relation을 보존했기 때문이라고 해석하는 것이 맞다.
- `coord_acc`만 보면 all-zero도 0.875라 높아 보인다. 그래서 이 실험의 주 지표는 coordinate accuracy가 아니라 exact recovery와 support F1이어야 한다.

### 결론

100/200 epoch 실험 모두에서 `Proposed / relation_pair`가 가장 좋은 exact recovery와 support F1을 보였다. 특히 `B-shuffle`의 붕괴는 row-wise `A_i, b_i` alignment가 필수 신호라는 근거다. `B-once`가 어느 정도 성공한다는 점은 `b_i` 자체에도 정보가 있음을 보여주지만, `n=16`에서 Proposed와의 격차가 커지는 것으로 보아 coordinate-level `A_ij, b_i` pairing이 더 안정적인 입력 표현이다.

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

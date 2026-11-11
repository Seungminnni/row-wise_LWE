# 관계 보존형 LWE 행 토큰화와 모델 비교

이 문서는 현재까지 구현하고 성공/비교한 1차 아이디어를 고정 기록으로 남긴다.

과거 실험의 설정과 결과를 보존한 문서이며 원시 로그와 체크포인트는 저장소에 포함되어 있지 않다. 현재 실행법은 [idea1 README](../README.md), 관계 보존형 CNN 논문 두 편의 서지정보는 [관련 논문과 인용](../../README.md#관련-논문과-인용)을 참조한다.

## 핵심 아이디어

LWE 식은 다음과 같다.

```text
b = A s + e mod q

A: [m,n]
s: [n]
e: [m]
b: [m]
```

각 row는 하나의 LWE 방정식이다.

```text
b_i = A_i1*s_1 + A_i2*s_2 + ... + A_in*s_n + e_i mod q
```

Idea 1의 핵심은 각 위치 `(i,j)`에 같은 row의 RHS `b_i`를 붙여서 relation pixel을 만드는 것이다.

```text
pixel(i,j) = [ENC(A_ij), ENC(b_i)]
```

`raw` 인코딩이면 다음과 같다.

```text
ENC(x) = x / (q - 1)
pixel(i,j) = [A_ij/(q-1), b_i/(q-1)]
```

따라서 한 row는 다음 형태가 된다.

```text
row_i = [(A_i1,b_i), (A_i2,b_i), ..., (A_in,b_i)]
```

이 표현은 자연 이미지처럼 예쁘게 보이기 위한 것이 아니라, LWE의 row-wise 선형 관계를 모델 입력 안에 보존하기 위한 것이다.

## 구현된 모델

현재 active model set은 세 개다.

```text
--model row_block
--model equation_transformer
--model row_cnn
```

### row_block

`row_block`은 현재 메인 모델이다.

```text
relation row
-> block/patch flatten
-> Linear projection
-> Transformer encoder
-> n개 secret query token
-> coordinate-wise secret prediction
```

`block_rows=1`, `block_cols=n`이면 한 LWE row 전체가 하나의 token이 된다.

예시 `n=16`, `block_cols=16`:

```text
token_i = [A_i1,b_i, A_i2,b_i, ..., A_i16,b_i]
```

이 구조는 엄밀히 말하면 자연 이미지 ViT 그 자체라기보다, LWE relation grid를 patch token으로 만드는 ViT-style Transformer다.

### equation_transformer

plain row-equation Transformer baseline이다.

```text
token_i = [ENC(A_i1), ENC(A_i2), ..., ENC(A_in), ENC(b_i)]
```

`row_block`과 같은 Transformer encoder와 query head를 쓰지만, `b_i`를 각 coefficient 옆에 반복해서 붙이지 않는다.

비교 의미:

```text
row_block vs equation_transformer
=> [A_ij,b_i] pair/broadcast tokenization이 이득인가?
```

### row_cnn

row-local CNN baseline이다.

```text
row_pixels: [B,m,n,C]
-> [B*m,C,n]
-> Conv1d over column direction only
-> [B,m,hidden,n]
-> mean pooling over m rows
-> coordinate-wise secret prediction
```

중요한 점:

```text
Conv1d는 한 LWE row 내부에서만 작동한다.
서로 다른 row는 이미지 이웃처럼 convolution하지 않는다.
row 간 정보 결합은 단순 mean pooling으로만 수행한다.
```

비교 의미:

```text
row_block vs row_cnn
=> 같은 [A_ij,b_i] relation input에서 Transformer attention/query가 이득인가?
```

## 인코딩

현재 공통 residue encoding은 다음 세 가지다.

```text
raw:
  x / (q - 1)

centered:
  centered_mod(x,q) / (q/2)

phase:
  centered_mod(x,q)/(q/2),
  sin(2*pi*x/q),
  cos(2*pi*x/q)
```

`phase`를 A와 b에 모두 쓰면 relation pixel channel 수는 6개가 된다.

```text
[centered(A), sin(A), cos(A), centered(b), sin(b), cos(b)]
```

과거 `fourier_k=2`는 제거했다. 현재 결론은 추가 harmonic이 핵심은 아니었다는 것이다.

## Loss와 Metric

현재 대표 실험은 CE-only를 기본으로 둔다.

```text
loss = coordinate-wise weighted cross entropy
residual_loss_weight = 0
```

fixed-h sparse binary secret에서는 0 class가 많기 때문에 `inverse_prior` class weight를 사용한다.

예시 `n=10`, `h=2`:

```text
nonzero rate = 0.2
class_weight_0 = 0.4
class_weight_1 = 1.6
```

residual consistency는 학습 loss가 아니라 검증 지표로 본다.

```text
r = centered_mod(b - A*s_hat, q)
```

좋은 예측이면 `r`의 std가 oracle noise std와 가까워진다.

주요 metric:

```text
coord_acc
exact_match
support_f1
support_precision
support_recall
macro_f1
pred_residual_std_mean
pred_residual_std_gap_mean
residual_success_rate
```

## 데이터 생성과 Split

synthetic LWE를 생성한다.

```text
A ~ Uniform(Z_q)^{m x n}
s ~ binary sparse secret
e ~ discrete Gaussian or configured noise
b = A s + e mod q
```

train/val/test는 seed offset으로 분리한다.

```text
train seed = run_seed * 1000 + 11
val seed   = run_seed * 1000 + 23
test seed  = run_seed * 1000 + 37
```

`--on-the-fly`를 쓰면 1M급 dataset을 RAM에 저장하지 않고 `__getitem__`에서 deterministic seed 기반으로 즉시 생성한다.

## 성공한 대표 결과

### row_block, raw, n=16, m=256

경로:

```text
runs/lwe_vit/raw_ceonly_bc16_n16_m256_100k_10k_10k_ep150
```

설정:

```text
model=row_block
residue_encoding=raw
n=16
m=256
q=257
fixed_h=2
block_rows=1
block_cols=16
num_train=100000
num_val=10000
num_test=10000
epochs=150
batch_size=512
embed_dim=128
depth=4
num_heads=8
residual_loss_weight=0
```

최종 로그 기준:

```text
train_eval exact_match=0.9985
val exact_match=0.9961
test exact_match=0.9984
test coord_acc=0.9999
test support_f1=0.9995
test pred_residual_std_mean=1.1532
test residual_success_rate=0.9984
```

해석:

```text
raw A,b만 사용해도 fixed_h=2, n=16, m=256에서는 거의 완전 복원된다.
```

### row_cnn, raw, n=10, m=128

경로:

```text
runs/lwe_vit/raw_ceonly_cnn_h2_n10_m128_100k_10k_10k_ep150_e128_d4
```

이 run은 이름에 `cnn`이 남아 있지만 현재 코드상 대응 모델은 `row_cnn`이다.

중간 로그 기준:

```text
epoch 44:
  train exact_match=1.0000
  val exact_match=0.9999
  train_loss=0.0000216
  val_loss=0.0000314

epoch 54:
  train exact_match=1.0000
  val exact_match=0.9999
  train_loss=0.0000059
  val_loss=0.0000228
```

해석:

```text
n=10, m=128, h=2에서는 row-local CNN도 거의 완벽하게 맞춘다.
따라서 이 쉬운 세팅에서는 Transformer attention만의 이득을 주장하기 어렵다.
```

## 왜 잘 맞는가

성공한 조건은 매우 제한된 toy LWE다.

```text
binary secret
fixed_h=2
n=10 또는 n=16
m=128 또는 m=256
noise_width=1.0
train=100k
```

가능한 support 후보 수가 작다.

```text
n=10, h=2: C(10,2)=45
n=16, h=2: C(16,2)=120
```

그리고 한 샘플 안에 방정식이 많다.

```text
m=128 또는 256
```

`h=2`에서는 다음 구조가 강하게 드러난다.

```text
b_i ~= A_ia + A_ib + e_i mod q
```

즉 모델은 많은 row를 보며 "어떤 두 column의 조합이 b와 반복적으로 맞는가"를 학습할 수 있다.

## 현재 결론

Idea 1은 다음을 보여준다.

```text
1. [A_ij,b_i] relation-row 표현은 매우 강하다.
2. fixed_h=2 작은 차원에서는 raw A,b만으로도 잘 맞춘다.
3. row_cnn도 성공하므로, 쉬운 세팅에서는 attention이 필수라고 보기 어렵다.
4. Transformer/ViT-style의 의미를 보려면 더 어려운 설정에서 row_block이 row_cnn보다 더 버티는지 확인해야 한다.
```

## 대표 명령어

row_block, n=10, m=128:

```bash
cd idea1

python scripts/train_lwe_vit.py \
  --model row_block \
  --run-name raw_ceonly_bc10_h2_n10_m128_100k_10k_10k_ep150_e128_d4 \
  --h-setting fixed_h --fixed-h 2 \
  --n 10 --m 128 --q 257 \
  --noise-dist discrete_gaussian --noise-width 1.0 \
  --block-rows 1 --block-cols 10 \
  --residue-encoding raw \
  --num-train 100000 --num-val 10000 --num-test 10000 \
  --train-eval-samples 10000 \
  --epochs 150 \
  --batch-size 512 \
  --embed-dim 128 --depth 4 --num-heads 8 \
  --lr 3e-4 --weight-decay 1e-4 \
  --residual-loss-weight 0 \
  --num-workers 8 \
  --device auto \
  --save-best \
  --output-dir runs/lwe_vit
```

equation_transformer, n=10, m=128:

```bash
cd idea1

python scripts/train_lwe_vit.py \
  --model equation_transformer \
  --run-name raw_ceonly_eqtrans_h2_n10_m128_100k_10k_10k_ep150_e128_d4 \
  --h-setting fixed_h --fixed-h 2 \
  --n 10 --m 128 --q 257 \
  --noise-dist discrete_gaussian --noise-width 1.0 \
  --residue-encoding raw \
  --num-train 100000 --num-val 10000 --num-test 10000 \
  --train-eval-samples 10000 \
  --epochs 150 \
  --batch-size 512 \
  --embed-dim 128 --depth 4 --num-heads 8 \
  --lr 3e-4 --weight-decay 1e-4 \
  --residual-loss-weight 0 \
  --num-workers 8 \
  --device auto \
  --save-best \
  --output-dir runs/lwe_vit
```

row_cnn, n=10, m=128:

```bash
cd idea1

python scripts/train_lwe_vit.py \
  --model row_cnn \
  --run-name raw_ceonly_rowcnn_h2_n10_m128_100k_10k_10k_ep150_e128_d4 \
  --h-setting fixed_h --fixed-h 2 \
  --n 10 --m 128 --q 257 \
  --noise-dist discrete_gaussian --noise-width 1.0 \
  --residue-encoding raw \
  --num-train 100000 --num-val 10000 --num-test 10000 \
  --train-eval-samples 10000 \
  --epochs 150 \
  --batch-size 512 \
  --embed-dim 128 --depth 4 --num-heads 8 \
  --lr 3e-4 --weight-decay 1e-4 \
  --residual-loss-weight 0 \
  --num-workers 8 \
  --device auto \
  --save-best \
  --output-dir runs/lwe_vit
```

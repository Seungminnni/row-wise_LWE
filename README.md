# vit_PQC

Sparse binary LWE secret-support recovery experiments.

이 저장소는 작은 synthetic LWE 설정에서

```text
b = A s + e (mod q)
```

로 생성된 샘플 `A, b`로부터 희소 binary secret `s`의 support를 복구하는 학습 실험을 담고 있다. 목적은 실제 암호 파라미터를 깨는 공격 구현이 아니라, 소규모 통제 실험에서 어떤 입력 표현이 secret recovery에 중요한지 확인하는 것이다.

메인 평가지표는 `exact_match` / `exact_recovery_rate`와 `support_f1`이다. `coord_acc`는 sparse secret에서 all-zero baseline도 높게 나올 수 있으므로 보조 지표로만 본다.

## Repository Layout

```text
idea1/   ViT/Transformer 기반 relation-preserving encoder 탐색
idea2/   1D CNN 기반 입력 표현 ablation 및 최종 비교 실험
result/  생성된 그림, CSV, checkpoint, 로그 모음
runs/    이전/기타 실험 산출물
```

`result/`, `runs/`, checkpoint, log, local environment 파일은 기본적으로 git에서 제외한다.

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

이 표현을 row/column block token으로 만들고, ViT-style model이 coordinate-wise secret bit를 예측한다. `row_block` 구조가 가장 명확히 성공한 축이며, `phase` encoding은 학습을 빠르게 만들지만 핵심은 additional Fourier harmonic이 아니라 row-equation token 구조라는 결론에 가깝다. `raw` encoding도 충분한 `m`, epoch에서는 성공했다.

주요 코드:

- `idea1/scripts/train_lwe_vit.py`: row-block / equation-transformer / row-CNN 학습 entrypoint
- `idea1/src/lwe_vit/`: LWE data, representation, tokenization, model, metric 구현
- `idea1/CNN+TRANS/`: CNN/Transformer 비교 및 embedding/t-SNE 관련 보조 코드
- `idea1/docs/lwe_vit_experiment_state.md`: 성공/실패 실험 상태 기록

빠른 실행 예시는 `idea1/README.md`를 본다.

## Idea 2

`idea2`는 최종 주장에 더 가까운, 작고 선명한 1D CNN 입력 표현 ablation이다. 목표는 모델 경쟁이 아니라 `b`를 썼기 때문에 되는 것인지, 아니면 같은 row의 `A_i`와 `b_i` 대응을 coordinate-level로 보존했기 때문에 되는 것인지 분리해서 확인하는 것이다.

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

`relation_pair`, `b_shuffle`, `b_once`는 같은 1D CNN row-pooling backbone을 쓰고 `--input-mode`만 바꾼다. `all_zero`는 학습 없이 baseline metric만 계산한다.

주요 코드:

- `idea2/scripts/train.py`: 네 가지 `input_mode` 학습/평가 entrypoint
- `idea2/src/lwe_idea2/data.py`: synthetic LWE 생성 및 `b_shuffle` 입력 생성
- `idea2/src/lwe_idea2/model.py`: `relation_pair`와 `b_once` 입력 텐서 구성 및 CNN backbone
- `idea2/src/lwe_idea2/metrics.py`: exact recovery, support F1, all-zero baseline metric
- `idea2/scripts/plot_*.py`: t-SNE 및 결과 그림 생성

대표 비교는 `n=10, m=40`과 `n=16, m=64`에서 seed 0-2 평균으로 정리했다. 포스터/보고서용 최종 그림과 CSV는 `result/plots/`와 `result/` 아래에 생성되어 있다.

자세한 실행 예시는 `idea2/README.md`를 본다.

## Reproducing The Main Idea 2 Runs

예를 들어 `n=10, m=40`의 Proposed run은 다음처럼 실행했다.

```bash
cd /home/yu_mcc/vit_PQC/idea2

PYTHONPATH=src /home/yu_mcc/miniconda3/envs/lattice_env/bin/python scripts/train.py \
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
  --seed 0 --num-workers 8 --device auto
```

같은 설정에서 `--input-mode`를 `b_shuffle`, `b_once`로 바꾸고 seed를 `0,1,2`로 반복하면 ablation 결과를 재현할 수 있다. `all_zero`는 다음처럼 따로 계산한다.

```bash
PYTHONPATH=src /home/yu_mcc/miniconda3/envs/lattice_env/bin/python scripts/train.py \
  --input-mode all_zero \
  --run-name n10_h2_noise3_m40_all_zero_baseline \
  --n 10 --m 40 --q 257 \
  --h-setting fixed_h --fixed-h 2 \
  --noise-dist discrete_gaussian --noise-width 3.0 \
  --num-test 1000 --seed 0
```

## Interpretation Guide

`Proposed >> B-shuffle`이면 row-wise `A_i, b_i` alignment가 핵심이라는 근거다.

`Proposed >> B-once`이면 단순히 `b`를 제공한 효과가 아니라 coordinate-level `A_ij, b_i` pairing이 중요하다는 근거다.

`B-once`가 Proposed에 가까우면 현재 작은 파라미터가 쉬워서 row 끝의 `b_i`만으로도 충분한 신호가 될 수 있음을 뜻한다. 따라서 `n=16`처럼 조금 더 어려운 설정에서 격차를 같이 보는 것이 중요하다.

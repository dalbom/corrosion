# Corrosion 생성 모델 — 권장 Recipe (LPIPS + BatchNorm)

**작성일**: 2026-05-06
**적용 대상**: cGAN (검증 완료), DDPM / DiT / cVAE (적용 예정)
**Branch**: `feat/recon-loss-pilot` on `/home/dalbom/dev/13_codes_corrosion_diffusion`

---

## 1. 핵심 변경 (3가지)

| 항목 | 변경 전 | 변경 후 |
|---|---|---|
| Generator loss | adversarial only (cGAN) / denoising MSE only (diffusion) | + L1 + (1−SSIM) + LPIPS |
| Conditioning input | flat 그대로 Linear/MLP에 입력 | `nn.BatchNorm1d(cond_dim)` 통과 후 입력 |
| Best-model 기준 | val MACE (1-D 메트릭) | val L1 + λ_ratio·(1−SSIM) + λ_ratio·LPIPS |

---

## 2. Loss Function 정확한 형태

```python
# cGAN의 generator update에 추가 (단일 채널 R [-1,1] 가정)
loss_g_adv  = -score_fake.mean()                                   # 기존 WGAN-GP
loss_g_l1   = F.l1_loss(fake, real)
loss_g_ssim = 1.0 - ssim(fake, real, data_range=2.0, win_size=11)  # pytorch_msssim
loss_g_perc = lpips_fn(fake.repeat(1,3,1,1),
                       real.repeat(1,3,1,1)).mean()                # lpips, VGG net

loss_g = loss_g_adv + 100 * loss_g_l1 + 50 * loss_g_ssim + 10 * loss_g_perc
loss_g.backward()
```

### Hyperparameters (검증된 값)
- `lambda_l1 = 100.0` (pix2pix 표준)
- `lambda_ssim = 50.0` (L1의 절반)
- `lambda_perceptual = 10.0` (pix2pix-HD 표준)

### 라이브러리
- `pytorch_msssim` (`pip install pytorch_msssim`) — single-scale SSIM, win_size=11, data_range=2.0 (이미지 [-1, 1])
- `lpips` (`pip install lpips`) — `lpips.LPIPS(net='vgg', verbose=False)`, frozen, eval()
- LPIPS 입력: 3채널 [-1, 1] 요구 → 단일 채널 R을 `repeat(1,3,1,1)`로 복제

### Diffusion methods (DDPM / DiT)에 적용할 때
- 기존 denoising MSE는 유지
- 추가 reconstruction loss는 **denoised x_0 예측에 적용**:
  - `x_0_pred = denoise(x_t, t, cond)`
  - `loss_recon = lambda_l1·L1(x_0_pred, x_0) + lambda_ssim·(1-SSIM(x_0_pred, x_0)) + lambda_perc·LPIPS(...)`
- 또는 step별 weight 적용 (작은 t에서만 — high-noise step에서는 의미 없음)

### cVAE에 적용할 때
- 기존 reconstruction loss (MSE 또는 L1)을 위 3가지 조합으로 교체
- KL term은 유지

---

## 3. BatchNorm Conditioning Input

### 위치
Generator와 Critic(/Discriminator/Decoder)의 **conditioning vector가 처음 들어오는 지점**에 BN 추가:

```python
# Generator
class Generator(nn.Module):
    def __init__(self, ..., cond_dim, cond_norm_type='batchnorm'):
        ...
        self.cond_norm = nn.BatchNorm1d(cond_dim, affine=True) \
                         if cond_norm_type == 'batchnorm' else nn.Identity()

    def forward(self, z, cond):
        cond = self.cond_norm(cond)        # ← 여기
        x = torch.cat([z, cond], dim=1)
        ...

# Critic / Discriminator
class Critic(nn.Module):
    def forward(self, img, cond):
        cond = self.cond_norm(cond)        # ← 여기
        embed = self.embed_cond(cond)
        ...
```

### 왜 이게 필요한가
- 채널별 scale 비대칭 (corrosion 데이터: S magnitude std~5, Phase std~100)이 critic의 projection score를 폭발시킴 → WGAN-GP 학습 실패
- 데이터셋 레벨 z-score는 fix하지만 **2ch에서 절대 magnitude 정보 손실로 성능 후퇴**
- **BatchNorm은 learnable affine (γ, β)으로 magnitude 회복 가능** → 모든 채널 조합에서 안정 + 성능

### Diffusion methods에 적용할 때
- **DiT**: `cond_embed_proj` 또는 condition embedding의 첫 입력에 BN
- **DDPM (UNet)**: time embedding 외 conditioning을 받는 곳 (각 ResBlock의 cond projection 입력)에 BN
- **cVAE**: encoder/decoder의 cond input에 BN

---

## 4. Best-Model Selection 기준

### 변경 이유
순수 WGAN-GP + val MACE 선정 시 **epoch 11-25의 "blurry mean-like" artifact**가 best로 선택됨 (실제로 학습된 게 아니라 우연히 R-mean 맞음). 8개 weak model이 정확히 이 케이스.

### 새 기준
**val L1 + (λ_ssim/λ_l1)·(1-val_SSIM) + (λ_perc/λ_l1)·val_LPIPS** — 학습 objective의 supervised 부분과 일치 (adversarial term 제외; Wasserstein scale drift)

```python
ssim_to_l1_ratio = lambda_ssim / lambda_l1      # 0.5 (=50/100)
perc_to_l1_ratio = lambda_perceptual / lambda_l1  # 0.1 (=10/100)
score = val_l1 + ssim_to_l1_ratio * (1 - val_ssim) + perc_to_l1_ratio * val_lpips

if score < best_score:
    save_checkpoint(...)
```

Val MACE는 **logging만 하고 selection엔 사용 안 함** (transparency).

---

## 5. Inference 시 주의

### Conditioning 정규화 정보를 checkpoint에 저장
```python
# train_cgan.py — checkpoint 저장 시
torch.save({
    'generator_state_dict': ...,
    'cond_norm_type': args.cond_norm_type,  # 'none' or 'batchnorm'
    'normalize_cond': args.normalize_cond,    # legacy dataset z-score
    'cond_stats': train_dataset.cond_stats,   # legacy z-score만 사용
    ...
}, path)
```

### Inference에서 같은 정규화 재현
```python
# inference_cgan.py
cond_norm_type = checkpoint.get('cond_norm_type', 'none')
generator = Generator(..., cond_norm_type=cond_norm_type)
generator.load_state_dict(checkpoint['generator_state_dict'])  # BN running stats 함께 로드됨
```

BN의 running mean/std는 state_dict에 자동 포함 — 별도 처리 불필요.

---

## 6. 검증된 결과 (cGAN, Trial 1 × S11_S21, 1164 test samples)

### Best config: **2ch (S11+S21) + BatchNorm + L1+SSIM+LPIPS**

| Variant | MAE | MSE | PSNR | SSIM | **MACE** | Raw↔Matched gap |
|---|---|---|---|---|---|---|
| baseline (no recon) raw | 0.1012 | 0.0175 | 18.28 | 0.5714 | 6.81 | — |
| baseline matched | 0.0977 | 0.0178 | 18.22 | 0.5587 | 5.51 | **1.30** |
| **lpips+BN raw** | **0.0876** | **0.0140** | **19.24** | 0.5964 | **5.27** | — |
| **lpips+BN matched** | 0.0898 | 0.0152 | 18.91 | 0.5849 | **5.22** | **−0.05** |

### 개선 폭
- **Raw MACE**: 6.81 → 5.27 (**−23%**)
- **Raw↔Matched gap**: 1.30 → −0.05 (**완전 소멸**, matching이 이제 의미 거의 없음)
- **PSNR**: 18.28 → 19.24 (**+0.96 dB**)
- **SSIM**: 0.571 → 0.596 (작지만 유의)
- **MAE**: 0.101 → 0.088 (−13%)

### Reviewer 비판 무력화
원본 비판: "Histogram matching forces every generated image's R-mean toward training set mean. Therefore MACE = |train_mean − specimen_mean|."

- baseline 시대: matching이 1.30 MACE 보정 (비판 적중)
- 새 config: gap 0.05, raw 5.27 자체가 baseline matched 5.51보다 낮음 → matching이 칼리브레이션 헤드룸 못 만들어냄. 비판 정면 반박.

### Best epoch 위치 정상화
- Pure WGAN-GP + val MACE 선정: epoch 11-25 artifact 빈발
- 새 selection: best epoch 89 (lpips+BN). 정상적인 후반 학습 결과.

---

## 7. 다른 cGAN sensor combo / Architecture 후보 비교 (참고)

| 조합 | Raw MACE | SSIM | 결론 |
|---|---|---|---|
| 2ch + L1+SSIM (no LPIPS) | 5.84 | 0.601 | LPIPS 추가가 의미있음 |
| 2ch + LPIPS, no BN | 5.33 | 0.599 | BN 추가가 marginal 개선 |
| **2ch + LPIPS + BN** ← BEST | **5.27** | 0.596 | 표준 채택 |
| 2ch + LPIPS + dataset zscore | 5.96 | 0.600 | zscore는 magnitude 정보 손실로 후퇴 |
| 2ch + LPIPS + PatchGAN | 5.47 | 0.583 | architecture 복잡도 대비 이득 미미 |
| 4ch + LPIPS + BN | 6.20 | 0.604 | conditioning 늘려도 SSIM 안 깸, MACE 후퇴 |
| 4ch + LPIPS + dataset zscore | 5.88 | 0.605 | (위 동일) |

**결론**:
- **LPIPS는 필수** (기본 L1+SSIM 대비 MACE −0.57)
- **BN은 유의** (no-norm 대비 MACE −0.06, 일관성 확보)
- **PatchGAN은 불필요** (이득 미미)
- **4채널은 불필요** (오히려 후퇴, Phase 채널이 spatial info를 더하지 않음)

---

## 8. SSIM 0.60 Ceiling 관찰

7 pilot 모두 SSIM 0.58–0.60 영역에서 plateau. Loss/architecture/conditioning bandwidth/normalization 4가지 변경 다 시도했으나 못 깸. 데이터 본질의 한계 추정 (S-params 정보량 부족 또는 spot 패턴의 intrinsic randomness).

→ **Paper claim은 "정량 추정 (MACE/MAE) + 시각화 보조"**가 안전. "시각 inspection 대체"는 0.85+ 필요한데 현재 method로 도달 어려움.

---

## 9. Migration Checklist (다른 method 적용 시)

- [ ] `pip install pytorch_msssim lpips` (env에 설치)
- [ ] Generator/Decoder/UNet의 cond input에 `nn.BatchNorm1d(cond_dim)` 추가 (CLI flag로 toggle)
- [ ] Generator/Decoder loss에 L1 + (1−SSIM) + LPIPS 추가 (각 lambda CLI flag)
- [ ] 단일 채널 출력이라면 LPIPS 입력 시 `repeat(1, 3, 1, 1)`
- [ ] Validation에서 L1, SSIM, LPIPS, MACE 모두 logging
- [ ] Best-model 기준을 val L1 + 0.5·(1−SSIM) + 0.1·LPIPS 로 변경
- [ ] Checkpoint에 `cond_norm_type` 저장, inference에서 재현
- [ ] Pilot test: 1 trial × S11_S21 100 epoch → 결과가 cGAN best (MACE 5.27, SSIM 0.596) 근처 또는 더 좋은지 확인
- [ ] 통과 시 전체 5 trial × 15 sensor combo로 확장

---

## 코드 위치
- 학습 스크립트: `train_cgan.py` (모든 변경 적용된 reference)
- 모델: `cgan/models.py` (Generator, Critic, PatchCritic — 모두 `cond_norm_type` 지원)
- Pilot 스크립트: `run_cgan_pilot_lpips_2ch_bn.sh` (best config의 reference command)
- Best checkpoint: `checkpoints/baseline/cgan/trial1/20260505-211805_S11_S21_wgangp_l1ssim_perc_bn/best_model.pt`

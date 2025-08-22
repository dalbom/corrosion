#export CUDA_VISIBLE_DEVICES=0

python train.py \
  --csv    ./datasets/Corrosion_train.csv \
  --img_root ./datasets/corrosion_img \
  --use_channels S11 S21 Phase11 \
  --image_size 64 \
  --timesteps 1000 \
  --sampling_timesteps 250 \
  --batch_size 16 \
  --lr      8e-5 \
  --num_steps 1000000 \
  --log_every 10000 \
  --save_every 100000 
  # --use_mlp 256

# python inference.py \
#   --checkpoint logs/20250816-213838_S11_S21/model_step_400000.pt \
#   --csv datasets/Corrosion_test.csv \
#   --output output/S11_S21_400k \
#   --img_root datasets/corrosion_img

# python inference.py \
#   --checkpoint logs/20250816-213838_S11_S21/model_step_300000.pt \
#   --csv datasets/Corrosion_test.csv \
#   --output output/S11_S21_300k \
#   --img_root datasets/corrosion_img


# python inference.py \
#   --checkpoint logs/20250814-150102_S11/model_step_500000.pt \
#   --csv datasets/Corrosion_test.csv \
#   --output output/S11_500k \
#   --use_channels S11 \
#   --img_root datasets/corrosion_img

# python inference.py \
#   --checkpoint logs/20250814-150102_S11/model_step_400000.pt \
#   --csv datasets/Corrosion_test.csv \
#   --output output/S11_400k \
#   --use_channels S11 \
#   --img_root datasets/corrosion_img

# python inference.py \
#   --checkpoint logs/20250814-150102_S11/model_step_300000.pt \
#   --csv datasets/Corrosion_test.csv \
#   --output output/S11_300k \
#   --use_channels S11 \
#   --img_root datasets/corrosion_img

# python compare.py \
#   --csv datasets/Corrosion_test.csv \
#   --img_root datasets/corrosion_img \
#   --gen_root output/S11_S21_500k \
#   --out_csv output/S11_S21_500k_metrics.csv
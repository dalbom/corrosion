#export CUDA_VISIBLE_DEVICES=0

# python train.py \
#   --csv    ./datasets/Corrosion_train.csv \
#   --img_root ./datasets/corrosion_img \
#   --use_channels S11 S21 Phase11 \
#   --image_size 128 \
#   --timesteps 1000 \
#   --sampling_timesteps 250 \
#   --batch_size 8 \
#   --lr      8e-5 \
#   --num_steps 1000000 \
#   --log_every 10000 \
#   --save_every 100000 
  
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
#   --checkpoint logs_ext/20251016-222036_S11_S21_Ph11_real/model_step_1000000.pt \
#   --csv datasets/Corrosion_test.csv \
#   --output output/S11_S21_Ph11_1M_128 \
#   --use_channels S11 S21 Phase11 \
#   --image_size 128 \
#   --img_root datasets/corrosion_img

for dir in generated_cGAN/*/; do
  dir_name=$(basename "$dir")
  python evaluate_metrics.py \
    --csv datasets/Corrosion_test.csv \
    --img_root datasets/corrosion_img \
    --gen_root "$dir" \
    --out_csv "output/${dir_name}_cGAN_metrics.csv"
done

# python compare.py \
#   --csv datasets/Corrosion_test.csv \
#   --img_root datasets/corrosion_img \
#   --gen_root generated/S11_S21_Ph11_1M_128 \
#   --out_csv "output/S11_S21_Ph11_1M_128_metrics.csv"
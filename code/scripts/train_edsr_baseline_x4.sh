cd ../;
python train.py \
 --ckp_dir student_baseline/edsr_baseline/baseline_x4 \
 --scale 4 \
 --teacher [EDSR] \
 --model EDSR \
 --alpha 0 \
 --feature_loss_used 0 \
 --gpu_id 1 \
 --epochs 200 \
 --save_results \
 --chop \
 --patch_size 192
 
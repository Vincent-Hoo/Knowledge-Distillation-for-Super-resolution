cd ../;
python main.py \
 --ckp_dir student_baseline/edsr_baseline/baseline_x8 \
 --scale 8 \
 --teacher [EDSR] \
 --model EDSR \
 --alpha 0 \
 --feature_loss_used 0 \
 --gpu_id 4 \
 --epochs 200 \
 --save_results \
 --chop \
 --patch_size 384
 
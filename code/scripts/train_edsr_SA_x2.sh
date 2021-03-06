cd ../;
python train.py \
 --ckp_dir overall_distilation/edsr/SA_x2/ \
 --scale 2 \
 --teacher [EDSR] \
 --model EDSR \
 --alpha 0.5 \
 --feature_loss_used 1 \
 --feature_distilation_type 10*SA \
 --features [1,2,3] \
 --gpu_id 1 \
 --epochs 200 \
 --save_results \
 --chop \
 --patch_size 96
 
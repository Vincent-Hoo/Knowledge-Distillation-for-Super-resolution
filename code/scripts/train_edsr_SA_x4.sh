cd ../;
python main.py \
 --ckp_dir overall_distilation/edsr/SA_x4/ \
 --scale 4 \
 --teacher [EDSR] \
 --model EDSR \
 --alpha 0.5 \
 --feature_loss_used 1 \
 --feature_distilation_type 10*SA \
 --features [1,2,3] \
 --gpu_id 2 \
 --epochs 200 \
 --save_results \
 --chop \
 --patch_size 192
 
cd ../;
python train.py \
 --ckp_dir overall_distilation/rcan/SA_x4/ \
 --scale 4 \
 --teacher [RCAN] \
 --model RCAN \
 --alpha 0.5 \
 --feature_loss_used 1 \
 --feature_distilation_type 10*SA \
 --features [1,2,3] \
 --gpu_id 1 \
 --epochs 200 \
 --save_results \
 --chop \
 --patch_size 192
 
cd ../;
python main.py \
 --ckp_dir overall_distilation/rcan/SA_x3/ \
 --scale 3 \
 --teacher [RCAN] \
 --model RCAN \
 --alpha 0.5 \
 --feature_loss_used 1 \
 --feature_distilation_type 10*SA \
 --features [2,3] \
 --gpu_id 1 \
 --epochs 200 \
 --save_results \
 --chop \
 --patch_size 144
 
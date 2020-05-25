cd ../;
python main.py \
 --ckp_dir student_baseline/rcan_baseline/baseline_x3 \
 --scale 3 \
 --teacher [RCAN] \
 --model RCAN \
 --alpha 0 \
 --feature_loss_used 0 \
 --gpu_id 0 \
 --epochs 200 \
 --save_results \
 --chop \
 --patch_size 144
 
 
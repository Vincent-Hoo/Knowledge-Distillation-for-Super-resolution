cd ../;
python main.py \
 --ckp_dir student_baseline/san_baseline/baseline_x4 \
 --scale 4 \
 --teacher [SAN] \
 --model SAN \
 --alpha 0 \
 --feature_loss_used 0 \
 --gpu_id 5 \
 --epochs 200 \
 --save_results \
 --chop \
 --patch_size 192 \
 --data_test Set5 \
 

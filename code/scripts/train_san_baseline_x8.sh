cd ../;
python main.py \
 --ckp_dir student_baseline/san_baseline/baseline_x8 \
 --scale 8 \
 --teacher [SAN] \
 --model SAN \
 --alpha 0 \
 --feature_loss_used 0 \
 --gpu_id 5 \
 --epochs 200 \
 --save_results \
 --chop \
 --patch_size 384 \
 --data_test Set5+Set14+B100+Urban100 \
 

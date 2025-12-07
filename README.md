# ThermalTracker3.0

screen -S train
screen -r 234844

python main.py \
  --data_path ./data/GTOT \
  --output_dir output/overfit_test \
  --batch_size 2 \
  --epochs 50 \
  --lr_drop 40 \
  --lr 1e-4

python main.py \
  --data_path ./data/GTOT \
  --output_dir output/overfit_test_fix \
  --batch_size 2 \
  --epochs 50 \
  --lr_drop 40 \
  --lr 1e-4 \
  --lr_backbone 1e-4

python demo.py \
  --seq_name Tricycle \
  --checkpoint output/final_hope/checkpoint0049.pth \
  --output_dir output/vis_result


python main.py \
  --data_path ./data/GTOT \
  --output_dir output/final_sleep_run \
  --batch_size 2 \
  --epochs 400 \
  --lr_drop 200 \
  --lr 1e-4 \
  --lr_backbone 1e-5 \
  --num_queries 10 \
  --bbox_loss_coef 10 \
  --giou_loss_coef 5 \
  --seed 42
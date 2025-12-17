# ThermalTracker3.0

screen -S train
screen -r 234844?


python demo_twostream.py \
  --seq_name Tricycle \
  --checkpoint output/siamese_adapter_run/checkpoint0049.pth \
  --output_dir output/vis_result_siamese_adapter_run


python main.py \
   --data_path ./data/GTOT \
   --output_dir output/brute_force_run \
   --batch_size 4 \
   --epochs 50 \
   --lr_drop 40 \
   --lr 1e-4 \
   --lr_backbone 1e-4 \
   --num_queries 10 \
   --bbox_loss_coef 10 \
   --giou_loss_coef 5
#python main_finetune_MM_MIL.py \
#python -m torch.distributed.launch --nproc_per_node=2 --master_port=48798 main_finetune_MM_MIL.py \
python main_finetune_MM_MIL.py \
    --batch_size 8 \
    --world_size 4 \
    --epochs 50 \
    --lr 1e-2 \
    --nb_classes 7 \
    --data_path ./data \
    --task ./over_sampling_output/finetune_topcon_multi_mil_head4_ResNet50_oversampling_best_mAP_run5/ \
    --modality multi_mil_head4_base \
    --input_size 256 \
    

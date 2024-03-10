python main_finetune_MM_MIL.py \
    --eval \
    --batch_size 8 \
    --world_size 1 \
    --epochs 50 \
    --lr 1e-2 \
    --modality multi_mil_head4_base \
    --nb_classes 7 \
    --data_path ./data \
    --task ./over_sampling_output/finetune_topcon_multi_mil_head4_ResNet50_oversampling_best_mAP_run5/ \
    --resume ./over_sampling_output/finetune_topcon_multi_mil_head4_ResNet50_oversampling_best_mAP_run5/checkpoint-best.pth \
    --input_size 256 \
    
    
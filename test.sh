python main_finetune.py \
    --eval \
    --batch_size 2 \
    --world_size 1 \
    --model MM_MIL_vit_large_patch16 \
    --epochs 50 \
    --blr 5e-3 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.2 \
    --modality multi_mil_head4_base \
    --nb_classes 7 \
    --data_path ./data \
    --task ./over_sampling_output/finetune_topcon_multi_mil_head4_base_oversampling/ \
    --resume ./over_sampling_output/finetune_topcon_multi_mil_head4_base_oversampling/checkpoint-best.pth \
    --num_block 12 \
    --global_pool \
    #--input_size 224
    #--class_wise \
    #--freeze
    
    
    
    
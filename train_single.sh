#python main_finetune.py \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=48798 main_finetune.py \
    --batch_size 2 \
    --world_size 4 \
    --model MIL_vit_large_patch16 \
    --epochs 50 \
    --blr 5e-3 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.2 \
    --nb_classes 7 \
    --data_path ./data \
    --task ./multi_layer_fusion_output/finetune_topcon_OCT_MIL_Layer_MIL/ \
    --finetune ./pretrained_weights/RETFound_oct_weights.pth \
    --modality OCT_mil \
    --num_block 24 \
    --global_pool
    #--freeze 


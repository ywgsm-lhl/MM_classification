#python main_finetune.py \
python -m torch.distributed.launch --nproc_per_node=2 --master_port=48798 main_finetune.py \
    --batch_size 4 \
    --world_size 4 \
    --model MM_MIL_vit_large_patch16 \
    --epochs 50 \
    --blr 5e-3 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.2 \
    --nb_classes 7 \
    --data_path ./data \
    --task ./over_sampling_output/finetune_topcon_multi_mil_head4_base_oversampling_best_mAP/ \
    --finetune True \
    --finetune_fundus ./pretrained_weights/jx_vit_base_p16_224-80ecf9dd.pth \
    --finetune_OCT ./pretrained_weights/jx_vit_base_p16_224-80ecf9dd.pth \
    --modality multi_mil_head4_base \
    --num_block 12 \
    --global_pool \
    #--input_size 224
    #--wo_cls_token \
    #--freeze 
    #--global_pool

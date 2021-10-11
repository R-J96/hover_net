python run_infer.py \
--gpu='0,1' \
--nr_types=6 \
--type_info_path=type_info_panNuke.json \
--batch_size=32 \
--model_mode=fast \
--model_path=/home/robj/Projects/hover_net/lowerGU/imageNetPreTrained_2/01/net_epoch=50.tar \
--nr_inference_workers=8 \
--nr_post_proc_workers=16 \
wsi \
--input_dir=/data/TCGA/Bladder/WSIs/ \
--output_dir=/data/TCGA/Bladder/LowerGU_HoVer-Net_annotations_withMasks/ \
--input_mask_dir=/data/TCGA/Bladder/WSI_masks/ \
--save_thumb \

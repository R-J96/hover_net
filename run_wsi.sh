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
--input_dir=/data/UHCW/WSIs/UHCW_Bladder \
--output_dir=/data/UHCW/LowerGU_HoVer-Net_annotations/ \
--input_mask_dir= \
--save_thumb \
--save_mask

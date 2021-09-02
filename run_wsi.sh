python run_infer.py \
--gpu='0,1' \
--nr_types=6 \
--type_info_path=type_info.json \
--batch_size=32 \
--model_mode=fast \
--model_path=models/pretrained/hovernet_fast_pannuke_type_tf2pytorch.tar \
--nr_inference_workers=8 \
--nr_post_proc_workers=16 \
wsi \
--input_dir=/data/UHCW/WSIs/Test/ \
--output_dir=/data/Test/hoverNetUHCWTest/ \
--input_mask_dir= \
--save_thumb \
--save_mask

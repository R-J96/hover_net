python run_infer.py \
--gpu='0,1' \
--nr_types=6 \
--type_info_path=type_info_panNuke.json \
--batch_size=32 \
--model_mode=fast \
--model_path=/mnt/user-temp/rob-tia/projects/Bladder-experiments/models/net_epoch=50.tar \
--nr_inference_workers=8 \
--nr_post_proc_workers=16 \
wsi \
--input_dir=/mnt/available_datasets/Bladder/UHCW/WSI/ \
--output_dir=/data/TCGA/Bladder/LowerGU_HoVer-Net_annotations_withMasks/ \
--input_mask_dir=/mnt/user-temp/rob-tia/data/UHCW_bladder/masks/ \
--save_thumb \

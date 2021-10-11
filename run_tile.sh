python run_infer.py \
--gpu='0,1' \
--nr_types=6 \
--type_info_path=type_info_panNuke.json \
--batch_size=32 \
--model_mode=fast \
--model_path=/home/robj/Projects/hover_net/lowerGU/imageNetPreTrained_2/01/net_epoch=50.tar \
--nr_inference_workers=8 \
--nr_post_proc_workers=16 \
tile \
--input_dir=/data/UHCW/WSIs/WSI_patches/H09-16765_A1RIB_1_patches// \
--output_dir=/data/UHCW/WSI_patches_hoverNet_lowerGU/H09-16765_A1RIB_1_patches_qupath_hoverNet/ \
--mem_usage=0.2 \
--save_qupath
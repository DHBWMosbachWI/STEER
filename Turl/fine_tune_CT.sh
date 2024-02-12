 OUTPUT_DIR=model_v1_table_0.2_0.6_0.7_10000_1e-4_candnew_0_adam_addparams
#OUTPUT_DIR=model_v1_table_0.2_0.6_0.7_10000_1e-4_candnew_0_adam_no_visibility_test
MODE=0
CUDA_VISIBLE_DEVICES="1" python3 run_table_CT_finetuning.py \
    --output_dir=output/CT/v2/$MODE/$OUTPUT_DIR \
    --model_name_or_path=output/provided/pretrained \
    --model_type=CT \
    --do_train \
    --do_eval \
    --data_dir=data/wikitables_v2 \
    --evaluate_during_training \
    --per_gpu_train_batch_size=20 \
    --per_gpu_eval_batch_size=20 \
    --gradient_accumulation_steps=2 \
    --learning_rate=5e-5 \
    --num_train_epochs=10 \
    --save_total_limit=10 \
    --seed=1 \
    --cache_dir=cache \
    --overwrite_output_dir \
    --config_name=configs/table-base-config_v2.json \
    --save_steps=5000 \
    --logging_steps=1500 \
    --warmup_steps=5000 \
    --mode=$MODE 
    #--model_name_or_path=output/hybrid/v2/$OUTPUT_DIR \
    #--config_name=configs/table-base-config_v2.json \
    #--learning_rate=5e-5 \
    #> /dev/null 2>&1 &

OUTPUT_DIR=model_v1_table_0.2_0.6_0.7_10000_1e-4_candnew_0_adam
MODE=0
CUDA_VISIBLE_DEVICES="0" python3 run_table_RE_finetuning.py \
    --output_dir=output/RE/v2/$MODE/$OUTPUT_DIR \
    --model_name_or_path=output/provided/pretrained \
    --model_type=RE \
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
    --save_steps=500 \
    --logging_steps=250 \
    --mode=$MODE
    #--model_name_or_path=output/hybrid/v2/$OUTPUT_DIR \ 
    #> /dev/null 2>&1 &

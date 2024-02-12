#OUTPUT_DIR=model_v1_table_0.2_0.6_0.7_10000_1e-4_candnew_0_adam_no_visibility_test
MODE=1
labeled_data_size=1
#unlabeled_data_size=absolute
test_data_size=20.00
random_state=1
add_comment=""
#OUTPUT_DIR=model_STEER_1_absolute_20.0_2
OUTPUT_DIR="model_STEER_${labeled_data_size}_${unlabeled_data_size}_20.0_${random_state}_${add_comment}"
CUDA_VISIBLE_DEVICES="1" python3 run_table_CT_finetuning_STEER.py \
    --output_dir=output/CT/v2/$MODE/$OUTPUT_DIR \
    --model_name_or_path=output/provided/pretrained \
    --model_type=CT \
    --do_train \
    --data_dir=data/wikitables_v2 \
    --per_gpu_train_batch_size=5 \
    --per_gpu_eval_batch_size=1 \
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
    --mode=$MODE \
    --labeled_data_size=$labeled_data_size \
    --unlabeled_data_size=$unlabeled_data_size \
    --test_data_size=$test_data_size \
    --random_state=$random_state 
    #--add_STEER_train_data
labeled_data_size=2
#unlabeled_data_size=absolute
OUTPUT_DIR="model_STEER_${labeled_data_size}_${unlabeled_data_size}_20.00_${random_state}_${add_comment}"
CUDA_VISIBLE_DEVICES="1" python3 run_table_CT_finetuning_STEER.py \
    --output_dir=output/CT/v2/$MODE/$OUTPUT_DIR \
    --model_name_or_path=output/provided/pretrained \
    --model_type=CT \
    --do_train \
    --data_dir=data/wikitables_v2 \
    --per_gpu_train_batch_size=5 \
    --per_gpu_eval_batch_size=1 \
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
    --mode=$MODE \
    --labeled_data_size=$labeled_data_size \
    --unlabeled_data_size=$unlabeled_data_size \
    --test_data_size=$test_data_size \
    --random_state=$random_state 
    #--add_STEER_train_data
labeled_data_size=3
#unlabeled_data_size=65.26
OUTPUT_DIR="model_STEER_${labeled_data_size}_${unlabeled_data_size}_20.00_${random_state}_${add_comment}"
CUDA_VISIBLE_DEVICES="1" python3 run_table_CT_finetuning_STEER.py \
    --output_dir=output/CT/v2/$MODE/$OUTPUT_DIR \
    --model_name_or_path=output/provided/pretrained \
    --model_type=CT \
    --do_train \
    --data_dir=data/wikitables_v2 \
    --per_gpu_train_batch_size=5 \
    --per_gpu_eval_batch_size=1 \
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
    --mode=$MODE \
    --labeled_data_size=$labeled_data_size \
    --unlabeled_data_size=$unlabeled_data_size \
    --test_data_size=$test_data_size \
    --random_state=$random_state 
    #--add_STEER_train_data
labeled_data_size=4
#unlabeled_data_size=65.14
OUTPUT_DIR="model_STEER_${labeled_data_size}_${unlabeled_data_size}_20.00_${random_state}_${add_comment}"
CUDA_VISIBLE_DEVICES="1" python3 run_table_CT_finetuning_STEER.py \
    --output_dir=output/CT/v2/$MODE/$OUTPUT_DIR \
    --model_name_or_path=output/provided/pretrained \
    --model_type=CT \
    --do_train \
    --data_dir=data/wikitables_v2 \
    --per_gpu_train_batch_size=5 \
    --per_gpu_eval_batch_size=1 \
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
    --mode=$MODE \
    --labeled_data_size=$labeled_data_size \
    --unlabeled_data_size=$unlabeled_data_size \
    --test_data_size=$test_data_size \
    --random_state=$random_state 
    #--add_STEER_train_data
labeled_data_size=5
#unlabeled_data_size=65.11
OUTPUT_DIR="model_STEER_${labeled_data_size}_${unlabeled_data_size}_20.00_${random_state}_${add_comment}"
CUDA_VISIBLE_DEVICES="1" python3 run_table_CT_finetuning_STEER.py \
    --output_dir=output/CT/v2/$MODE/$OUTPUT_DIR \
    --model_name_or_path=output/provided/pretrained \
    --model_type=CT \
    --do_train \
    --data_dir=data/wikitables_v2 \
    --per_gpu_train_batch_size=5 \
    --per_gpu_eval_batch_size=1 \
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
    --mode=$MODE \
    --labeled_data_size=$labeled_data_size \
    --unlabeled_data_size=$unlabeled_data_size \
    --test_data_size=$test_data_size \
    --random_state=$random_state 
    #--add_STEER_train_data

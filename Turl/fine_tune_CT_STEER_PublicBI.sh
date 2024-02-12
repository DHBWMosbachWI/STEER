#OUTPUT_DIR=model_v1_table_0.2_0.6_0.7_10000_1e-4_candnew_0_adam_no_visibility_test
MODE=1
labeled_data_size=43.19
#unlabeled_data_size=absolute
unlabeled_data_size=36.80
test_data_size=20.01
random_state=2
add_comment="PublicBI_sameTraindata"
#OUTPUT_DIR=model_STEER_1_absolute_20.0_2
OUTPUT_DIR="model_STEER_${labeled_data_size}_${unlabeled_data_size}_20.0_${random_state}_${add_comment}"
CUDA_VISIBLE_DEVICES="1" python3 run_table_CT_finetuning_STEER_PublicBI.py \
    --output_dir=output/CT/v2/$MODE/$OUTPUT_DIR \
    --model_name_or_path=output/provided/pretrained \
    --model_type=CT \
    --do_train \
    --data_dir=data/public_bi \
    --per_gpu_train_batch_size=1 \
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
labeled_data_size=50.84
unlabeled_data_size=29.14
OUTPUT_DIR="model_STEER_${labeled_data_size}_${unlabeled_data_size}_20.0_${random_state}_${add_comment}"
CUDA_VISIBLE_DEVICES="1" python3 run_table_CT_finetuning_STEER_PublicBI.py \
    --output_dir=output/CT/v2/$MODE/$OUTPUT_DIR \
    --model_name_or_path=output/provided/pretrained \
    --model_type=CT \
    --do_train \
    --data_dir=data/public_bi \
    --per_gpu_train_batch_size=1 \
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
labeled_data_size=55.06
unlabeled_data_size=24.93
OUTPUT_DIR="model_STEER_${labeled_data_size}_${unlabeled_data_size}_20.0_${random_state}_${add_comment}"
CUDA_VISIBLE_DEVICES="1" python3 run_table_CT_finetuning_STEER_PublicBI.py \
    --output_dir=output/CT/v2/$MODE/$OUTPUT_DIR \
    --model_name_or_path=output/provided/pretrained \
    --model_type=CT \
    --do_train \
    --data_dir=data/public_bi \
    --per_gpu_train_batch_size=1 \
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
labeled_data_size=59.27
unlabeled_data_size=20.72
OUTPUT_DIR="model_STEER_${labeled_data_size}_${unlabeled_data_size}_20.0_${random_state}_${add_comment}"
CUDA_VISIBLE_DEVICES="1" python3 run_table_CT_finetuning_STEER_PublicBI.py \
    --output_dir=output/CT/v2/$MODE/$OUTPUT_DIR \
    --model_name_or_path=output/provided/pretrained \
    --model_type=CT \
    --do_train \
    --data_dir=data/public_bi \
    --per_gpu_train_batch_size=1 \
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
labeled_data_size=61.24
unlabeled_data_size=18.75
OUTPUT_DIR="model_STEER_${labeled_data_size}_${unlabeled_data_size}_20.0_${random_state}_${add_comment}"
CUDA_VISIBLE_DEVICES="1" python3 run_table_CT_finetuning_STEER_PublicBI.py \
    --output_dir=output/CT/v2/$MODE/$OUTPUT_DIR \
    --model_name_or_path=output/provided/pretrained \
    --model_type=CT \
    --do_train \
    --data_dir=data/public_bi \
    --per_gpu_train_batch_size=1 \
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
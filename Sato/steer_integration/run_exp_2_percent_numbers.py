import numpy as np
import os
import sys

# set env-var
os.environ['BASEPATH'] = '/home/anonymous/sato'
os.environ['RAW_DIR'] = '/mnt/batch/tasks/shared/LS_root/mounts/clusters/anonymousmlc-ds12-v2/' # path to the raw data
os.environ['SHERLOCKPATH'] = os.environ['BASEPATH']+'\\sherlock'
os.environ['EXTRACTPATH'] = os.environ['BASEPATH']+'\\extract'
#os.environ['PYTHONPATH'] = os.environ['PYTHONPATH']+':'+os.environ['SHERLOCKPATH']
#os.environ['PYTHONPATH'] = os.environ['PYTHONPATH']+':'+os.environ['BASEPATH']
os.environ['TYPENAME'] = 'type_public_bi'

sys.path.append("..")
sys.path.append(os.environ['BASEPATH'])

corpus = "public_bi_num"

random_state = 2

if __name__=="__main__":

    #for index,percent in enumerate(np.arange(5,55,5)):
    for index, percent in enumerate([79.9]):
        # if index != 0:
        #     continue
        comment = f"labeled{percent}_unlabeled{100-20.0-percent:.1f}_test{20.0}_{random_state}"
        column_level_split_file_path = f"/home/anonymous/semantic_data_lake/data/extract/out/labeled_unlabeled_test_split/{corpus}_{percent}_{100-20.0-percent:.1f}_20.0_{random_state}.json"
        pretrained_shelock_path = f"sherlock_retrain_labeled{percent}_unlabeled{100-20.0-percent:.1f}_test{20.0}_{random_state}.pt"
        if os.environ['TYPENAME'] == "type78":
            pretrained_CRF_LDA_path = f"CRF+LDA_retrain_labeled{percent}_unlabeled{100-20.0-percent:.1f}_test{20.0}_{random_state}.pt"
        else:
            pretrained_CRF_LDA_path = f"CRF+LDA_retrain_labeled{percent}_unlabeled{100-20.0-percent:.1f}_test{20.0}_{random_state}_train100.pt"

        # retrain sherlock
        os.system(
            f"python3 ../model/train_sherlock.py -c ../model/params/{corpus}/sherlock_retrain.txt --comment {comment} --column_level_split_file_path {column_level_split_file_path}"
        )

        # retrain/train sato (no retrain for turl dataset)
        os.system(
            f"python3 ../model/train_CRF_LC.py -c ../model/params/{corpus}/CRF+LDA_retrain.txt --pre_trained_sherlock_path {pretrained_shelock_path} --comment {comment} --column_level_split_file_path {column_level_split_file_path}"
        )

        # validate sato model
        os.system(
            f"python3 ../model/train_CRF_LC.py -c ../model/params/{corpus}/CRF+LDA_eval.txt --model_list {pretrained_CRF_LDA_path} --comment eval_{comment} --column_level_split_file_path {column_level_split_file_path}"
        )
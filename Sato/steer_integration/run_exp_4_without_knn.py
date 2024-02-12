import numpy as np
import os
import sys

# set env-var
os.environ['BASEPATH'] = '/home/sanonymous/sato'
# path to the raw data
os.environ['RAW_DIR'] = '/mnt/batch/tasks/shared/LS_root/mounts/clusters/anonymousmlc-ds12-v2/code/Users/svenanonymous/viznet-master/raw'
os.environ['SHERLOCKPATH'] = os.environ['BASEPATH']+'\\sherlock'
os.environ['EXTRACTPATH'] = os.environ['BASEPATH']+'\\extract'
#os.environ['PYTHONPATH'] = os.environ['PYTHONPATH']+':'+os.environ['SHERLOCKPATH']
#os.environ['PYTHONPATH'] = os.environ['PYTHONPATH']+':'+os.environ['BASEPATH']
os.environ['TYPENAME'] = 'type_public_bi'

sys.path.append("..")
sys.path.append(os.environ['BASEPATH'])

test_data_size = 20.0

corpus = "public_bi_num"


if __name__ == "__main__":

    # for index,percent in enumerate(np.arange(5,55,5)):
    for index, percent in enumerate([1,2,3,4,5]):
        for distance_threshold in [1e-2]:
            for random_state in [1,2,3,4,5]:
                comment = f"embclus_without_knn_{distance_threshold}_labeled{percent}_unlabeledAbsolute_test{test_data_size}_{random_state}_both_GoogleUSEv3"
                column_level_split_file_path = f"/home/sanonymous/semantic_data_lake/data/extract/out/labeled_unlabeled_test_split/{corpus}_{percent}_absolute_{test_data_size}_{random_state}.json"
                #embclus_gen_train_data_path = f"D:\\semantic_data_lake\\semantic_data_lake\\emb_clus\\knn_classifier\\out\\gen_training_data\\{corpus}_gen_training_data_1_{distance_threshold}_{percent}_absolute_{test_data_size}.csv"
                embclus_gen_train_data_path = f"/home/sanonymous/semantic_data_lake/emb_clus/without_knn/out/gen_training_data/{corpus}_gen_training_data_{distance_threshold}_{percent}_absolute_{20.0}_{random_state}_both_GoogleUSEv3.csv"
                pretrained_shelock_path = f"sherlock_retrain_embclus_without_knn_{distance_threshold}_labeled{percent}_unlabeledAbsolute_test{test_data_size}_{random_state}_both_GoogleUSEv3.pt"
                
                if os.environ["TYPENAME"] == "type78":
                    pretrained_CRF_LDA_path = f"CRF+LDA_retrain_embclus_without_knn_{distance_threshold}_labeled{percent}_unlabeledAbsolute_test{test_data_size}_{random_state}_both_GoogleUSEv3.pt"
                else:
                    pretrained_CRF_LDA_path = f"CRF+LDA_retrain_embclus_without_knn_{distance_threshold}_labeled{percent}_unlabeledAbsolute_test{test_data_size}_{random_state}_both_GoogleUSEv3_train100.pt"
                # retrain sherlock
                os.system(
                    f"python3 ../model/train_sherlock.py -c ../model/params/{corpus}/sherlock_retrain.txt --comment {comment} --column_level_split_file_path {column_level_split_file_path} --embclus_gen_train_data_path {embclus_gen_train_data_path}"
                )

                # retrain sato
                os.system(
                    f"python3 ../model/train_CRF_LC.py -c ../model/params/{corpus}/CRF+LDA_retrain.txt --pre_trained_sherlock_path {pretrained_shelock_path} --comment {comment} --column_level_split_file_path {column_level_split_file_path} --embclus_gen_train_data_path {embclus_gen_train_data_path}"
                )

                # validate sato
                os.system(
                    f"python3 ../model/train_CRF_LC.py -c ../model/params/{corpus}/CRF+LDA_eval.txt --model_list {pretrained_CRF_LDA_path} --comment eval_{comment} --column_level_split_file_path {column_level_split_file_path}"
                )

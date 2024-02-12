import numpy as np
import os
import sys

# set env-var
os.environ['BASEPATH'] = 'D:\\STEER\\Sato'
os.environ['RAW_DIR'] = '/mnt/batch/tasks/shared/LS_root/mounts/clusters/anonymousmlc-ds12-v2/code/' # path to the raw data
os.environ['SHERLOCKPATH'] = os.environ['BASEPATH']+'\\sherlock'
os.environ['EXTRACTPATH'] = os.environ['BASEPATH']+'\\extract'
#os.environ['PYTHONPATH'] = os.environ['PYTHONPATH']+':'+os.environ['SHERLOCKPATH']
#os.environ['PYTHONPATH'] = os.environ['PYTHONPATH']+':'+os.environ['BASEPATH']
os.environ['TYPENAME'] = 'type78'

sys.path.append("..")
sys.path.append(os.environ['BASEPATH'])

corpus = "public_bi"

if __name__=="__main__":
    for random_state in [1,2,3,4,5]:
        comment = f"sato_baseline_unlabeledAbsolute_test{20.0}_{random_state}"
        column_level_split_file_path = f"D:\\STEER\\data\\extract\\out\\labeled_unlabeled_test_split\\{corpus}_1_absolute_20.0_{random_state}.json"

        # validate sato model
        os.system(
            f"python ../model/train_CRF_LC.py -c ../model/params/public_bi/CRF+LDA_eval.txt --model_list model.pt --comment eval_{comment} --column_level_split_file_path {column_level_split_file_path}"
        )
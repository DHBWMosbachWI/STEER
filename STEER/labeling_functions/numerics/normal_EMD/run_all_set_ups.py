import numpy as np
import os
from dotenv import load_dotenv
load_dotenv(override=True)

corpus = "public_bi_num"
gen_train_data = True
threshold_EMD_factor = 0.01
absolute = True

if __name__ == "__main__":
    if absolute:
        for labeled_data_size in [1, 2, 3, 4, 5]:
            for random_state in [1, 2, 3, 4, 5]:
                print(labeled_data_size)
                if gen_train_data:
                    os.system(
                        f"{os.environ['PYTHON']} run_normal_EMD.py --labeled_data_size {labeled_data_size} --threshold_EMD_factor {threshold_EMD_factor} --absolute_numbers {True} --corpus {corpus} --gen_train_data {True} --random_state {random_state} --n_worker 1"
                    )
    #             else:
    #                 os.system(
    #                     f"{os.environ['PYTHON']} combine_LFs_labels.py --labeled_data_size {labeled_data_size} --absolute_numbers {True} --corpus {corpus} --snorkel_label_model {snorkel_label_model} --random_state {random_state}"
    #                 )
    # else:
    #     for labeled_data_size in np.around(np.arange(0.2, 2.2, 0.2), 2):
    #         for random_state in [2]:
    #             print(labeled_data_size)
    #             if gen_train_data:
    #                 os.system(
    #                     f"{os.environ['PYTHON']} combine_LFs_labels.py --labeled_data_size {labeled_data_size} --unlabeled_data_size {100.0-20.0-labeled_data_size} --corpus {corpus} --snorkel_label_model {snorkel_label_model} --gen_train_data {True} --random_state {random_state}"
    #                 )
    #             else:
    #                 os.system(
    #                     f"{os.environ['PYTHON']} combine_LFs_labels.py --labeled_data_size {labeled_data_size} --unlabeled_data_size {100.0-20.0-labeled_data_size} --corpus {corpus} --snorkel_label_model {snorkel_label_model} --random_state {random_state}"
    #                 )

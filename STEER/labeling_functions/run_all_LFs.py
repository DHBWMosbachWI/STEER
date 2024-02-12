import os
import numpy as np
from dotenv import load_dotenv
load_dotenv(override=True)

corpus = "public_bi_num"
gen_train_data = True
absolute = True


if __name__ == "__main__":
    # load all types for the differnet LFs
    if corpus == "turl":
        check_elements_types = [
            "american_football.football_team",
            "automotive.model",
            "baseball.baseball_team",
            "film.film_genre",
            "ice_hockey.hockey_team",
            "location.us_county",
            "location.us_state",
            "music.genre",
            "soccer.football_team",
            "soccer.football_player",
            "sports.sports_league",
        ]
        regex_elements_in_col = [
            "aviation.aircraft_model", "internet.website",
            "award.award_category", "film.director",
            "american_football.football_player", "boats.ship_class",
            "cricket.cricket_player", "military.military_unit"
        ]
    elif corpus == "public_bi":
        #check_elements_types = ["gender", "language"]
        check_elements_types = []
        #regex_elements_in_col = ["description", "name"]
        regex_elements_in_col = []
    elif corpus == "public_bi_num":
        check_elements_types = []
        regex_elements_in_col = []
        
    if absolute:
        for labeled_data_size in [1,2,3,4,5]:
            for random_state in [1,2,3,4,5]:
                if gen_train_data:
                    os.system(
                        f"{os.environ['PYTHON']} header_to_sem_type_sim/header_to_sem_type_sim.py --corpus {corpus} --gen_train_data {True} --labeled_data_size {labeled_data_size} --absolute_number {True} --n_worker 20 --random_state {random_state}"
                    )
                else:
                    os.system(
                        f"{os.environ['PYTHON']} header_to_sem_type_sim/header_to_sem_type_sim.py --corpus {corpus} --labeled_data_size {labeled_data_size} --absolute_number {True} --random_state {random_state}"
                    )
                    
                for check_element in check_elements_types:
                    if gen_train_data:
                        os.system(
                            f"{os.environ['PYTHON']} check_elements_in_col/run_check_elements_in_col.py -c check_elements_in_col/params/{corpus}/{check_element}.txt --labeled_data_size {labeled_data_size} --absolute_number {True} --corpus {corpus} --gen_train_data {True} --n_worker 20 --random_state {random_state}"
                        )
                    else:
                        os.system(
                            f"{os.environ['PYTHON']} check_elements_in_col/run_check_elements_in_col.py -c check_elements_in_col/params/{corpus}/{check_element}.txt --labeled_data_size {labeled_data_size} --absolute_number {True} --corpus {corpus} --random_state {random_state}"
                        )
                        
                for check_element in regex_elements_in_col:
                    if gen_train_data:
                        os.system(
                            f"{os.environ['PYTHON']} regex_elements_in_col/run_regex_elements_in_col.py -c regex_elements_in_col/params/{corpus}/{check_element}.txt --labeled_data_size {labeled_data_size} --absolute_number {True} --corpus {corpus} --gen_train_data {True} --n_worker 20 --random_state {random_state}"
                        )
                    else:
                        os.system(
                            f"{os.environ['PYTHON']} regex_elements_in_col/run_regex_elements_in_col.py -c regex_elements_in_col/params/{corpus}/{check_element}.txt --labeled_data_size {labeled_data_size} --absolute_number {True} --corpus {corpus} --random_state {random_state}"
                        )
    else:
        for labeled_data_size in np.around(np.arange(0.2, 2.2, 0.2), 2):
            for random_state in [2]:
                if gen_train_data:
                    os.system(
                        f"{os.environ['PYTHON']} header_to_sem_type_sim/header_to_sem_type_sim.py --corpus {corpus} --gen_train_data {True} --labeled_data_size {labeled_data_size} --unlabeled_data_size {100.0-20.0-labeled_data_size} --n_worker 20 --random_state {random_state}"
                    )
                else:
                    os.system(
                        f"{os.environ['PYTHON']} header_to_sem_type_sim/header_to_sem_type_sim.py --corpus {corpus} --labeled_data_size {labeled_data_size} --unlabeled_data_size {100.0-20.0-labeled_data_size} --random_state {random_state}"
                    )
                    
                for check_element in check_elements_types:
                    if gen_train_data:
                        os.system(
                            f"{os.environ['PYTHON']} check_elements_in_col/run_check_elements_in_col.py -c check_elements_in_col/params/{corpus}/{check_element}.txt --labeled_data_size {labeled_data_size} --unlabeled_data_size {100.0-20.0-labeled_data_size} --corpus {corpus} --gen_train_data {True} --n_worker 20 --random_state {random_state}"
                        )
                    else:
                        os.system(
                            f"{os.environ['PYTHON']} check_elements_in_col/run_check_elements_in_col.py -c check_elements_in_col/params/{corpus}/{check_element}.txt --labeled_data_size {labeled_data_size} --unlabeled_data_size {100.0-20.0-labeled_data_size} --corpus {corpus} --random_state {random_state}"
                        )
                        
                for check_element in regex_elements_in_col:
                    if gen_train_data:
                        os.system(
                            f"{os.environ['PYTHON']} regex_elements_in_col/run_regex_elements_in_col.py -c regex_elements_in_col/params/{corpus}/{check_element}.txt --labeled_data_size {labeled_data_size} --unlabeled_data_size {100.0-20.0-labeled_data_size} --corpus {corpus} --gen_train_data {True} --n_worker 20 --random_state {random_state}"
                        )
                    else:
                        os.system(
                            f"{os.environ['PYTHON']} regex_elements_in_col/run_regex_elements_in_col.py -c regex_elements_in_col/params/{corpus}/{check_element}.txt --labeled_data_size {labeled_data_size} --unlabeled_data_size {100.0-20.0-labeled_data_size} --corpus {corpus} --random_state {random_state}"
                        )
                            

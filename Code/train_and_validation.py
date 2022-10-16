import warnings
warnings.filterwarnings("ignore")

import os, sys, time
#os.chdir(os.path.dirname(os.path.realpath(__file__)))

import utils_eval
import pickle 

from class_SurvivalQuilts import SurvivalQuilts

def train_the_model(MODE, K, num_outer):

    time_stamp = time.strftime("%Y%m%d%H")

    if MODE == 11:
        label_mode = "Vital status"
        filter_conds = "NULL"
        train_dataset_path = "Dataset/SEER_valid_train.csv"
        test_dataset_path = "Dataset/SEER_valid_test.csv"
        test_ncc_dataset_path = "Dataset/NCC_valid.csv"
        model_label = "QS_VS_%s" % (time_stamp)
    elif MODE == 12:
        label_mode = "Vital status"
        filter_conds = "Neoadjuvant therapy=2"
        train_dataset_path = "Dataset/SEER_valid_NeoadjuvantTherapy_2_train.csv"
        test_dataset_path = "Dataset/SEER_valid_NeoadjuvantTherapy_2_test.csv"
        test_ncc_dataset_path = "Dataset/NCC_valid_NeoadjuvantTherapy_2.csv"
        model_label = "QS_VS_NT2_%s" % (time_stamp)
    elif MODE == 21:
        label_mode = "Cancer specific death"
        filter_conds = "NULL"
        train_dataset_path = "Dataset/SEER_valid_train.csv"
        test_dataset_path = "Dataset/SEER_valid_test.csv"
        model_label = "QS_CSD_%s" % (time_stamp)
    elif MODE == 22:
        label_mode = "Cancer specific death"
        filter_conds = "Neoadjuvant therapy=2"
        train_dataset_path = "Dataset/SEER_valid_NeoadjuvantTherapy_2_train.csv"
        test_dataset_path = "Dataset/SEER_valid_NeoadjuvantTherapy_2_test.csv"
        model_label = "QS_CSD_NT2_%s" % (time_stamp)


    # Load data
    tr_X, tr_Y, tr_T = utils_eval.load_csv_dataset(train_dataset_path, label_mode)
    te_X, te_Y, te_T = utils_eval.load_csv_dataset(test_dataset_path, label_mode)
    if MODE in [11, 12]:
        te_ncc_X, te_ncc_Y, te_ncc_T = utils_eval.load_csv_dataset(test_ncc_dataset_path, label_mode)

    # Eval time 
    eval_time_horizons = [6, 12, 24, 36, 60, 120]
    eval_time_ratios = [utils_eval.get_eval_time_ratio(eval_time_horizon, tr_T[tr_Y.iloc[:,0] == 1]) \
                            for eval_time_horizon in eval_time_horizons]

    # Train
    model_sq = SurvivalQuilts(K = K, num_outer = num_outer)
    model_sq.train(tr_X, tr_T, tr_Y)

    # Save model
    try:
        filename = 'Results/Models/%s_%d_%d.sav' \
            % (model_label, model_sq.num_outer, model_sq.K)
        pickle.dump(model_sq, open(filename, 'wb'))
        print("Save the model to %s" % filename)
    except Exception as e:
        print("Save failed! Exception: %s" % e)

    # Validation
    pred = model_sq.predict(te_X, eval_time_horizons)
    for e_idx, eval_time in enumerate(eval_time_horizons):
        c_index, brier_score = utils_eval.calc_metrics(tr_T, tr_Y, te_T, te_Y, pred[:, e_idx], eval_time)
        print("Eval_time: %d (%.3f%%) | c_index = %.10f | brier_score = %.10f" % \
            (eval_time, 100 * eval_time_ratios[e_idx], c_index, brier_score))

    if MODE in [11, 12]:
        pred = model_sq.predict(te_ncc_X, eval_time_horizons)
        for e_idx, eval_time in enumerate(eval_time_horizons):
            c_index, brier_score = utils_eval.calc_metrics(tr_T, tr_Y, te_ncc_T, te_ncc_Y, pred[:, e_idx], eval_time)
            print("Eval_time: %d (%.3f%%) | c_index = %.10f | brier_score = %.10f" % \
                (eval_time, 100 * eval_time_ratios[e_idx], c_index, brier_score))

    print("Finished!")

def validation_the_model(model_name):
    
    MODE = 11 if "VS" in model_name else 21
    if "NT2" in model_name:
        MODE = MODE + 1
    if MODE == 11:
        label_mode = "Vital status"
        filter_conds = "NULL"
        train_dataset_path = "Dataset/SEER_valid_train.csv"
        test_dataset_path = "Dataset/SEER_valid_test.csv"
        test_ncc_dataset_path = "Dataset/NCC_valid.csv"
    elif MODE == 12:
        label_mode = "Vital status"
        filter_conds = "Neoadjuvant therapy=2"
        train_dataset_path = "Dataset/SEER_valid_NeoadjuvantTherapy_2_train.csv"
        test_dataset_path = "Dataset/SEER_valid_NeoadjuvantTherapy_2_test.csv"
        test_ncc_dataset_path = "Dataset/NCC_valid_NeoadjuvantTherapy_2.csv"
    elif MODE == 21:
        label_mode = "Cancer specific death"
        filter_conds = "NULL"
        train_dataset_path = "Dataset/SEER_valid_train.csv"
        test_dataset_path = "Dataset/SEER_valid_test.csv"
    elif MODE == 22:
        label_mode = "Cancer specific death"
        filter_conds = "Neoadjuvant therapy=2"
        train_dataset_path = "Dataset/SEER_valid_NeoadjuvantTherapy_2_train.csv"
        test_dataset_path = "Dataset/SEER_valid_NeoadjuvantTherapy_2_test.csv"

    print("=============Start validation of %s============="% (model_name))

    # Load model
    model_dir_path = "Results/Models/"
    model_path = model_dir_path + model_name
    model_sq = pickle.load(open(model_path, 'rb'))
    print("-------------Model Parameters-------------")
    print("Time-horizons for temporal quilting: K = %d" % (model_sq.K))
    print("BO iteration: num_bo = %d" % (model_sq.num_bo))
    print("Maximum number of BO: num_outer = %d" % (model_sq.num_outer))
    print("Quilting patterns: W = ")
    print(model_sq.quilting_patterns)


    ## Load data
    print("-------------Load Datasets-------------")
    tr_X, tr_Y, tr_T = utils_eval.load_csv_dataset(train_dataset_path, label_mode)
    te_X, te_Y, te_T = utils_eval.load_csv_dataset(test_dataset_path, label_mode)
    if MODE in [11, 12]:
        te_ncc_X, te_ncc_Y, te_ncc_T = utils_eval.load_csv_dataset(test_ncc_dataset_path, label_mode)

    # Eval time 
    eval_time_horizons = [6, 12, 24, 36, 60, 120]

    # Validation
    ## SEER Train
    print("-------------Validation dataset: %s-------------" % train_dataset_path)
    pred = model_sq.predict(tr_X, eval_time_horizons)
    for e_idx, eval_time in enumerate(eval_time_horizons):
        c_index, _ = utils_eval.calc_metrics(tr_T, tr_Y, tr_T, tr_Y, pred[:, e_idx], eval_time)
        print("Eval_time: %d | c_index = %.10f." % (eval_time, c_index))

    ## SEER Test
    print("-------------Validation dataset: %s-------------" % test_dataset_path)
    pred = model_sq.predict(te_X, eval_time_horizons)
    for e_idx, eval_time in enumerate(eval_time_horizons):
        c_index, _ = utils_eval.calc_metrics(tr_T, tr_Y, te_T, te_Y, pred[:, e_idx], eval_time)
        print("Eval_time: %d | c_index = %.10f." % (eval_time, c_index))
    
    ## NCC
    if MODE in [11, 12]:
        print("-------------Validation dataset: %s-------------" % test_ncc_dataset_path)
        pred = model_sq.predict(te_ncc_X, eval_time_horizons)
        for e_idx, eval_time in enumerate(eval_time_horizons):
            c_index, _ = utils_eval.calc_metrics(tr_T, tr_Y, te_ncc_T, te_ncc_Y, pred[:, e_idx], eval_time)
            print("Eval_time: %d | c_index = %.10f." % (eval_time, c_index))
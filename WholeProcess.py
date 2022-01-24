# new file to figure out the six stages best structure
import random
import pandas as pd
import train
import train2
import inference
import inference2


def proposed_model(Structure_H2Y, Structure_X2Y, is_noise):
    # reset_config()  # h2y的配置
    hidden1, hidden2, hidden3, hidden4 = Structure_H2Y
    Hidden1, Hidden2, Hidden3, Hidden4 = Structure_X2Y
    hidden_feat_len, out_feat_len, n_feature_len = 53, 1, 13
    train.main_func_train_val(hidden1, hidden2, hidden3, hidden4, hidden_feat_len, out_feat_len, Hidden1,
                              Hidden2, Hidden3, Hidden4, n_feature_len, is_noise)
    proposed_model_val_error = train.main_func_train_val(hidden1, hidden2, hidden3, hidden4, hidden_feat_len,
                                                         out_feat_len, Hidden1,
                                                         Hidden2, Hidden3, Hidden4, n_feature_len, is_noise, H_model=0)
    proposed_model_test_error = inference.main_func_test(hidden1, hidden2, hidden3, hidden4, hidden_feat_len,
                                                         out_feat_len, Hidden1,
                                                         Hidden2, Hidden3, Hidden4, n_feature_len, is_noise, H_model=0)
    return proposed_model_val_error, proposed_model_test_error


def NormalDNN(Structure_H2Y, Structure_X2Y, is_noise):
    # reset_config()  # h2y的配置
    hidden1, hidden2, hidden3, hidden4 = Structure_H2Y
    Hidden1, Hidden2, Hidden3, Hidden4 = Structure_X2Y
    hidden_feat_len, out_feat_len, n_feature_len = 53, 1, 13
    NormalDNN_val_error = train.main_func_train_val(hidden1, hidden2, hidden3, hidden4, hidden_feat_len, out_feat_len,
                                                    Hidden1,
                                                    Hidden2, Hidden3, Hidden4, n_feature_len, is_noise, H_model=0,
                                                    add_physical_info=0)
    NormalDNN_test_error = inference.main_func_test(hidden1, hidden2, hidden3, hidden4, hidden_feat_len, out_feat_len,
                                                    Hidden1,
                                                    Hidden2, Hidden3, Hidden4, n_feature_len, is_noise, H_model=0,
                                                    add_physical_info=0)
    return NormalDNN_val_error, NormalDNN_test_error


def ExtraDNN(Structure_H2Y, Structure_X2Y, is_noise):
    hidden1, hidden2, hidden3, hidden4 = Structure_H2Y
    Hidden1, Hidden2, Hidden3, Hidden4 = Structure_X2Y
    hidden_feat_len, out_feat_len, n_feature_len = 53, 1, 66
    NormalDNN_val_error = train2.main_func_train_val(hidden1, hidden2, hidden3, hidden4, hidden_feat_len, out_feat_len,
                                                     Hidden1,
                                                     Hidden2, Hidden3, Hidden4, n_feature_len, is_noise, H_model=0,
                                                     add_physical_info=0)
    NormalDNN_test_error = inference2.main_func_test(hidden1, hidden2, hidden3, hidden4, hidden_feat_len, out_feat_len,
                                                     Hidden1,
                                                     Hidden2, Hidden3, Hidden4, n_feature_len, is_noise, H_model=0,
                                                     add_physical_info=0)
    return NormalDNN_val_error, NormalDNN_test_error


if __name__ == '__main__':
    error = {"StructureSingleH2Y": [], "StructureSingleX2Y": [], "noise": [], "proposed_model_error": [], "Normal_DNN_error": [], "Extra_DNN_error": []}
    for j in range(8):
        is_noise = [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]
        StructureH2Y = [[80, 40, 40, 20]]
        StructureX2Y = [[80, 40, 40, 20]]
        StructureSingleH2Y = StructureH2Y[0]
        StructureSingleX2Y = StructureX2Y[0]
        error["StructureSingleH2Y"].append(StructureSingleH2Y)
        error["StructureSingleX2Y"].append(StructureSingleX2Y)
        error["noise"].append(is_noise[j])
        proposed_model_error = proposed_model(StructureSingleH2Y, StructureSingleX2Y, is_noise[j])
        Normal_DNN_error = NormalDNN(StructureSingleH2Y, StructureSingleX2Y, is_noise[j])
        Extra_DNN_error = ExtraDNN(StructureSingleH2Y, StructureSingleX2Y, is_noise[j])
        error["proposed_model_error"].append(proposed_model_error)
        error["Normal_DNN_error"].append(Normal_DNN_error)
        error["Extra_DNN_error"].append(Extra_DNN_error)
        print(error)
    df = pd.DataFrame(error,
                      columns=['StructureSingleH2Y', 'StructureSingleX2Y', 'noise', "proposed_model_error",
                               "Normal_DNN_error",
                               "Extra_DNN_error"])
    df.to_csv("ErrorDifferentStructure.csv")

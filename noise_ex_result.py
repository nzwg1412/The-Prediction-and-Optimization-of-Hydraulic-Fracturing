# RAE =\frac{|\mathit{sMSE}^\ast -\mathit{sMSE}|}{\mathit{sMSE}}
import pandas as pd
import re

data = pd.read_csv('ErrorDifferentStructure.csv', index_col=0)
proposed_model_error_new = []
Normal_DNN_error_new = []
Extra_DNN_error_new = []
for i in range(8):
    proposed_model_error = data["proposed_model_error"][i]
    Normal_DNN_error = data["Normal_DNN_error"][i]
    Extra_DNN_error = data["Extra_DNN_error"][i]
    p1 = re.compile(r'[\s](.*?)[)]', re.S)
    proposed_model = re.findall(p1, proposed_model_error)
    Normal_DNN = re.findall(p1, Normal_DNN_error)
    Extra_DNN = re.findall(p1, Extra_DNN_error)
    proposed_model_error_new.append(eval(proposed_model[0]))
    Normal_DNN_error_new.append(eval(Normal_DNN[0]))
    Extra_DNN_error_new.append(eval(Extra_DNN[0]))

data["proposed_model_error_new"] = proposed_model_error_new
data["Normal_DNN_error_new"] = Normal_DNN_error_new
data["Extra_DNN_error_new"] = Extra_DNN_error_new
print(data["proposed_model_error_new"])
data.to_csv("new.csv")

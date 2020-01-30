import numpy as np
from DATA import load_data_mnnist
from MODEL import MY_MODEL
from DATA import transform_labels

xtrain,ytrain,xtest,ytest=load_data_mnnist()
ytrain,ytrain=transform_labels(ytrain,ytrain)
my_model=MY_MODEL(xtrain,ytrain,xtest,ytest,show_summary=False,show_details=True,save_model=True)
my_model.build_model()
my_model.fit()
my_model.predict()
print(my_model.get_score())
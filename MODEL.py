import numpy as np
import keras
from sklearn.metrics import accuracy_score as score

class MY_MODEL:

    def __init__(self,xtrain,ytrain,xtest,ytest,show_summary,show_details,save_model,learning_rate=0.01,batch_size=1000,num_epochs=1000):
        self.xtrain=xtrain
        self.ytrain=ytrain
        self.xtest=xtest
        self.ytest=ytest
        self.learning_rate=learning_rate
        self.show_summary=show_summary
        self.show_details=show_details
        self.save_model=save_model
        self.batch_size=batch_size
        self.num_epocchs=num_epochs
    
    def load_model(self,path):
        self.path=path
        self.my_model=keras.models.load_model(path)

    def build_model(self):
        self.input_shape=self.xtrain.shape[1:]
        self.my_model=keras.models.Sequential()
        self.my_model.add(keras.layers.Conv2D(28,kernel_size=(3,3),activation='relu',input_shape=self.input_shape))
        self.my_model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
        self.my_model.add(keras.layers.Flatten())
        self.my_model.add(keras.layers.Dense(units=200,activation='relu'))
        self.my_model.add(keras.layers.Dense(units=10,activation='softmax'))
        self.my_optimizer=keras.optimizers.adam(lr=self.learning_rate)
        self.my_loss=keras.losses.categorical_crossentropy
        self.my_model.compile(loss=self.my_loss,optimizer=self.my_optimizer,metrics=['accuracy'])
        if self.show_summary:
            self.my_model.summary()
    
    def fit(self):
        self.my_model.fit(self.xtrain,self.ytrain,batch_size=self.batch_size,epochs=self.num_epocchs,verbose=self.show_details)
        if self.save_model:
            self.save_my_model()
    
    def save_my_model(self):
        self.my_model.save("my_model.hdf5")
    
    def predict(self):
        self.ypred=self.my_model.predict(self.xtest,verbose=False)

    def get_score(self):
        self.ypred=np.argmax(self.ypred,axis=1)
        assert(self.ypred.shape==self.ytest.shape)
        return score(self.ypred,self.ytest)
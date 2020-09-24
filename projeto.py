"""
esse projeto tem o mero teor educativo,
a intenção e conseguir utilizar os métodos aprendidos em deeplearning.ai
do Cousera

"""

#importando bibliotecas
import pandas as pd
import sklearn
import numpy as np 
import seaborn as sns
from sklearn.model_selection import train_test_split


#importando os dados e lendo os dados
#caminho_padrão  = C:\Users\usuario\Documents\Felipe\FEA USP\FEA.dev\Grupo de estudos AI\projeto redese neurais
df_item = pd.read_csv(r"C:\Users\usuario\Documents\Felipe\FEA USP\FEA.dev\Grupo de estudos AI\projeto redese neurais\items.csv")
df_train = pd.read_csv(r"C:\Users\usuario\Documents\Felipe\FEA USP\FEA.dev\Grupo de estudos AI\projeto redese neurais\sales_train.csv")
df_itemc = pd.read_csv(r"C:\Users\usuario\Documents\Felipe\FEA USP\FEA.dev\Grupo de estudos AI\projeto redese neurais\item_categories.csv")
df_dist  = pd.read_csv(r"C:\Users\usuario\Documents\Felipe\FEA USP\FEA.dev\Grupo de estudos AI\projeto redese neurais\sample_submission.csv")
df_shop = pd.read_csv(r"C:\Users\usuario\Documents\Felipe\FEA USP\FEA.dev\Grupo de estudos AI\projeto redese neurais\shops.csv")
df_test = pd.read_csv(r"C:\Users\usuario\Documents\Felipe\FEA USP\FEA.dev\Grupo de estudos AI\projeto redese neurais\test.csv")

#transformação da coluna date para datetime 
df_train["date"]= pd.to_datetime(df_train["date"])
df_train["date"] = df_train["date"].dt.strftime('%Y-%m-%d')

#transformando a tabela, no formato certo para conseguir aplicar as redes neurais
#lembrando que para aplicar redes neurais precisa deixar claro quais são suas features(variáveis) e qual é seu objetivo 
dataset = df_train.pivot_table(index = ['shop_id','item_id'],values = ['item_cnt_day'],columns = ['date_block_num'],fill_value = 0,aggfunc='sum')
#esse processo transforma o item_cnt_day em 

dataset.head(5)

#o index será a a coluna date_block_num 
dataset.reset_index(inplace = True)

dataset.head(3)

#dando merge nos datasets
dataset = pd.merge(df_test,dataset,on = ['item_id','shop_id'], how ="left")

dataset.head()

#agora meu dataset tem colunas as colunas do testset que tem item_cnt_day (número vendidos) igual a Nan
#transformando em 0 
dataset = dataset.fillna(0)

#%%
#dataset drop 
dataset.drop(['shop_id','item_id','ID'],inplace = True, axis = 1)


dataset.head()


#%%
#estamos pegando todas as colunas, menos a útlima como o X train 
X_train = np.expand_dims(dataset.values[:,:-1],axis = 2)
#a última coluna é o label (ou o dev set)
y_train = dataset.values[:,-1:]
#o test set serão as últimas colunas 
X_test = np.expand_dims(dataset.values[:,1:],axis = 2) 
print(X_train.shape,y_train.shape,X_test.shape)


#%%
import keras
from keras import Sequential
from keras.layers import Dense 
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Flatten, Dropout
#%%
model = Sequential()
model.add(LSTM(units = 64, input_shape = (X_train.shape[1], X_train.shape[2])))
model.add(Dense(units = 1))

model.compile(optimizer = "adam", loss = "mse", metrics  = ["mean_squared_error"])
model.summary()
#%%
model.fit(X_train, y_train, batch_size =4096, nb_epoch = 5)


#%%
from sklearn.metrics import confusion_matrix, accuracy_score
precisão = accuracy_score(y_true, y_pred)
#%%
# creating submission file 
predição = model.predict(X_test)
# we will keep every value between 0 and 20
predição = predição.clip(0,20)
# creating dataframe with required columns 
predição = pd.DataFrame({"ID": df_test["ID"],'item_cnt_month':submission_pfs.ravel()})
# creating csv file from dataframe
#%%
predição.to_excel(r'C:\Users\usuario\Documents\Felipe\FEA USP\FEA.dev\Grupo de estudos AI\Previsão.xlsx',index = False)

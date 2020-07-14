#!/usr/bin/env python
# coding: utf-8

# In[405]:


import torch
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.preprocessing import LabelEncoder


# # draw the coefficient scale

# In[406]:


df = pd.read_csv('covid_19.csv')

df_num = df.drop(['Country/Region', 'Lat', 'Long'], axis=1)
df_num = df_num.drop(index=[0])
df_arr = df_num.to_numpy()

# change df to array to compute coefficient
for i in range(len(df_num)):
    for j in range(df_num.shape[1]):
        df_arr[i][j] = float(df_arr[i][j])

df_float = df_arr.astype(float)

# difference sequence

diff = np.zeros((185,82))
for i in range(df_float.shape[0]):
    for j in range(df_float.shape[1]-1):
        diff[i][j] = df_float[i][j+1] - df_float[i][j]

diff = np.delete(diff, -1, axis=1)


coef_pairs = []
for x in range(diff.shape[0]-1):
    for y in range(x+1, diff.shape[0]):
        coef = np.corrcoef(diff[x], diff[y], rowvar=True)
        coef_pairs.append(coef[1][0])


#change list to df for coloring
coef_to_df = []
coef_to_df_each = []
j = 0
next_start = 0
for i in range(df_float.shape[0]-1):
    for j in range(df_float.shape[0]-1-i):
        coef_to_df_each.append(coef_pairs[next_start + j])
#        print(j)
    coef_to_df.append(coef_to_df_each)
    next_start = next_start + len(coef_to_df_each)
    coef_to_df_each = []

# filling zeros in the upper triangle    
zero = [0]*185
coef_to_df.append(zero)


dataframe_list = []
for i in range(185):
#     print(i)
    temp_list = []
    for k in range(i+1):
        temp_list.append(0)
    if i != 184:
        for x in coef_to_df[i]:
            temp_list.append(x)
    dataframe_list.append(temp_list)

d = pd.DataFrame(np.zeros((185, 185)))
for i in range(185):
    d[i]=dataframe_list[i]
    
# s will be the colored coefficint 
cm = sns.light_palette("green", as_cmap=True)
s = d.style.background_gradient(cmap=cm)


# # assign train, test data into loader

# In[407]:


C = set()
for i in range(185):
    for j in range(185):
        if d.iloc[i,j]>0.8:
            C.add(i)
            C.add(j)
            
x_data = []
y_data = []
start_index = 0
# L = interval
L = 30
for x in (C):
    country_data = list(df_num.iloc[x:x+1,0:].values[0]) 
#     print(country_data)
    for i in range(0,df_num.shape[1]-L):
        x_data.append(country_data[i:i+L])
        if country_data[i+L]-country_data[i+L-1]:
            y_data.append(1)
        else:
            y_data.append(0)


# In[408]:


x_data


# In[409]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
     x_data, y_data, test_size=0.33, random_state=42)


# ### Run only L=20 START

# In[410]:


df_diff = pd.DataFrame(diff)
df_diff_last20 = df_diff.iloc[0:,-30:]
all_last20 = []
for i in range(185):
    all_last20.append(list(df_diff_last20.iloc[0:1,0:]))


# In[411]:


len(last_20_y_test)


# In[412]:


# add data for predicting(last 20 data)
last_20_x_test = []
last_20_y_test = []
for x in all_last20:
    last_20_x_test.append(x)
for i in range(185):
    last_20_y_test.append(0)


# ### Run only L=20 START

# In[413]:


X_train = np.array(X_train) 
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
last_20_x_test = np.array(last_20_x_test)
last_20_y_test = np.array(last_20_y_test)


# In[414]:


last_20_x_test[0]


# In[415]:


import torch.utils.data as Data


X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)
last_20_x_test = torch.from_numpy(last_20_x_test)
last_20_y_test = torch.from_numpy(last_20_y_test)

# y_tensor = torch.tensor(y_train, dtype=torch.long, device=device)



# In[416]:


X_train = X_train.view(-1, 1, L).type(torch.FloatTensor)

X_test = X_test.view(-1, 1, L).type(torch.FloatTensor)

last_20_x_test = last_20_x_test.view(-1, 1, L).type(torch.FloatTensor)


# In[417]:


print(X_train.shape, y_train.shape, X_test.shape, last_20_x_test.shape)


# In[418]:


train_dataset = Data.TensorDataset(X_train, y_train)
test_dataset = Data.TensorDataset(X_test, y_test)

train_loader = Data.DataLoader(dataset = train_dataset, batch_size =128, shuffle = True, num_workers=4) 


test_loader = Data.DataLoader(dataset = test_dataset, batch_size = 20, shuffle = False, num_workers=4,)


# ### Run only L=20 START

# In[419]:


last_20_test_dataset = Data.TensorDataset(last_20_x_test, last_20_y_test)
last_20_test_loader = Data.DataLoader(dataset = last_20_test_dataset, batch_size = 185, shuffle = False, num_workers=4,)


# In[420]:


len(last_20_test_loader)


# # building RNN, LSTM, GRU model

# In[421]:


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(
            input_size=L,
            hidden_size=256,     # rnn hidden unit
            num_layers=1,  )     # number of rnn layer
           # batch_first=True)   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        
        self.out = nn.Linear(256, 2)  # (hidden node數量, output數量)

    def forward(self, x):
        # x: (batch_size, time_step, input_size)
        # h_state: (num_layers, batch_size, hidden_size)
        # r_out: (batch_size, time_step, hidden_size)
        r_out, h_state = self.rnn(x, None)
        outs = self.out(r_out.squeeze(0))
        return outs.squeeze(1)
    
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        
        self.lstm = nn.LSTM(input_size=L, hidden_size=256, num_layers=1)
        self.output = nn.Linear(in_features=256, out_features=2)
    
    def forward(self, x):
        r_out, (r_h, r_c) = self.lstm(x, None)
        out = self.output(r_out[:, -1, :])
        return out

class GRU(nn.Module):
    def __init__(self):
        super(GRU, self).__init__()
        
        self.gru = nn.GRU(input_size=L, hidden_size=256, num_layers=1)
        self.output = nn.Linear(in_features=256, out_features=2)
    
    def forward(self, x):
        r_out, h_state = self.gru(x, None)
        out = self.output(r_out[:, -1, :])
        return out


# In[422]:


# Assigning Hyper Parameters
# TIME_STEP = 10      # rnn time step
# INPUT_SIZE = 1      # rnn input size
# LR = 0.02           # learning rate

class Model():
    def __init__(self, net, train_data , test_data , EPOCH=20, LR=0.0001):
        self.net = net
        self.optimizer = torch.optim.Adam(net.parameters(), lr = LR)
        self.criterion = nn.CrossEntropyLoss()
        self.train_loader = train_data
        self.test_loader = test_data
        self.Epoch = EPOCH
        self.LR_ = LR

#        self.net = self.net.to(device)
#        if device == 'cuda':
#            # self.net = torch.nn.DataParallel(self.net)
#            torch.backends.cudnn.benchmark = True

    def plot_acc(self):

        every_history_loss = []
        every_history_train_acc = []
        every_history_test_acc = []
        every_predict = []
        for epoch in range(self.Epoch):
            print('======Epoch=======:', epoch)
            train_loss, train_acc = self.train()
            test_loss, test_acc, predict = self.test()

            every_history_loss.append(train_loss)
            every_history_train_acc.append(train_acc)
            every_history_test_acc.append(test_acc)
            every_predict.append(predict)
        return every_history_loss, every_history_train_acc, every_history_test_acc, every_predict

    def train(self):
        self.net.train()
        training_loss = 0
        correct_ans = 0
        total = 0
        # put the data in the loader into batch_x, batch_y
        for step, (batch_X, batch_y) in enumerate(self.train_loader):
            batch_X, batch_y = batch_X.to('cpu'), batch_y.to('cpu')
            self.optimizer.zero_grad()
            pred_outputs = self.net(batch_X)
#             print(pred_outputs.shape,batch_y.shape )
#             batch_y = batch_y.view(-1, 1).type(torch.FloatTensor)
#             print(pred_outputs.shape,batch_y.shape )
            loss = self.criterion(pred_outputs, batch_y.long())
#             batch_y = batch_y.type_as(pred_outputs)
#             loss = self.criterion(pred_outputs.squeeze(), batch_y)
            #從loss計算反向傳播
            loss.backward()
            #更新所有權種和偏差
            self.optimizer.step()
            training_loss += loss.item()
            
            #從output中取最大的出來作為預測值
#             _, predicted = torch.max(pred_outputs, 1)
            _, predicted = pred_outputs.max(1)
            #batch_y.size(0)=這個 batch裡面有多少筆資料
            total += batch_y.size(0)
            #計算 predicted和 batch_y(actual)之間預測對的個數
            correct_ans += predicted.eq(batch_y).sum().item()        
        
        print('    **Training**')
        print('Loss: %.3f ' % ( training_loss ))
        print('Acc: %.3f%% (%d/%d)' % (100.*(correct_ans/total), correct_ans, total ))
        return training_loss, (correct_ans/total)



    def test(self):
        #evaluate
        self.net.eval()
        testing_loss = 0
        correct_ans = 0
        total = 0
        with torch.no_grad(): 
            # put the data in the loader into batch_x, batch_y
            predict = []
            for step, (batch_X, batch_y) in enumerate(self.test_loader):
                batch_X, batch_y = batch_X.to('cpu'), batch_y.to('cpu')
#                 print(batch_y)
                #predict the outputs
                pred_outputs = self.net(batch_X)
        
                # interval = 20 draw map
                predict.append(pred_outputs)
                # interval = 20 draw map

                loss = self.criterion(pred_outputs, batch_y.long())
                
                testing_loss += loss.item()
                #從output中取最大的出來作為預測值


                _, predicted = pred_outputs.max(1)
                #batch_y.size(0)=這個 batch裡面有多少筆資料
                total += batch_y.size(0)
                #計算 predicted和 batch_y(actual)之間預測對的個數
                correct_ans += predicted.eq(batch_y).sum().item()   

        print('    **Testing**')
        print('Loss: %.3f ' % ( testing_loss ))
        print('Acc: %.3f%% (%d/%d)' % (100.*(correct_ans/total), correct_ans, total ))
        return testing_loss, (correct_ans/total), predict


# # run GRU model

# In[432]:


gru_module = Model(GRU(), train_loader, test_loader, EPOCH=100, LR=0.0005)
# draw the acc and loss
history_loss, history_train_acc, history_test_acc, predict_GRU = gru_module.plot_acc()


# In[ ]:


predict_GRU[-1]


# In[433]:


fig, ax = plt.subplots(1, 2)
fig.set_size_inches(12, 4)
ax[0].set_title('Train Accuracy')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Accuracy')
ax[0].plot(history_train_acc)

ax[1].set_title('Test Accuracy')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy')
ax[1].plot(history_test_acc)
# plt.legend(loc=1)


# # run RNN model

# In[434]:


rnn_module = Model(RNN(), train_loader, test_loader, EPOCH=100, LR=0.0005)


history_loss, history_train_acc, history_test_acc, predict_RNN = rnn_module.plot_acc()


# In[437]:


fig, ax = plt.subplots(1, 2)
fig.set_size_inches(12, 4)
ax[0].set_title('Train Accuracy')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Accuracy')
ax[0].plot(history_train_acc)

ax[1].set_title('Test Accuracy')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy')
ax[1].plot(history_test_acc)
# plt.legend(loc=1)


# # run LSTM model

# In[423]:


lstm_module = Model(LSTM(), train_loader, test_loader, EPOCH=100, LR=0.0005)


history_loss, history_train_acc, history_test_acc, predict_LSTM = lstm_module.plot_acc()


# In[424]:


fig, ax = plt.subplots(1, 2)
fig.set_size_inches(12, 4)
ax[0].set_title('Train Accuracy')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Accuracy')
ax[0].plot(history_train_acc)

ax[1].set_title('Test Accuracy')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy')
ax[1].plot(history_test_acc)
# plt.legend(loc=1)


# In[425]:


class last_20_prediction():
    def __init__(self, net , test_data):
        self.net = net
        self.test_loader = test_data

    def test(self):
        #evaluate
        self.net.eval()
        with torch.no_grad(): 
            # put the data in the loader into batch_x, batch_y
            predict = []
            for step, (batch_X, batch_y) in enumerate(self.test_loader):
                batch_X, batch_y = batch_X.to('cpu'), batch_y.to('cpu')
                pred_outputs = self.net(batch_X)
        
                # interval = 20 draw map
                predict.append(pred_outputs)

        return predict


# In[426]:


# first comment the train fuction
last_20_prediction_module = last_20_prediction(LSTM(), last_20_test_loader)
last_20_prediction_ans = last_20_prediction_module.test()


# In[395]:


last_20_prediction_ans


# # draw the world map

# In[435]:


import pygal
from pygal.style import DarkStyle


# In[397]:


last_20_prediction_ans[0][0][0].item()


# In[427]:


# convert probs into outputs and get the country names
ascending = []
ascending_prob = []
descending = []
descending_prob =[]
for i, x in enumerate(last_20_prediction_ans[0]):
    if last_20_prediction_ans[0][i][0] >= last_20_prediction_ans[0][i][1]:
        ascending.append(df.iloc[i+1:i+2,0:1].values[0][0])
        ascending_prob.append(last_20_prediction_ans[0][i][0].item())
    else:
        descending.append(df.iloc[i+1:i+2,0:1].values[0][0])
        descending_prob.append(last_20_prediction_ans[0][i][1].item())


# In[399]:


print(ascending[0])


# In[400]:


len(ascending)


# In[428]:


# 返回il8n模塊中COUNTRIES字典中對應國家名的國別碼
from pygal_maps_world.i18n import COUNTRIES

def get_country_code(last_20_country_name):
    country_code = []
    for i in range(len(last_20_country_name)):
        # 返回字典的所有鍵值對
        for code, name in COUNTRIES.items():
            if name == last_20_country_name[i]:  # 根據國家名返回兩個字母的國別碼
                country_code.append(code)
                
    return country_code
#     return None  # 如果沒有找到則返回None


# In[429]:


# get all the country code
country_code_ascending = get_country_code(ascending)
country_code_descending = get_country_code(descending)


# In[430]:


ascending_dict = dict(zip(country_code_ascending, ascending_prob))
descending_dict = dict(zip(country_code_descending, descending_prob))


# In[436]:


worldmap_chart = pygal.maps.world.World(style=DarkStyle)

worldmap_chart.title = 'Some countries'
worldmap_chart.add('ascending', ascending_dict)
worldmap_chart.add('descending',descending_dict)
worldmap_chart.render_in_browser()


# In[ ]:





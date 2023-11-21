# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 20:54:46 2023

@author: HP
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from io import BytesIO 



st.set_page_config(page_title="ELECTROSAGE")

title_style = '<p style="font-family:Georgia; color:coral; font-size: 42px;">ELECTROSAGE</p>'
st.markdown(title_style, unsafe_allow_html=True)

#Import the dataset and visualise.

#------------------------------FANS-----------------------------

training_set1 = pd.read_csv('fans.csv')

#training_set = training_set.iloc[:,1:2].values

#plt.plot(training_set, label = 'Shampoo Sales Data')
plt.plot(training_set1['Time'],training_set1['Watt'] ,label = 'Energy')
plt.show()

#Seperate training and test data set.


def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)

sc = MinMaxScaler()
training_data = sc.fit_transform(training_set1)

seq_length = 4
x, y = sliding_windows(training_data, seq_length)

train_size = int(len(y) * 0.67)
test_size = len(y) - train_size

dataX = Variable(torch.Tensor(np.array(x)))
dataY = Variable(torch.Tensor(np.array(y)))

trainX = Variable(torch.Tensor(np.array(x[0:train_size])))
trainY = Variable(torch.Tensor(np.array(y[0:train_size])))

testX = Variable(torch.Tensor(np.array(x[train_size:len(x)])))
testY = Variable(torch.Tensor(np.array(y[train_size:len(y)])))

#LSTM CLASS CODE


class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))

        h_out = h_out.view(-1, self.hidden_size)

        out = self.fc(h_out)

        return out

#MODEL TRAINING
@st.cache_data
def train_lstm(_trainX, _trainY, _num_epochs, _learning_rate, _input_size, _hidden_size, _num_layers):
    lstm = LSTM(num_classes, input_size, hidden_size, num_layers)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        outputs = lstm(trainX * 1.02)
        optimizer.zero_grad()
        loss = criterion(outputs, trainY)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

    return lstm

# Assuming the following parameters for training
num_epochs = 10000
learning_rate = 0.03
input_size = 2
hidden_size = 15
num_layers = 1
num_classes = 1

# Training the model using caching
lstm = train_lstm(trainX, trainY, num_epochs, learning_rate, input_size, hidden_size, num_layers)

#Evaluating the model


lstm.eval()
train_predict = lstm(dataX)

data_predict = train_predict.data.numpy()
dataY_plot = dataY.data.numpy()

# data_predict = sc.inverse_transform(data_predict)
# dataY_plot = sc.inverse_transform(dataY_plot)

plt.axvline(x=train_size, c='r', linestyle='--')

plt.plot(dataY_plot, label='actual')
plt.plot(data_predict,label='Predicted')
plt.suptitle('Energy Time-Series Prediction')
plt.legend(loc="upper right")
plt.show()

#Saving the model and it's weights


torch.save(lstm.state_dict(),'Energy_prediction.pt')


#print(lstm.state_dict())

#Loading the weights


# input_size = 1
# hidden_size = 2
# num_layers = 1

# num_classes = 1

# l2=LSTM(num_classes, input_size, hidden_size, num_layers)
# l2.load_state_dict(torch.load('/content/Energy_prediction.pt'))
# l2

#Actual Energy consumed!


y=np.array(dataY_plot)
y=y.flatten()
TW=np.sum(y)
print("The total Watt usage = ",round(TW,3),"W")

#Predicted Energy Consumed!


data_predict=data_predict.flatten()
TW_pred =np.sum(data_predict)
TW_predic= TW_pred/1000;
print("The total predicted Watt usage = ",round(TW_predic,3),"kW")

#a= float(input("enter the number of fans: "))
#t= float(input("enter total hours of usage: "))





#Import the dataset and visualise.


#-------------------TELEVISION-----------------------

training_set2 = pd.read_csv('television.csv')

#training_set = training_set.iloc[:,1:2].values

#plt.plot(training_set, label = 'Shampoo Sales Data')
plt.plot(training_set2['Time'],training_set2['Watt'] ,label = 'Energy')
plt.show()

#Seperate training and test data set.


def sliding_windows2(data, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)

sc2 = MinMaxScaler()
training_data2 = sc2.fit_transform(training_set2)

seq_length2 = 4
x2, y2 = sliding_windows2(training_data2, seq_length2)

train_size2 = int(len(y2) * 0.67)
test_size2 = len(y2) - train_size2

dataX2 = Variable(torch.Tensor(np.array(x2)))
dataY2 = Variable(torch.Tensor(np.array(y2)))

trainX2 = Variable(torch.Tensor(np.array(x2[0:train_size2])))
trainY2 = Variable(torch.Tensor(np.array(y2[0:train_size2])))

testX2 = Variable(torch.Tensor(np.array(x2[train_size2:len(x2)])))
testY2 = Variable(torch.Tensor(np.array(y2[train_size2:len(y2)])))

#LSTM CLASS CODE


class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))

        h_out = h_out.view(-1, self.hidden_size)

        out = self.fc(h_out)

        return out

#MODEL TRAINING






# Assuming the following parameters for training
num_epochs2 = 10000
learning_rate2 = 0.03
input_size2 = 2
hidden_size2 = 15
num_layers2 = 1
num_classes2 = 1

# Training the model using caching
lstm2 = train_lstm(trainX2, trainY2, num_epochs2, learning_rate2, input_size2, hidden_size2, num_layers2)


#Evaluating the model


lstm2.eval()
train_predict2 = lstm2(dataX2)

data_predict2 = train_predict2.data.numpy()
dataY2_plot = dataY2.data.numpy()

# data_predict = sc.inverse_transform(data_predict)
# dataY_plot = sc.inverse_transform(dataY_plot)

plt.axvline(x=train_size2, c='r', linestyle='--')

plt.plot(dataY2_plot, label='actual')
plt.plot(data_predict2,label='Predicted')
plt.suptitle('Energy Time-Series Prediction')
plt.legend(loc="upper right")
plt.show()

#Saving the model and it's weights


torch.save(lstm2.state_dict(),'Energy_prediction.pt')


#print(lstm.state_dict())

#Loading the weights


# input_size = 1
# hidden_size = 2
# num_layers = 1

# num_classes = 1

# l2=LSTM(num_classes, input_size, hidden_size, num_layers)
# l2.load_state_dict(torch.load('/content/Energy_prediction.pt'))
# l2

#Actual Energy consumed!


y2=np.array(dataY2_plot)
y2=y2.flatten()
TW2=np.sum(y2)
print("The total Watt usage = ",round(TW2,3),"W")

#Predicted Energy Consumed!


data_predict2=data_predict2.flatten()
TW_pred2 =np.sum(data_predict2)
TW_predic2= TW_pred2/1000;
print("The total predicted Watt usage := ",round(TW_predic2,3),"kW")

#t2= float(input("enter total hours of usage: "))

#Import the dataset and visualise.

#-------------------FRIDGE--------------------------
training_set3 = pd.read_csv('fridge.csv')

#training_set = training_set.iloc[:,1:2].values

#plt.plot(training_set, label = 'Shampoo Sales Data')
plt.plot(training_set3['Time'],training_set3['Watt'] ,label = 'Energy')
plt.show()

#Seperate training and test data set.


def sliding_windows3(data, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)

sc3 = MinMaxScaler()
training_data3 = sc3.fit_transform(training_set3)

seq_length3 = 4
x3, y3 = sliding_windows3(training_data, seq_length)

train_size3 = int(len(y) * 0.67)
test_size3 = len(y) - train_size3

dataX3 = Variable(torch.Tensor(np.array(x3)))
dataY3 = Variable(torch.Tensor(np.array(y3)))

trainX3 = Variable(torch.Tensor(np.array(x3[0:train_size3])))
trainY3 = Variable(torch.Tensor(np.array(y3[0:train_size3])))

testX3 = Variable(torch.Tensor(np.array(x3[train_size:len(x3)])))
testY3 = Variable(torch.Tensor(np.array(y3[train_size:len(y3)])))

#LSTM CLASS CODE


class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))

        h_out = h_out.view(-1, self.hidden_size)

        out = self.fc(h_out)

        return out

#MODEL TRAINING





# Assuming the following parameters for training
num_epochs3 = 10000
learning_rate3 = 0.03
input_size3 = 2
hidden_size3 = 15
num_layers3 = 1
num_classes3 = 1

# Training the model using caching
lstm3 = train_lstm(trainX3, trainY3, num_epochs3, learning_rate3, input_size3, hidden_size3, num_layers3)

#Evaluating the model


lstm3.eval()
train_predict3 = lstm3(dataX3)

data_predict3 = train_predict3.data.numpy()
dataY3_plot = dataY3.data.numpy()

# data_predict = sc.inverse_transform(data_predict)
# dataY_plot = sc.inverse_transform(dataY_plot)

plt.axvline(x=train_size, c='r', linestyle='--')

plt.plot(dataY3_plot, label='actual')
plt.plot(data_predict3,label='Predicted')
plt.suptitle('Energy Time-Series Prediction')
plt.legend(loc="upper right")
plt.show()

#Saving the model and it's weights


torch.save(lstm3.state_dict(),'Energy_prediction.pt')


#print(lstm.state_dict())

#Loading the weights


# input_size = 1
# hidden_size = 2
# num_layers = 1

# num_classes = 1

# l2=LSTM(num_classes, input_size, hidden_size, num_layers)
# l2.load_state_dict(torch.load('/content/Energy_prediction.pt'))
# l2

#Actual Energy consumed!


y3=np.array(dataY_plot)
y3=y3.flatten()
TW3=np.sum(y3)
print("The total Watt usage = ",round(TW3,3),"W")

#Predicted Energy Consumed!


data_predict3=data_predict3.flatten()
TW_pred3 =np.sum(data_predict3)
TW_predic3= TW_pred3/1000;
print("The total predicted Watt usage = ",round(TW_predic3,3),"kW")


#Import the dataset and visualise.

#------------------------LIGHTS--------------------------

training_set4 = pd.read_csv('lighting.csv')

#training_set = training_set.iloc[:,1:2].values

#plt.plot(training_set, label = 'Shampoo Sales Data')
plt.plot(training_set4['Time'],training_set4['Watt'] ,label = 'Energy')
plt.show()

#Seperate training and test data set.


def sliding_windows4(data, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)

sc4 = MinMaxScaler()
training_data4 = sc4.fit_transform(training_set4)

seq_length4 = 4
x4, y4 = sliding_windows4(training_data4, seq_length4)

train_size4 = int(len(y4) * 0.67)
test_size4 = len(y4) - train_size4

dataX4 = Variable(torch.Tensor(np.array(x4)))
dataY4 = Variable(torch.Tensor(np.array(y4)))

trainX4 = Variable(torch.Tensor(np.array(x4[0:train_size4])))
trainY4 = Variable(torch.Tensor(np.array(y4[0:train_size4])))

testX4 = Variable(torch.Tensor(np.array(x4[train_size4:len(x4)])))
testY4 = Variable(torch.Tensor(np.array(y4[train_size4:len(y4)])))

#LSTM CLASS CODE


class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))

        h_out = h_out.view(-1, self.hidden_size)

        out = self.fc(h_out)

        return out

#MODEL TRAINING





# Assuming the following parameters for training
num_epochs4 = 10000
learning_rate4 = 0.03
input_size4 = 2
hidden_size4 = 15
num_layers4 = 1
num_classes4 = 1

# Training the model using caching
lstm4 = train_lstm(trainX4, trainY4, num_epochs4, learning_rate4, input_size4, hidden_size4, num_layers4)


#Evaluating the model


lstm4.eval()
train_predict4 = lstm4(dataX4)

data_predict4 = train_predict4.data.numpy()
dataY4_plot = dataY4.data.numpy()

# data_predict = sc.inverse_transform(data_predict)
# dataY_plot = sc.inverse_transform(dataY_plot)

plt.axvline(x=train_size4, c='r', linestyle='--')

plt.plot(dataY4_plot, label='actual')
plt.plot(data_predict4,label='Predicted')
plt.suptitle('Energy Time-Series Prediction')
plt.legend(loc="upper right")
plt.show()

#Saving the model and it's weights


torch.save(lstm4.state_dict(),'Energy_prediction.pt')


#print(lstm.state_dict())

#Loading the weights


# input_size = 1
# hidden_size = 2
# num_layers = 1

# num_classes = 1

# l2=LSTM(num_classes, input_size, hidden_size, num_layers)
# l2.load_state_dict(torch.load('/content/Energy_prediction.pt'))
# l2

#Actual Energy consumed!


y4=np.array(dataY4_plot)
y4=y4.flatten()
TW4=np.sum(y4)
print("The total Watt usage = ",round(TW4,3),"W")

#Predicted Energy Consumed!


data_predict4=data_predict4.flatten()
TW_pred4 =np.sum(data_predict4)
TW_predic4= TW_pred4/1000;
print("The total predicted Watt usage = ",round(TW_predic4,3),"kW")

def calculateBill(units):
  if units <= 50:
    bill = units * 2.00
  elif units <= 100:
    bill = 50 * 2.00 + (units - 50) * 2.50
  elif units <= 150:
    bill = 50 * 2.00 + 50 * 2.50 + (units - 100) * 2.75
  elif units <= 250:
    bill = 50 * 2.00 + 50 * 2.50 + 50 * 2.75 + (units - 150) * 5.25
  elif units <= 500:
    bill = 50 * 2.00 + 50 * 2.50 + 50 * 2.75 + 100 * 5.25 + (units - 250) * 6.30
  elif units <= 800:
    bill = 50 * 2.00 + 50 * 2.50 + 50 * 2.75 + 100 * 5.25 + 250 * 6.30 + (units - 500) * 7.10
  else:
    bill = 50 * 2.00 + 50 * 2.50 + 50 * 2.75 + 100 * 5.25 + 250 * 6.30 + 300 * 7.10 + (units - 800) * 7.10
  return bill









appliance_input = st.number_input("Enter the number of appliances: ", min_value=0, max_value=4)
options=["Fans","Lights","Fridge","Television"]
tot_unit=float(0)
fan_unit=float(0)
light_unit=float(0)
fridge_unit=float(0)
Tele_unit=float(0)
for i in range (0,appliance_input):
    selectbox_key = f"selectbox_{i}"
    selectbox_key1= f"selectbox_{i+9}"
    selectbox_key2=f"selectbox_{i+4}"
    selectbox_key3=f"selectbox_{i+69}"
    choice = st.selectbox("Enter the type of appliances",options,key=selectbox_key)
    if choice=="Fans":
        title_style = "font-size: 12px; color:#0EE8FF ; font-family: Georgia, serif;"
        st.markdown(f'<h1 style="{title_style}">Fans</h1>', unsafe_allow_html=True)
        a= st.number_input("enter the number of fans: ",key=selectbox_key1)
        t= st.number_input("enter total hours of usage: ",key=selectbox_key2)
        TW_unit= TW_predic*a;
        fan_unit= TW_unit*t;
         #print("The total predicted units consumed by fan := ",round(fan_unit,3))
        if(st.button("Show power consumed: ",key=selectbox_key3)):
            st.write("The total predicted units consumed by fan := ",round(fan_unit,3))
        tot_unit+=fan_unit
    elif choice=="Lights":
        title_style = "font-size: 12px; color: #0EE8FF; font-family: Georgia, serif;"
        st.markdown(f'<h1 style="{title_style}">Lights</h1>', unsafe_allow_html=True)
        l= st.number_input("enter total number of lights:",key="bal")
        t4 = st.number_input("Enter total hours of usage:", key="total_hours_input4")
        TW_unit4= TW_predic4*l;
        light_unit= TW_unit4*t4;
        #print("The total predicted units consumed by fan := ",round(fan_unit,3))
        if(st.button("Show power consumed: ",key="my dick is huge3")):
            st.write("The total predicted units consumed by Lights := ",round(light_unit,3))
        tot_unit+=light_unit
    elif choice=="Fridge":
        title_style = "font-size: 12px; color:#0EE8FF; font-family: Georgia, serif;"
        st.markdown(f'<h1 style="{title_style}">Refrigerator</h1>', unsafe_allow_html=True)
        t3 = st.number_input("Enter total hours of usage:", key="total_hours_input3")
        fridge_unit= TW_predic3*t3;
        #print("The total predicted units consumed by fan := ",round(fan_unit,3))
        if(st.button("Show power consumed: ",key="my dick is huge2")):
            st.write("The total predicted units consumed by Refrigerator := ",round(fridge_unit,3))
        tot_unit+=fridge_unit
    elif choice=="Television":
        title_style = "font-size: 12px; color:#0EE8FF; font-family: Georgia, serif;"
        st.markdown(f'<h1 style="{title_style}">Television</h1>', unsafe_allow_html=True)
        t2 = st.number_input("Enter total hours of usage:", key="total_hours_input")
        Tele_unit= TW_predic2*t2;
        #print("The total predicted units consumed by fan := ",round(fan_unit,3))
        if(st.button("Show power consumed: ",key="my dick is huge")):
            st.write("The total predicted units consumed by Television := ",round(Tele_unit,3))   
        tot_unit+=Tele_unit




if(st.button("Total power consumed: ",key="I am God")):
    st.write("The total predicted units consumed by fan monthly := ",round(fan_unit*30.0,3), "units")
    st.write("The total predicted units consumed by Lights monthly:= ",round(light_unit*30.0,3), "units")
    st.write("The total predicted units consumed by Refrigerator monthly:= ",round(fridge_unit*30.0,3), "units")
    st.write("The total predicted units consumed by Television monthly:= ",round(Tele_unit*30.0,3),"units")
    st.write("The total predicted units consumed monthly:= monthly",round(tot_unit*30.0,3),"units")
    fan_p=((fan_unit* 30)/((tot_unit)*30))*100.0
    light_p=((light_unit*30)/((tot_unit)*30))*100.0
    fridge_p=((fridge_unit*30)/((tot_unit)*30))*100.0
    tele_p=((Tele_unit*30)/((tot_unit)*30))*100
    
    labels = ['Fans', 'Lights', 'Refrigerator','Television']
    sizes = [fan_p,light_p,fridge_p,tele_p]
    
    
    # Create a pie chart
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['skyblue','red', 'lightgreen', 'lightcoral'])
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Display the chart using Streamlit
    st.pyplot(fig)

    


if(st.button("Bill generated ")):
    st.write("Total monthly bill:= Rs",round(calculateBill(tot_unit*30.0),3))


#streamlit run c:\users\hp\onedrive\desktop\fanny\app.py   

# ... (your existing code)

# Import the necessary libraries for PDF generation
from reportlab.pdfgen import canvas
import base64


# ... (your existing code)

# Function to generate PDF report
def generate_pdf_report():
    pdf_buffer = BytesIO()
    chart_buffer=BytesIO()
    

    # Create a PDF canvas
    pdf_canvas = canvas.Canvas(pdf_buffer)

    # Add Streamlit content to the PDF
    # Example: Add a title
    pdf_canvas.setFont("Times-Bold", 40)
    pdf_canvas.drawString(40, 780,"ELECTROSAGE")
    pdf_canvas.setFont("Helvetica", 20)

    pdf_canvas.drawString(80, 740, f"Energy consumption report:-")
    pdf_canvas.setFont("Helvetica", 14)
    pdf_canvas.drawString(100, 700, f"> The total predicted units consumed by Fans monthly: {round(fan_unit * 30.0, 3)} units")
    pdf_canvas.drawString(100, 680, f"> The total predicted units consumed by Lights monthly: {round(light_unit * 30.0, 3)} units")
    pdf_canvas.drawString(100, 660, f"> The total predicted units consumed by Refrigerator monthly: {round(fridge_unit * 30.0, 3)} units")
    pdf_canvas.drawString(100, 640, f"> The total predicted units consumed by Television monthly: {round(Tele_unit * 30.0, 3)} units")
    pdf_canvas.drawString(100, 620, f"> The total predicted units consumed monthly: {round(tot_unit * 30.0, 3)} units")
    pdf_canvas.drawString(100, 600, f"> Total monthly bill: Rs {round(calculateBill(tot_unit * 30.0), 3)}")
    


    # Save the PDF
    pdf_canvas.save()

    # Move the buffer's cursor to the beginning
    pdf_buffer.seek(0)

    return pdf_buffer.getvalue()

# ... (your existing code)

# Button to download the PDF report
if st.button("Download PDF Report"):
    pdf_report = generate_pdf_report()
    st.markdown(f'<a href="data:application/octet-stream;base64,{base64.b64encode(pdf_report).decode()}" download="electrosage_report.pdf">Click to download PDF Report</a>', unsafe_allow_html=True)






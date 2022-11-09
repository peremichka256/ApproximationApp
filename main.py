import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tkinter import *
import tkinter as tk
import tkinter.filedialog as fd
import matplotlib.pyplot as mp
import pandas as pd
import numpy as np
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.layers import BatchNormalization
from tensorflow import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping


def Function1():
    filetypes = [('File', '.xlsx *.csv'), ('All files', '')]
    dlg = fd.Open(filetypes=filetypes)
    fl = dlg.show()

    if fl != '':
        message1_entry1.delete(0, END)
        message1_entry1.insert(0, fl)
    x, y = file_acceptance()

    model = Function_Model()
    Function_training(model, x, y)

    array = np.arange(-16, 16, 1)

    model_prediction = model.predict(array)

    print(model_prediction)

    mp.plot(x, y, 'o', array, model.predict(array))
    mp.show()


def file_acceptance():
    url = message1_entry1.get()

    if (url.find(".xlsx", len(url) - 5) != -1):
        WS = pd.read_excel(url)
        WS_np = np.array(WS)
        print(WS.columns.ravel())
        num_rows, num_cols = WS_np.shape
        name_colum = WS.columns.ravel()
        x = np.ones((num_rows, num_cols - 1))
        y = WS_np[:, num_cols - 1]
        x = np.delete(WS_np, np.s_[-1:], axis=1)
        return x, y


#Функция создания модели
def Function_Model():
    model = Sequential()
    model.add(BatchNormalization())
    model.add(Dense(1280, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(640, activation='softmax'))
    model.add(BatchNormalization())
    model.add(Dense(1, 'elu'))
    model.compile(loss='mean_squared_error', optimizer='Adam')
    return model


def Function_training(Model_main, x, y):
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=500)
    mc = keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_acc',
                                         mode='max', verbose=0, save_best_only=True)
    Model_main.fit(x, y, epochs=1500, batch_size=40, shuffle=True,
                             validation_split=0.0, validation_freq=2, callbacks=[es, mc])
    return Model_main


root = Tk()
root.title("Approximation program")
root.geometry("500x250")
message1 = StringVar()
message1_entry1 = Entry(textvariable=message1)
message1_entry1.place(relx=.5, rely=.5, anchor="c")

btn_file1 = Button(text="Choose file", command=Function1)
btn_file1.place(relx=.5, rely=.4, anchor="c")

tk.mainloop()
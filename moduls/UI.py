# Third party moduls
from pygame import mixer
import _tkinter
import tkinter as tk
from tkinter import ttk
import datetime
import time
import numpy as np
import pandas as pd
from PIL import Image, ImageTk, ImageSequence
import matplotlib.pyplot as plt
from scipy import stats
import random
from random import shuffle
from typing import List, Tuple
import sys
import asyncio
import nest_asyncio
nest_asyncio.apply()
from async_tkinter_loop import async_handler, async_mainloop
from pylsl import StreamInlet, resolve_byprop 

# own moduls
import tr_moduls as trm
from neurofeedback import record

import warnings
warnings.filterwarnings("ignore")

class FirstPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.configure(bg='#fafafa')
        
        label_desc = tk.Label(self, text="""\n\n
                             Esta es una prueba de atención, prueba gradCPT.
                             En ella observarás como pasan imágenes de forma gradual,
                             imágenes de ciudades y montañas. Cada vez que veas la 
                             imagen de una ciudad, debes presionar el botón azul.
                                             
                             Presiona continuar para ver las imágenes que encontraras
                             en la prueba. Luego, "Entrenar" para acostumbrarse a la 
                             prueba. Finalmente, "Iniciar"" para hacer la prueba completa.""",
                             fg="#4d4e4f", bg='#fafafa', font=("Arial", 14))
        label_desc.place(x=2, y=5)
        
        Button = tk.Button(self, text="Continuar", font=("Arial", 14), command=lambda: controller.show_frame(SecondPage))
        Button.place(x=590, y=300)
        
        self.bind('<Enter>', self.enter)
        
    def enter(self, event):
        if mixer.music.get_busy():
            mixer.music.stop()
        
        

class SecondPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        
        self.configure(bg='#fafafa')
        
        self.img_num = 1
        imagia = Image.open("pics/images/c1.jpg").convert('L')
        imagia = imagia.resize((350, 350), Image.Resampling.LANCZOS)
        img = ImageTk.PhotoImage(imagia)
        
        self.lbl_img = tk.Label(self, image=img)
        self.lbl_img.image = img 
        self.lbl_img.place(x=140,y=25)
        
        
        Button = tk.Button(self, text="Atrás", font=("Arial", 14), command=lambda: controller.show_frame(FirstPage))
        Button.place(x=612, y=7)

        Button = tk.Button(self, text="<", font=("Arial", 14), command=self.back_img)
        Button.place(x=600, y=100)
        Button = tk.Button(self, text=">", font=("Arial", 14), command=self.next_img)
        Button.place(x=650, y=100)

        Button = tk.Button(self, text="Entrenar", font=("Arial", 14), command=lambda: controller.show_frame(ThirdPage))
        Button.place(x=600, y=200)
        
        Button = tk.Button(self, text="Iniciar", font=("Arial", 14), command=lambda: controller.show_frame(FourthPage))
        Button.place(x=610, y=300)
        
        self.bind('<Enter>', self.enter)
        
    def enter(self, event):
        if mixer.music.get_busy():
            mixer.music.stop()
    
        
    def next_img(self):
        if self.img_num < 20:
            self.img_num += 1
            imagia = Image.open(f"pics/images/c{self.img_num}.jpg").convert('L')
            imagia = imagia.resize((350, 350), Image.Resampling.LANCZOS)
            img = ImageTk.PhotoImage(imagia)
            self.lbl_img.config(image=img)
            self.lbl_img.image = img 
            self.update()
       
    def back_img(self):
        if self.img_num > 1:
            self.img_num -= 1
            imagia = Image.open(f"pics/images/c{self.img_num}.jpg").convert('L')
            imagia = imagia.resize((350, 350), Image.Resampling.LANCZOS)
            img = ImageTk.PhotoImage(imagia)
            self.lbl_img.config(image=img)
            self.lbl_img.image = img
            self.update()
        
        

class ThirdPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.configure(bg='#fafafa')
        
        self.lbl_img = tk.Label(self)
        self.lbl_img.place(x=140,y=25)

        Button = tk.Button(self, text="Atrás", font=("Arial", 14), command=lambda: controller.show_frame(SecondPage))
        Button.place(x=605, y=7)
        
        Button = tk.Button(self, text="Iniciar", font=("Arial", 14), command=lambda: controller.show_frame(FourthPage))
        Button.place(x=605, y=70)
        
        Button = tk.Button(self, text="O", font=("Arial", 14), command = async_handler(self.play_gif))
        Button.place(x=620, y=150)
        
        Button = tk.Button(self, text="            ", font=("Arial", 14), bg="#036ffc")
        Button.place(x=600, y=300)
        
        
    async def play_gif(self):
        imagia = Image.open("gifs/p1-8-min.gif")
        
        mixer.music.play(0)
        print("Test Started.")
        
        for i in ImageSequence.Iterator(imagia):
            img = ImageTk.PhotoImage(i)
            self.lbl_img.config(image=img)
            self.update()
            await asyncio.sleep(0.2)
            if mixer.music.get_busy() == False:
                break
        
        mixer.music.stop()
        
        print("Test Stopped.")
            
        
        
class FourthPage(tk.Frame):

    def __init__(self, parent, controller):
        
        tk.Frame.__init__(self, parent)
        self.configure(bg='#fafafa')
        
        self.pb = ttk.Progressbar(self, orient='horizontal',
                                    mode='determinate',
                                    length=350)
        self.pb.place(x=141,y=3)
        self.pb.step(0)
                
        self.lbl_img = tk.Label(self)
        self.lbl_img.place(x=140,y=25)

        btn_home = tk.Button(self, text="Home", font=("Arial", 14), command=lambda: controller.show_frame(FirstPage))
        btn_home.place(x=605, y=7)

        btn_back = tk.Button(self, text="Atrás", font=("Arial", 14), command=lambda: controller.show_frame(SecondPage))
        btn_back.place(x=607, y=50)      
        
        self.btn_start = tk.Button(self, text="O", font=("Arial", 14), command = self.play_gif) #lambda:[self.play_gif, record]) # record 
        self.btn_start.place(x=620, y=150)
        
        self.btn_result = tk.Button(self, text="Resultados", font=("Arial", 12), command=lambda: controller.show_frame(FivePage))
        self.btn_result.place(x=590, y=200)
        self.btn_result["state"] = "disabled"
        
        btn_click = tk.Button(self, text="            ", font=("Arial", 14), bg="#036ffc", command=self.take_time)
        btn_click.place(x=600, y=300)
        
        self.n = 1
        self.last = 0
        
        self.bind('<Enter>', self.enter)
        
    def enter(self, event):
        if mixer.music.get_busy() and recording==False:
            mixer.music.stop()
            
    def take_time(self):
        global t_details 
        t_details["time"].append(datetime.datetime.now()) # transition start time
        t_details["tag"].append("click")

    def creador_de_lista_final(self):
        lista_final=[]
        lista=[]
        for j in range(60):#el 60 es para 8 minutos, 6 es para prueba de 48 seg, 15 para 2 minutos
            while(len(lista)<9): # inserta las ciudades
                x=random.randint(1,10)
                x=str(x)
                y='pics/grays/ct{}.jpg'.format(x)
                
                if y not in lista:
                    lista.append(y)
            # inserta una montana en una posicion random
            lista.insert(random.randrange(1,10),'pics/grays/mt{}.jpg'.format(random.randrange(1,11)))
            lista_final = lista_final + lista
            lista = []
        return lista_final
        

    @async_handler 
    async def play_gif(self): 
        global recording
        #global f
        global auxCount
        global t_details
        global all_raw_data
        t_details = {"time":[], "tag":[], "tr":[], "flag":[]}
        all_raw_data = {'eeg':[], 'time':[]}
        lst_de_nombres=[]
        images=[]
        auxCount = -1
        self.pb["value"] = 0 # progress bar value

        lst_random_images = self.creador_de_lista_final()
        
        #imagia = Image.open("prueba-8-min.gif")
        
        self.btn_start["state"] = "disabled"
        
        len_list_images = len(lst_random_images)
        for l in range (len_list_images - 1):
            img_act = lst_random_images[l]
            img_sig = lst_random_images[l+1]
            lst_de_nombres.append(img_act)
            lst_de_nombres.append(img_sig)
            
            self.pb["value"] = (int(100*(l+2)/len_list_images)-0.1) 

            with Image.open(img_act) as source_img, Image.open(img_sig) as dest_img:
                # add start image
                images.append(ImageTk.PhotoImage(source_img))
                
                for tran in range(3):
                    source_img = Image.blend(source_img, dest_img, 0.25*(tran+1))
                    images.append(ImageTk.PhotoImage(source_img))
                    await asyncio.sleep(0.0001)  
        
        # Save the images as a GIF - asegurate de guardar el formato correcto de imagen
        #images[0].save("gifs/p3-8-min.gif", save_all=True, append_images=images[1:], duration=200, loop=1)
        recording = True
        mixer.music.play(0)
        self.btn_result["state"] = "normal"
        print("Recording Started.")
        
        inicio = time.time()
        for i in range(len(images)):
            self.lbl_img.config(image = images[i])
            if i%4 == 0:
                print(time.time()- inicio )
                inicio = time.time()
                t_details["time"].append(datetime.datetime.now()) # transition start time
                t_details["tag"].append(lst_random_images[i+1])
                
                if mixer.music.get_busy() == False:
                    recording = False
                    break
                
            self.update()
            await asyncio.sleep(0.2)
       
        
        # Closing process
        recording = False
        mixer.music.stop()
        #f.close()
        self.btn_start["state"] = "normal"
        
        print("Recording Stopped.")



"""
    @async_handler 
    async def play_gif(self): 
        global recording
        #global f
        global auxCount
        global t_details
        global all_raw_data
        t_details = {"time":[], "tag":[], "tr":[], "flag":[]}
        all_raw_data = {'eeg':[], 'time':[]}
        lst_de_nombres=[]
        images=[]
        auxCount = -1

        lst_random_images = self.creador_de_lista_final()
        self.pb['value'] = 10
        #imagia = Image.open("prueba-8-min.gif")
        recording = True
        mixer.music.play(0)
        
        self.btn_start["state"] = "disabled"
        self.btn_result["state"] = "normal"
        print("Recording Started.")
        
        for l in range (len(lst_random_images)-1):
            img_act = lst_random_images[l]
            img_sig = lst_random_images[l+1]
            lst_de_nombres.append(img_act)
            lst_de_nombres.append(img_sig)
            inicio = time.time()
            try:
                with Image.open(img_act) as source_img, Image.open(img_sig) as dest_img:
                    # add start image
                    images.append(source_img.copy())
                    # Update image in TKinter
                    img = ImageTk.PhotoImage(source_img)
                    self.lbl_img.config(image=img)   
                    t_details["time"].append(datetime.datetime.now()) # transition start time
                    t_details["tag"].append(img_sig)
                    self.update()
                
                    # We need to process in that 2 seconds of wait, not then of that time
                    for tran in range(3):
                        source_img = Image.blend(source_img, dest_img, 0.25*(tran+1))
                        images.append(source_img.copy())
                        # Update image in TKinter
                        img = ImageTk.PhotoImage(source_img)
                        self.lbl_img.config(image=img) 
                        await asyncio.sleep(0.001)  
                        self.update()
                    
                    await asyncio.sleep(0.001)

                    fin = time.time()

                    # recording EEG close if not music, is the same that end of test
                    if mixer.music.get_busy() == False:
                        recording = False
                        #f.close()
                        break

            except IOError:
                print(img_act, "           ",img_sig)
                print("Cannot read source or destination as image.")

            print(fin-inicio)
        
        # Closing process
        recording = False
        mixer.music.stop()
        #f.close()
        self.btn_start["state"] = "normal"
        
        print("Recording Stopped.")

        # Save the images as a GIF
        #images[0].save("gifs/p3-8-min.gif", save_all=True, append_images=images[1:], duration=200, loop=1)
        
"""
   

    
class FivePage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.configure(bg='#fafafa')
        
        self.lbl_standard = tk.Label(self, text="Estándar ", bg='#fafafa', font=("Arial", 15))
        self.lbl_standard .place(x=2, y=5)
        self.lbl_tr = tk.Label(self, text="TR promedio: ", bg='#fafafa', font=("Arial", 12))
        self.lbl_tr.place(x=2, y=35)
        self.lbl_correct = tk.Label(self, text="Correctas: ", bg='#fafafa', font=("Arial", 12))
        self.lbl_correct.place(x=2, y=65)
        self.lbl_lapse = tk.Label(self, text="Errores: ", bg='#fafafa', font=("Arial", 12))
        self.lbl_lapse.place(x=2, y=95)

        btn_home = tk.Button(self, text="Home", font=("Arial", 14), command=lambda: controller.show_frame(FirstPage))
        btn_home.place(x=600, y=7)
        
        self.btn_result = tk.Button(self, text="Resultados", font=("Arial", 12), command=self.show_tr)
        self.btn_result.place(x=590, y=200)
        
        self.bind('<Enter>', self.enter)
        
    def enter(self, event):
        global recording
        if mixer.music.get_busy():
            self.lbl_tr.configure(text="TR promedio: ") 
            self.lbl_correct.configure(text="Correctas: ") 
            self.lbl_lapse.configure(text="Errores: ")
            recording = False
            mixer.music.stop()
            #f.close()
        
    def show_tr(self):
        global t_details
        global all_raw_data

        t_details["tr"].append("")
        t_details["flag"].append("")

        details = trm.clean_trs(t_details)
        df_details = pd.DataFrame.from_dict(details)
        df_details.to_csv(fileTimes, index=False)
        
        # mean TR 
        lst_re_times = [i.total_seconds() for i in details["tr"] if i != ""]
        tr_mean = np.mean(lst_re_times)
        # correct and lapse
        lst_correct = ["correct" for i in details["flag"] if i == "correct comission" or i == "correct omission"]
        lst_lapse = ["lapse" for i in details["flag"] if i == "comission error" or i=="omission error"]
        # VTC
        tr_mean_vtc = np.mean(lst_re_times)
        tr_var_vtc = np.std(lst_re_times)
        # values for error trials (CEs and OEs) and
        # correct omissions (COs) were interpolated linearly—that is,
        # by weighting the two neighboring baseline trial RTs.
    
        VTC= [abs(i - tr_mean_vtc)/tr_var_vtc  for i in lst_re_times]
        
        # show standard calculation
        self.lbl_tr.configure(text="TR promedio: {:.4f} seg".format(tr_mean))
        self.lbl_correct.configure(text=f"Correctas: {len(lst_correct)}") 
        self.lbl_lapse.configure(text=f"Errores: {len(lst_lapse)}")  

        # plot
        plt.figure()
        plt.plot(VTC)
        plt.plot([i for i in range(len(VTC))], [np.mean(VTC) for i in range(len(VTC))])
        plt.xlabel('Trial number')
        plt.ylabel('Variance time course')
        plt.show()
        
        # EEG processing  - 'TP9', 'AF7', 'AF8', 'TP10', 'Right AUX'
        all_raw_data['TP9'] = []
        all_raw_data['AF7'] = []
        all_raw_data['AF8'] = []
        all_raw_data['TP10'] = []
        all_raw_data['Right AUX'] = []
        
        for raw in all_raw_data['eeg']:
            all_raw_data['TP9'].append(raw[0])
            all_raw_data['AF7'].append(raw[1])
            all_raw_data['AF8'].append(raw[2])
            all_raw_data['TP10'].append(raw[3])
            all_raw_data['Right AUX'].append(raw[4])
            
        all_raw_data.pop('eeg') # drop column with data now separte in various columns
        df_raw = pd.DataFrame.from_dict(all_raw_data)
        df_raw.to_csv(filePath, index=False)
        
        

class App(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        # creating a window
        window = tk.Frame(self)
        window.pack()

        window.grid_rowconfigure(0, minsize=500)
        window.grid_columnconfigure(0, minsize=800)

        self.frames = {}
        for F in (FirstPage, SecondPage, ThirdPage, FourthPage, FivePage):
            frame = F(window, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(FirstPage)
        
        

    def show_frame(self, page):
        frame = self.frames[page]
        frame.tkraise()
        self.title("Prueba de atención")
        
        
async def eeg_writer():
    global all_raw_data
    # This buffer will hold last n seconds of data and be used for calculations
    BUFFER_LENGTH = 5
    # Length of the epochs used to compute the FFT (in seconds)
    EPOCH_LENGTH = 1
    # Amount of overlap between two consecutive epochs (in seconds)
    OVERLAP_LENGTH = 0.8
    # Amount to 'shift' the start of each next consecutive epoch
    SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH
    
    """ 1. CONNECT TO EEG STREAM """

    # Search for active LSL streams
    print('Looking for an EEG stream...')
    streams = resolve_byprop('type', 'EEG', timeout=2)
    if len(streams) == 0:
        raise RuntimeError('Can\'t find EEG stream.')

    # Set active EEG stream to inlet and apply time correction
    print("Start acquiring data")
    inlet = StreamInlet(streams[0], max_chunklen=12)
    eeg_time_correction = inlet.time_correction()
    
    # Get the stream info and description
    info = inlet.info()
    description = info.desc()
    
    fs = int(info.nominal_srate())
    # capture
    while True:

        """ 3.1 ACQUIRE DATA """
        # Obtain EEG data from the LSL stream
        if recording:
            eeg_data, timestamp = inlet.pull_chunk(
                timeout=1, max_samples=int(SHIFT_LENGTH * fs))

            all_raw_data['eeg'] = all_raw_data['eeg'] + eeg_data
            all_raw_data['time'] = all_raw_data['time'] + timestamp
        await asyncio.sleep(0.00390625) #0.00390625
        
     
# a little necesary time to don't interrupt eeg capture init         
time.sleep(1)

# Initialize variables
filePath = "exports/eeg_data.csv"
fileTimes = "exports/times_data.csv"
recording = False # EEG recording status
all_raw_data = {'eeg':[], 'time':[]}

# Initialize music
mixer.init(44100)
mixer.music.load("lofi-mod.mp3")
mixer.music.play(0)
mixer.music.stop()

# Initialize azync process
app = App()
loop = asyncio.get_event_loop_policy().get_event_loop()

try:
    asyncio.ensure_future(eeg_writer())
    asyncio.ensure_future(async_mainloop(app))#main_loop(app))
    loop.run_forever()
except:
    pass
finally:
    print("Closing")
    loop.close()
    mixer.music.stop()


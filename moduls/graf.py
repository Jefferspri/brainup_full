import tkinter as tk
import matplotlib
from matplotlib.ticker import AutoMinorLocator, FixedLocator
from scipy.ndimage import gaussian_filter1d
import numpy as np
import pandas as pd

matplotlib.use('TkAgg')

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk
)


features = pd.read_csv('features_tk.csv')  

app = tk.Tk()
app.geometry("800x450")
app.title('Tkinter Matplotlib Demo')

label1 = tk.Label(app,text="""TR promedio gradCPT:
correctas:
errores:""", font=("Verdana",12)).place(x=1, y=1)
label2 = tk.Label(app,text="TR promedio modelo:", font=("Verdana",12)).place(x=1, y=70)

frame1 = tk.Frame(app, width=300, height=200, background="bisque")
frame1.place(x=12, y=150) 

frame2 = tk.Frame(app, width=200, height=700, background="bisque")
frame2.place(x=470, y=0)

# create a figure
figure = Figure(figsize=(4.5, 2.5), dpi=100)
figure2 = Figure(figsize=(3.5, 4.5), dpi=100)

# create FigureCanvasTkAgg object
figure_canvas = FigureCanvasTkAgg(figure, frame1)
figure_canvas2 = FigureCanvasTkAgg(figure2, frame2)

# create the toolbar
#NavigationToolbar2Tk(figure_canvas, app)
NavigationToolbar2Tk(figure_canvas2, app)

# create axes
axes = figure.add_subplot()
# data
x = [1,2,3,4,5,6,7,8,9,10]
y1 = [0.8,0.6,0.75,0.78,0.9,1,0.8,0.85,1.12,0.8]
y2 = [0.82,0.7,0.8,0.78,0.8,0.9,0.8,0.75,0.85,0.85]
# create the barchart
major_locator =FixedLocator(x)
axes.xaxis.set_major_locator(major_locator)
axes.scatter(x, y1, color= '#a0a0a3') # estándar
axes.scatter(x, y2, color='#203ee6') # modelo
axes.set_title('Tiempos de Respuesta')
axes.set_ylabel('TR (seg.)')
axes.set_xlabel('Muestra')
axes.legend(['real', 'modelo'])
axes.grid(axis='x')
figure_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# create axes
axes2 = figure2.add_subplot()
# data
data = features[features['channel']=='AF8_fil']
data['cont'] = np.linspace(0,8,data.shape[0])

mask_correct = (data['flag']== 'correct comission')|(data['flag']== 'correct omission')
mask_error = (data['flag']== 'comission error')|(data['flag']== 'omission error')

t_correct = data['cont'][mask_correct]
point_correct = [0.6 for i in t_correct]
t_error = data['cont'][mask_error]
point_error = [0.5 for i in t_error]

data = data.dropna()
x = data['cont'].to_list()
tr = data['tr'].to_list()
tr_smooth = gaussian_filter1d(tr, sigma=2)

power = gaussian_filter1d(data['p_beta']+0.5, sigma=2)

# create the barchart
axes2.plot(x, tr_smooth, color= '#a0a0a3') # estándar
axes2.plot(x, power, color='#203ee6') # modelo
axes2.scatter(t_error, point_error, color= '#e8053d') # estándar
axes2.scatter(t_correct, point_correct, color='#1de096') # modelo
axes2.set_title('TR y Potencia en el tiempo')
axes2.set_ylabel('')
axes2.set_xlabel('Muestra')
axes2.legend(['tr', 'power', 'error', 'correcto'])
axes2.grid(True)
figure_canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)

   
app.mainloop()


"""
# plot
        plt.figure()
        plt.plot(VTC)
        plt.plot([i for i in range(len(VTC))], [np.mean(VTC) for i in range(len(VTC))])
        plt.xlabel('Trial number')
        plt.ylabel('Variance time course')
        plt.show()
"""


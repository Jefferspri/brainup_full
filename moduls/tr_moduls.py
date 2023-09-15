# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 01:18:44 2022

@author: Jefferson EP
"""

import random
import datetime

def elimina_señal(lst_tiempos):
    lst_tempor=[]
    for jota in range (len(lst_tiempos)-1):
        primi=lst_tiempos[jota]
        secun=lst_tiempos[jota+1]
        dif = secun-primi
        lst_tempor.append(dif)
    doble_cklc=datetime.timedelta(seconds=0,milliseconds=40)
    muycorto=datetime.timedelta(seconds=0,milliseconds=560)
    correcto=datetime.timedelta(seconds=1,milliseconds=120)
    lst_deleted = lst_tempor.copy()
    
    
    lst_etiquetas=[]
    for ki in range (len(lst_tempor)):
        if (lst_tempor[ki]<doble_cklc):
            lst_etiquetas.append("indice: {}, doble click".format(ki))
            lst_deleted.remove(lst_tempor[ki])
        elif (doble_cklc <= lst_tempor[ki]<muycorto):
            lst_etiquetas.append("indice: {}, muy corto".format(ki))
            lst_deleted.remove(lst_tempor[ki])
        elif (muycorto <= lst_tempor[ki]<correcto):
            lst_etiquetas.append("indice: {}, correcto".format(ki))
        elif (correcto <= lst_tempor[ki]):
            lst_etiquetas.append("indice: {}, te pasaste".format(ki))
            lst_deleted.remove(lst_tempor[ki]) 
    return lst_tempor, lst_etiquetas, lst_deleted



def clean_trs(t_details):
    t_details["tag"] = [text[11:13] if text!="click" else text for text in t_details["tag"]]
    # filter double click
    try:
        for i in range(len(t_details["tag"])):
            while t_details["tag"][i] == "click" and t_details["tag"][i+1] == "click":
                if (t_details["time"][i+1] - t_details["time"][i]) < datetime.timedelta(seconds=0, milliseconds=200): # 200
                    t_details["time"].pop(i+1)
                    t_details["tag"].pop(i+1)
                else:
                    break

    except:
        pass 

    # filter if first click is part of last transition
    for i in range(len(t_details["tag"])):
        if t_details["tag"][i] == "click":
            if (t_details["time"][i] - t_details["time"][i-1]) < datetime.timedelta(seconds=0, milliseconds=320): # 320
                t_details["time"].insert(i-1, t_details["time"][i])
                t_details["time"].pop(i+1)
                t_details["tag"].insert(i-1, t_details["tag"][i])
                t_details["tag"].pop(i+1)

    # filter ambigous, conserve the fast click
    try:
        for i in range(len(t_details["tag"])):
            while t_details["tag"][i] == "click" and t_details["tag"][i+1] == "click":
                    t_details["time"].pop(i+1)
                    t_details["tag"].pop(i+1)
    except:
        pass   

    # calculate TRs 
    for i in range(len(t_details["tag"])-1):
        if t_details["tag"][i] == "click" and t_details["tag"][i+1] == "ct":
            # In the correct TR range 
            if (t_details["time"][i] - t_details["time"][i-1]) > datetime.timedelta(seconds=0, milliseconds=560) and\
            (t_details["time"][i] - t_details["time"][i-1]) < datetime.timedelta(seconds=0, milliseconds=1120): # 560 and 1120
                t_details["tr"].append(float('nan'))
                t_details["flag"].append("")
                tr = t_details["time"][i] - t_details["time"][i-1]
                tr_seconds = tr.total_seconds()
                t_details["tr"].append(tr_seconds)
                t_details["flag"].append("correct comission")
            else: # To fast TR 
                t_details["tr"].append(float('nan'))
                t_details["flag"].append("")
                tr = t_details["time"][i] - t_details["time"][i-1]
                tr_seconds = tr.total_seconds()
                t_details["tr"].append(tr_seconds)
                t_details["flag"].append("correct comission") 
       
        # Lapse whecn click in mountain
        elif t_details["tag"][i] == "click" and t_details["tag"][i+1] == "mt":
            t_details["tr"].append(float('nan'))
            t_details["flag"].append("")
            tr = t_details["time"][i] - t_details["time"][i-1]
            tr_seconds = tr.total_seconds()
            t_details["tr"].append(tr_seconds)
            t_details["flag"].append("comission error")
        # Correct when dont click for mountain
        elif (t_details["tag"][i] == "ct" and t_details["tag"][i+1] == "mt") or (t_details["tag"][i] == "mt" and t_details["tag"][i+1] == "mt"):
            t_details["tr"].append(float('nan'))
            t_details["flag"].append("correct omission")
        # Lapse because of no click for city
        elif (t_details["tag"][i] == "ct" and t_details["tag"][i+1] == "ct") or (t_details["tag"][i] == "mt" and t_details["tag"][i+1] == "ct"):
            t_details["tr"].append(float('nan'))
            t_details["flag"].append("omission error")

        elif ((t_details["tag"][i] == "ct" and t_details["tag"][i+1] == "click") or (t_details["tag"][i] == "mt" and t_details["tag"][i+1] == "click")) and i == len(t_details["tag"])-2:
            t_details["flag"].append("")
            t_details["tr"].append(float('nan'))
            
    # len correction
    min_len = min(len(t_details["time"]), len(t_details["tag"]), len(t_details["flag"]), len(t_details["tr"]))
    t_details["time"] = t_details["time"][:min_len]
    t_details["tag"] = t_details["tag"][:min_len]
    t_details["flag"] = t_details["flag"][:min_len]
    t_details["tr"] = t_details["tr"][:min_len]

    #for i in range(len(t_details["tag"])):
    #print(len(t_details["tag"]), len(t_details["flag"]), len(t_details["tr"]), len(t_details["time"])) 
        
    #for i in range(len(t_details["flag"])):
     #   print(t_details["tag"][i],t_details["flag"][i],t_details["tr"][i]) 
    
    return t_details
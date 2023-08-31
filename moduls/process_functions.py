import pandas as pd
import datetime
import numpy as np
import math
from scipy import signal
from scipy.signal import filtfilt
import matplotlib.pyplot as plt
import spkit as sp
import pywt

def formating_data(df_eeg, df_rt):
    df_rt = df_rt.copy()
    df_eeg = df_eeg.copy()

    # Format time to seconds in float 
    df_rt['time'] =  pd.to_datetime(df_rt['time'])
    
    # Formating EEG
    df_eeg['time'] = df_eeg['time'].map(lambda x: datetime.datetime.fromtimestamp(x))
    df_eeg = df_eeg.iloc[:,:-1]
    df_eeg = df_eeg[df_eeg['time']>df_rt['time'].iloc[0]]
    df_eeg = df_eeg[df_eeg['time']<df_rt['time'].iloc[-1]]
    df_eeg.reset_index(inplace=True, drop=True)

    return df_eeg, df_rt

def generate_df_rt_date_no_mean(df_rt):
    df_rt = df_rt.copy()

    dic_details = {'start':[], 'end':[], 'rt':[], 'flag':[]}
    # Substract first row if is a click
    if df_rt['tag'].iloc[-1] == 'click':
        df_rt = df_rt.iloc[:-1,:]
    
    # Creating date ranges in each 0.8sec transitions
    for i in range(df_rt.shape[0]-1):
        if df_rt['tag'].iloc[i] != 'click':
            dic_details['start'].append(df_rt['time'].iloc[i])
            
            if df_rt['tag'].iloc[i+1] == 'click':
                dic_details['end'].append(df_rt['time'].iloc[i+2])
                dic_details['flag'].append(df_rt['flag'].iloc[i+2])
                
                if not math.isnan(df_rt['tr'].iloc[i+2]):
                    rt = df_rt['tr'].iloc[i+2]
                    dic_details['rt'].append(rt)
                else:
                    dic_details['rt'].append(float('nan')) # no rt availabe: only avilable for correct comission
                    
            else:
                dic_details['end'].append(df_rt['time'].iloc[i+1])
                dic_details['flag'].append(df_rt['flag'].iloc[i+1])
                
                if not math.isnan(df_rt['tr'].iloc[i+1]):
                    rt = df_rt['tr'].iloc[i+1]
                    dic_details['rt'].append(rt)
                else:
                    dic_details['rt'].append(float('nan')) # no rt availabe: only avilable for correct comission
                       
    df_rt_date = pd.DataFrame(dic_details)
    mask = []
    
    # Filter only valid answers between 0.56 and 1.12 seconds
    for n in range(df_rt_date.shape[0]):
        if not math.isnan(df_rt_date['rt'].iloc[n]):
            if (df_rt_date['rt'].iloc[n]>=0.56)&(df_rt_date['rt'].iloc[n]<=1.12): 
                mask.append(True)
            else:
                mask.append(False)
        else:
            mask.append(True)
            
    df_rt_date = df_rt_date[mask] 
    df_rt_date.reset_index(inplace=True, drop=True)
    
    return df_rt_date


def preprocessimg_data(df_eeg):
    
    df_eeg = df_eeg.copy()
    
    # Applying notch filter in 60Hz.
    b_notch = [0.9879, -0.1937, 0.9879]
    a_notch = [1.0, -0.1937, 0.9758]
    size = df_eeg['TP9'].shape[0]
    df_eeg['TP9_fil'] = filtfilt(b_notch, a_notch, df_eeg['TP9'])
    df_eeg['TP10_fil'] = filtfilt(b_notch, a_notch, df_eeg['TP10'])
    df_eeg['AF7_fil'] = filtfilt(b_notch, a_notch, df_eeg['AF7'])
    df_eeg['AF8_fil'] = filtfilt(b_notch, a_notch, df_eeg['AF8'])

    # Applying high pass filter in 0.5Hz.
    b_high, a_high = signal.butter(3, 0.5, 'hp', fs=256)
    df_eeg['TP9_fil'] = filtfilt(b_high, a_high, df_eeg['TP9_fil'])
    df_eeg['TP10_fil'] = filtfilt(b_high, a_high, df_eeg['TP10_fil'])
    df_eeg['AF7_fil'] = filtfilt(b_high, a_high, df_eeg['AF7_fil'])
    df_eeg['AF8_fil'] = filtfilt(b_high, a_high, df_eeg['AF8_fil'])
    
    # Filter eye noise
    df_eeg['TP9_fil'] = sp.eeg.ATAR(df_eeg['TP9_fil'], wv='db9', winsize=128, beta=0.2,thr_method='ipr',OptMode='soft', verbose=1)
    df_eeg['TP10_fil'] = sp.eeg.ATAR(df_eeg['TP10_fil'], wv='db9', winsize=128, beta=0.2,thr_method='ipr',OptMode='soft', verbose=1)
    df_eeg['AF7_fil'] = sp.eeg.ATAR(df_eeg['AF7_fil'], wv='db9', winsize=128, beta=0.2,thr_method='ipr',OptMode='soft', verbose=1)
    df_eeg['AF8_fil'] = sp.eeg.ATAR(df_eeg['AF8_fil'], wv='db9', winsize=128, beta=0.2,thr_method='ipr',OptMode='soft', verbose=1)

    return df_eeg


def wavelet_packet_decomposition(df_eeg, df_rt_date):
    
    features = {'channel':[],'p_delta':[], 'p_theta':[],'p_alpha':[], 'p_beta':[], 'p_gamma':[],
                'p_max_delta':[], 'p_max_theta':[],'p_max_alpha':[], 'p_max_beta':[], 'p_max_gamma':[],
                'p_min_delta':[], 'p_min_theta':[],'p_min_alpha':[], 'p_min_beta':[], 'p_min_gamma':[],
                'p_beta_theta':[], 'p_beta_alpha':[], 'p_beta_alpha_theta':[],
                'std_delta':[],'std_theta':[], 'std_alpha':[], 'std_beta':[], 'std_gamma':[], 'tr':[], 'time':[], 'flag':[]}
    
    for i in range(df_rt_date.shape[0]):
        df_trans = df_eeg[(df_eeg['time']>=df_rt_date.iloc[i,0]) & (df_eeg['time']<df_rt_date.iloc[i,1])]
        
        if df_trans.shape[0]>=200: # only consider operate feature extracction if we have more than 200 EEG point for 0.8seg
            
            for channel in ['TP9_fil', 'TP10_fil', 'AF7_fil','AF8_fil']:
                chirp_signal = df_trans[channel]

                # Decomposing signal

                # [0, 64] [64, 128] 
                (B11, B12) = pywt.dwt(chirp_signal, 'db9', 'zero')

                # [0, 32] [32, 64]
                (B21, B22) = pywt.dwt(B11, 'db9', 'zero')
                # [64, 96] [96, 128]
                (B23, B24) = pywt.dwt(B12, 'db9', 'zero')

                # [0, 16] [16, 32]
                (B31, B32) = pywt.dwt(B21, 'db9', 'zero')

                # [0, 8] [8, 16]
                (B41, B42) = pywt.dwt(B31, 'db9', 'zero')

                # [0, 4] [4, 8]
                (B51, B52) = pywt.dwt(B41, 'db9', 'zero')
                # [8, 12] [12, 16]
                (B53, B54) = pywt.dwt(B42, 'db9', 'zero')

                # [12, 14] [14, 16]
                (B61, B62) = pywt.dwt(B54, 'db9', 'zero')

                # [12, 13] [13, 14]
                (B71, B72) = pywt.dwt(B61, 'db9', 'zero')

                # grouping signals
                group_delta = [B51] # [0, 4]
                group_theta = [B52] # [4, 8]
                group_alpha = [B71, np.zeros_like(B72), np.zeros_like(B62), B53] # [12, 13] [8, 12] 
                group_beta  = [B72, np.zeros_like(B71), B62, np.zeros_like(B54), np.zeros_like(B42), B32] # [13, 14] [14, 16] [16, 32]
                group_gamma = [B22, B23] # [32, 64] [64, 96]

                # reconstruction
                delta = pywt.waverec(group_delta, 'db9', 'zero') # [0, 4]
                theta = pywt.waverec(group_theta, 'db9', 'zero') # [4, 8]
                alpha = pywt.waverec(group_alpha, 'db9', 'zero') # [8, 13]
                beta = pywt.waverec(group_beta, 'db9', 'zero')   # [13, 32]
                gamma = pywt.waverec(group_gamma, 'db9', 'zero') # [32, 96]
                
                # Compute Characteristics 
                # Welch’s power spectral density
                fs = 256
                (f, S_delta)= signal.welch(delta, fs, nperseg=len(delta)) # nperseg is the number of points that delta have
                (f, S_theta)= signal.welch(theta, fs, nperseg=len(theta))
                (f, S_alpha)= signal.welch(alpha, fs, nperseg=len(alpha))
                (f, S_beta)= signal.welch(beta, fs, nperseg=len(beta))
                (f, S_gamma)= signal.welch(gamma, fs, nperseg=len(gamma))

                # Clasic features: β/θ, β/α, β/(α+θ) 
                beta_theta = sum(S_beta)/sum(S_theta)
                beta_alpha = sum(S_beta)/sum(S_alpha)
                beta_alpha_theta =  sum(S_beta)/(sum(S_alpha)+sum(S_theta))
                
                # Temporal std from band
                std_delta = delta.std()
                std_theta = theta.std() 
                std_alpha = alpha.std()
                std_beta = beta.std()
                std_gamma = gamma.std()

                # Saving features
                     # Power per channel
                features['channel'].append(channel)
                features['p_delta'].append(sum(S_delta))
                features['p_theta'].append(sum(S_theta))
                features['p_alpha'].append(sum(S_alpha))
                features['p_beta'].append(sum(S_beta))
                features['p_gamma'].append(sum(S_gamma))
                    # Max Power per channel
                features['p_max_delta'].append(max(S_delta))
                features['p_max_theta'].append(max(S_theta))
                features['p_max_alpha'].append(max(S_alpha))
                features['p_max_beta'].append(max(S_beta))
                features['p_max_gamma'].append(max(S_gamma))
                    # Min Power per channel
                features['p_min_delta'].append(min(S_delta))
                features['p_min_theta'].append(min(S_theta))
                features['p_min_alpha'].append(min(S_alpha))
                features['p_min_beta'].append(min(S_beta))
                features['p_min_gamma'].append(min(S_gamma))
                    # Power ratios per channel
                features['p_beta_theta'].append(beta_theta)
                features['p_beta_alpha'].append(beta_alpha)
                features['p_beta_alpha_theta'].append(beta_alpha_theta)
                    # STD from temporal EEG
                features['std_delta'].append(delta.std())
                features['std_theta'].append(theta.std())
                features['std_alpha'].append(alpha.std())
                features['std_beta'].append(beta.std())
                features['std_gamma'].append(gamma.std())
                    # Target Etiquetes 
                features['tr'].append(df_rt_date.iloc[i,2])
                features['time'].append(df_rt_date.iloc[i,1])
                features['flag'].append(df_rt_date.iloc[i,3])

    return pd.DataFrame(features)

def z_normalization(df):
    df_zscore = (df - df.mean())/df.std()
    return df_zscore

def normalization_zero_to_one(df):
    df_score = (df - df.min())/(df.max() - df.min())
    return df_score

def normalization(df_features):
        # Power per channel 
    df_features['p_delta'] = normalization_zero_to_one(df_features['p_delta']) 
    df_features['p_theta'] = normalization_zero_to_one(df_features['p_theta']) 
    df_features['p_alpha'] = normalization_zero_to_one(df_features['p_alpha']) 
    df_features['p_beta'] = normalization_zero_to_one(df_features['p_beta']) 
    df_features['p_gamma'] = normalization_zero_to_one(df_features['p_gamma']) 
        # Max Power per channel
    df_features['p_max_delta'] = normalization_zero_to_one(df_features['p_max_delta']) 
    df_features['p_max_theta'] = normalization_zero_to_one(df_features['p_max_theta']) 
    df_features['p_max_alpha'] = normalization_zero_to_one(df_features['p_max_alpha']) 
    df_features['p_max_beta'] = normalization_zero_to_one(df_features['p_max_beta']) 
    df_features['p_max_gamma'] = normalization_zero_to_one(df_features['p_max_gamma']) 
        # Min Power per channel
    df_features['p_min_delta'] = normalization_zero_to_one(df_features['p_min_delta']) 
    df_features['p_min_theta'] = normalization_zero_to_one(df_features['p_min_theta']) 
    df_features['p_min_alpha'] = normalization_zero_to_one(df_features['p_min_alpha']) 
    df_features['p_min_beta'] = normalization_zero_to_one(df_features['p_min_beta']) 
    df_features['p_min_gamma'] = normalization_zero_to_one(df_features['p_min_gamma']) 
        # Power ratios per channel
    df_features['p_beta_theta'] = normalization_zero_to_one(df_features['p_beta_theta']) 
    df_features['p_beta_alpha'] = normalization_zero_to_one(df_features['p_beta_alpha']) 
    df_features['p_beta_alpha_theta'] = normalization_zero_to_one(df_features['p_beta_alpha_theta']) 
        # STD from temporal EEG
    df_features['std_delta'] = normalization_zero_to_one(df_features['std_delta']) 
    df_features['std_theta'] = normalization_zero_to_one(df_features['std_theta']) 
    df_features['std_alpha'] = normalization_zero_to_one(df_features['std_alpha']) 
    df_features['std_beta'] = normalization_zero_to_one(df_features['std_beta']) 
    df_features['std_gamma'] = normalization_zero_to_one(df_features['std_gamma'])
        # test number
    df_features['test'] = [str(1) for l in range(df_features.shape[0])]
    
    return df_features

def pivot_channels(features):
    features =  features.copy()
    #features = features.dropna()

    df_all_features = pd.DataFrame()
    col_names = []

    # concatenate characteristics columns per channel
    for ch in ['TP9','TP10','AF7']: # 
        dfx = features[features['channel']== ch+'_fil' ]
        dfx.reset_index(inplace = True, drop = True)
        dfx = dfx.iloc[:,1:24]
        col_names = col_names + [ch+'_'+text for text in list(dfx.columns)]
        df_all_features = pd.concat([df_all_features, dfx], axis=1, ignore_index=True)
        
    dfx = features[features['channel']==  'AF8_fil']
    dfx.reset_index(inplace = True, drop = True)
    dfx = dfx.iloc[:,1:]
    col_names = col_names + ['AF8_'+text for text in list(dfx.columns)][:-4] + ['tr','time','flag','test']
    df_all_features = pd.concat([df_all_features, dfx], axis=1, ignore_index=True)

    df_all_features.columns = col_names
    df_all_features = df_all_features.dropna()
    df_all_features.reset_index(drop=True, inplace=True)

    return df_all_features

"""
from sklearn import metrics
import pickle

regressor = pickle.load(open('rfr_model.sav', 'rb'))  

y_pred = regressor.predict(X_test)
x = [i for i in range(len(y_pred))] 

print('mae: ',metrics.mean_absolute_error(y_test, y_pred))
print('%mae:', np.mean(100*abs(y_pred-y_test)/y_test))

"""
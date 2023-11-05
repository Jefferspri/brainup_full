import math
import pandas as pd
import datetime
import numpy as np

from scipy import signal
from scipy.signal import filtfilt
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import spkit as sp
import EntropyHub as eh
from scipy.stats import skew

import pywt
import statistics as st



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


# generate date range and RT mean  
def generate_df_rt_date(df_rt):
    import math
    df_rt = df_rt.copy()
    cont = 0
    lst_tr_mean = []
    lst_f_tr_mean = []
    lst_date_start = []
    lst_date_end = []
    for i in range(df_rt.shape[0]):

        if df_rt['tag'].iloc[i] != 'click':
            cont +=1
            if cont == 1 :
                lst_date_start.append(df_rt['time'].iloc[i])
        
        if not math.isnan(df_rt['tr'].iloc[i]):
            
            rt = df_rt['tr'].iloc[i]
            lst_tr_mean.append(rt)

        if cont == 10: # 10
            lst_date_end.append(df_rt['time'].iloc[i])
            lst_f_tr_mean.append(np.mean(lst_tr_mean))
            lst_tr_mean = []
            cont = 0

    if cont != 0:
        lst_date_start = lst_date_start[:-1]

    df_rt_date = pd.DataFrame()
    df_rt_date['start'] = lst_date_start
    df_rt_date['end'] = lst_date_end
    df_rt_date['rt'] = lst_f_tr_mean
    
    df_rt_date = df_rt_date[(df_rt_date['rt']>=0.56)&(df_rt_date['rt']<=1.12)] # filter only valid answers
    df_rt_date.reset_index(inplace=True, drop=True)
    
    return df_rt_date



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


### Interpolate missing reaction times using the average of proximal values.
# Note that this technique behaves poorly when two NaN are following each other 
def interp_rt(df_rt_date):
    df_rt = df_rt_date.copy()
    
    for i in range(df_rt.shape[0]):
        if df_rt['flag'].iloc[i] != "comission error":
            if math.isnan(df_rt['rt'].iloc[i]) and not math.isnan(df_rt['rt'].iloc[i-1]):
                try:
                    df_rt.loc[i,'rt'] = np.mean((df_rt['rt'].iloc[i-1], df_rt['rt'].iloc[i-1]))
                except:
                    df_rt.loc[i,'rt'] = df_rt['rt'].iloc[i-1]
            elif math.isnan(df_rt['rt'].iloc[i]):
                if i < df_rt.shape[0]-1:
                    df_rt.loc[i,'rt'] = df_rt['rt'].iloc[i+1]
    
    return df_rt

### Compute the variance time course (VTC) of the array RT_interp
def compute_VTC(df_rt_date, filt=True, filt_order=3, filt_cutoff=0.05):
    df_rt = df_rt_date.copy()

    VTC = (df_rt['rt'] - df_rt['rt'].mean(skipna=True))/df_rt['rt'].std(skipna=True)
    VTC = VTC.fillna(0)
    fwhm = 9
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

    if filt == True:
        #b, a = signal.butter(filt_order,filt_cutoff)
        #VTC_filtered = signal.filtfilt(b, a, abs(VTC))
        VTC_filtered = gaussian_filter1d(abs(VTC), sigma, mode='reflect')
    
    #df_rt['vtc_noise'] = abs(VTC)
    df_rt['vtc'] = VTC_filtered
    return df_rt 


def calculate_mean_vtc(df_rt_date):
    # Calculate the mean of each group of 10 numbers and replace the 'vtc' column
    group_size = 10
    num_groups = df_rt_date.shape[0]//group_size
    new_range = {'start':[],'end':[],'vtc':[]}
    for i in range(num_groups):
        group_start = i * group_size
        group_end = (i + 1) * group_size
        group_mean = df_rt_date['vtc'][group_start:group_end].mean()
        new_range['start'].append(df_rt_date['start'].iloc[group_start])
        new_range['end'].append(df_rt_date['end'].iloc[group_end-1])
        new_range['vtc'].append(group_mean)
    
    return pd.DataFrame(new_range)


def preprocessimg_data(df_eeg):
    
    df_eeg = df_eeg.copy()
    
    # Applying notch filter in 60Hz.
    b_notch = [0.9879, -0.1937, 0.9879]
    a_notch = [1.0, -0.1937, 0.9758]
    size = df_eeg['TP9'].shape[0]
    df_eeg['TP9_fil'] = filtfilt(b_notch, a_notch, df_eeg['TP9'], padlen=size-1)
    df_eeg['TP10_fil'] = filtfilt(b_notch, a_notch, df_eeg['TP10'], padlen=size-1)
    df_eeg['AF7_fil'] = filtfilt(b_notch, a_notch, df_eeg['AF7'], padlen=size-1)
    df_eeg['AF8_fil'] = filtfilt(b_notch, a_notch, df_eeg['AF8'], padlen=size-1)

    # Applying high pass filter in 0.5Hz.
    #b_high, a_high = signal.butter(3, 0.5, 'hp', fs=256)
    #df_eeg['TP9_fil'] = filtfilt(b_high, a_high, df_eeg['TP9_fil'], padlen=size-1)
    #df_eeg['TP10_fil'] = filtfilt(b_high, a_high, df_eeg['TP10_fil'], padlen=size-1)
    #df_eeg['AF7_fil'] = filtfilt(b_high, a_high, df_eeg['AF7_fil'], padlen=size-1)
    #df_eeg['AF8_fil'] = filtfilt(b_high, a_high, df_eeg['AF8_fil'], padlen=size-1)
    
    # Filter eye noise
    df_eeg['TP9_fil'] = sp.eeg.ATAR(df_eeg['TP9_fil'], wv='db9',beta=0.1,OptMode='elim',verbose=1,k1=10, k2=100, winsize=256)
    df_eeg['TP10_fil'] = sp.eeg.ATAR(df_eeg['TP10_fil'], wv='db9',beta=0.1,OptMode='elim',verbose=1,k1=10, k2=100, winsize=256)
    df_eeg['AF7_fil'] = sp.eeg.ATAR(df_eeg['AF7_fil'], wv='db9',beta=0.1,OptMode='elim',verbose=1,k1=10, k2=100, winsize=256)
    df_eeg['AF8_fil'] = sp.eeg.ATAR(df_eeg['AF8_fil'], wv='db9',beta=0.1,OptMode='elim',verbose=1,k1=10, k2=100, winsize=256)

    return df_eeg

# Calculate Zero Crossing Rate 
def calculate_zcr(signal, frame_length=256, hop_length=256):
    num_frames = (len(signal) - frame_length) // hop_length + 1 # frames number in wich divide the signal to calculate ZCR.
    zcr = np.zeros(num_frames)
    zcr_mod = np.zeros(num_frames)

    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        frame = signal[start:end]
        zcr[i] = np.sum(np.diff(np.sign(frame))) / (2 * frame_length)
        
        zcr_mod[i] = np.count_nonzero(np.diff(np.sign(frame))==0)/frame_length 
        
    #print("Total zcr:")    
    #print(np.count_nonzero(np.diff(np.sign(zcr))==0)/len(zcr))
        
    return zcr, zcr_mod


# Pivot channels from rows to columns 
def pivot_channels(features):
    features =  features.copy()
    #features = features.dropna()

    df_all_features = pd.DataFrame()
    col_names = []

    # concatenate characteristics columns per channel
    for ch in ['TP9','TP10','AF7']: # 
        dfx = features[features['channel']== ch+'_fil' ]
        dfx.reset_index(inplace = True, drop = True)
        dfx = dfx.iloc[:,1:44]  # 26 + 7 + 11 = 44
        col_names = col_names + [ch+'_'+text for text in list(dfx.columns)]
        df_all_features = pd.concat([df_all_features, dfx], axis=1, ignore_index=True)
        
    dfx = features[features['channel']==  'AF8_fil']
    dfx.reset_index(inplace = True, drop = True)
    dfx = dfx.iloc[:,1:47] # 27 + 7 + 11 +2= 45
    col_names = col_names + ['AF8_'+text for text in list(dfx.columns)][:-3] + ['rt','vtc','class']#,'time','flag']
    df_all_features = pd.concat([df_all_features, dfx], axis=1, ignore_index=True)

    df_all_features.columns = col_names
    #df_all_features = df_all_features.dropna()
    df_all_features.reset_index(drop=True, inplace=True)

    return df_all_features


# Wavelet descomposition and calculate characteristics 
def wavelet_packet_decomposition(df_eeg, df_rt_date):
    
    features = {'channel':[],'p_delta':[], 'p_theta':[],'p_alpha':[], 'p_beta':[], 'p_gamma':[],
                'p_max_delta':[], 'p_max_theta':[],'p_max_alpha':[], 'p_max_beta':[], 'p_max_gamma':[],
                'p_min_delta':[], 'p_min_theta':[],'p_min_alpha':[], 'p_min_beta':[], 'p_min_gamma':[],
                'p_beta_theta':[], 'p_beta_alpha':[], 'p_beta_alpha_theta':[],
                'std_delta':[],'std_theta':[], 'std_alpha':[], 'std_beta':[], 'std_gamma':[],
                'mean_zcr':[], 'std_zcr':[], 'total_variation':[],'ap_entropy':[],
                'skew_delta':[],'skew_theta':[], 'skew_alpha':[], 'skew_beta':[], 'skew_gamma':[],
                'mm_p_min_alpha':[],'mm_p_min_beta':[], 'mm_p_max_alpha':[], 'mm_p_max_beta':[], 'mm_total_variation':[],'mm_mean_zcr':[],'mm_std_zcr':[], 'mm_skew_theta':[], 'mm_skew_alpha':[], 'mm_skew_beta':[],'mm_skew_gamma':[],
                'rt':[],'vtc':[],'class':[]}#, 'time':[], 'flag':[]}
    
    lst_p_min_alpha = []
    lst_p_min_beta = []
    lst_p_max_alpha = []
    lst_p_max_beta = []
    lst_total_variation = []
    lst_mean_zcr = []
    lst_std_zcr = []
    lst_skew_theta = []
    lst_skew_alpha = []
    lst_skew_beta = []
    lst_skew_gamma = []

    for i in range(df_rt_date.shape[0]):
        df_trans = df_eeg[(df_eeg['time']>=df_rt_date.iloc[i,0]) & (df_eeg['time']<df_rt_date.iloc[i,1])]
        
        if df_trans.shape[0]>=190: # only consider operate feature extracction if we have more than 190 EEG point for 0.8seg
            
            for channel in ['TP9_fil', 'TP10_fil', 'AF7_fil','AF8_fil']:
                chirp_signal = df_trans[channel].to_numpy()

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

                # Classic features: β/θ, β/α, β/(α+θ) 
                beta_theta = sum(S_beta)/sum(S_theta)
                beta_alpha = sum(S_beta)/sum(S_alpha)
                beta_alpha_theta =  sum(S_beta)/(sum(S_alpha)+sum(S_theta))
                
                # Aproximate Entropy (ApEn)
                Ap_En, Phi1 = eh.ApEn(chirp_signal, m = 10)

                # Total Variation (TV)
                t_vari = np.sum(np.abs(np.diff(chirp_signal)))

                # Skeness
                skew_delta = skew(delta)
                skew_theta = skew(theta)
                skew_alpha = skew(alpha)
                skew_beta = skew(beta)
                skew_gamma = skew(gamma)

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
                    # Zero-crossing rate
                zcr, zcr_mod = calculate_zcr(chirp_signal, 10, 10)
                features['mean_zcr'].append(np.mean(zcr_mod))
                features['std_zcr'].append(np.std(zcr_mod))
                    # Aproximate Entropy (ApEn)
                features['ap_entropy'].append(Ap_En[-1])
                    # Total Variation (TV)
                features['total_variation'].append(t_vari)
                    # Skeness
                features['skew_delta'].append(skew_delta)
                features['skew_theta'].append(skew_theta)
                features['skew_alpha'].append(skew_alpha)
                features['skew_beta'].append(skew_beta)
                features['skew_gamma'].append(skew_gamma)
                    # Nine means
                if len(lst_p_min_alpha) == 9:
                    lst_p_min_alpha = lst_p_min_alpha[1:]
                    lst_p_min_beta = lst_p_min_beta[1:]
                    lst_p_max_alpha = lst_p_max_alpha[1:]
                    lst_p_max_beta = lst_p_max_beta[1:]
                    lst_total_variation = lst_total_variation[1:]
                    lst_mean_zcr = lst_mean_zcr[1:]
                    lst_std_zcr = lst_std_zcr[1:]
                    lst_skew_theta = lst_skew_theta[1:]
                    lst_skew_alpha = lst_skew_alpha[1:]
                    lst_skew_beta = lst_skew_beta[1:]
                    lst_skew_gamma = lst_skew_gamma[1:]
                
                lst_p_min_alpha.append(min(S_alpha))
                lst_p_min_beta.append(min(S_beta))
                lst_p_max_alpha.append(max(S_alpha))
                lst_p_max_beta.append(max(S_beta))
                lst_total_variation.append(t_vari)
                lst_mean_zcr.append(np.mean(zcr_mod))
                lst_std_zcr.append(np.std(zcr_mod))
                lst_skew_theta.append(skew_theta)
                lst_skew_alpha.append(skew_alpha)
                lst_skew_beta.append(skew_beta)
                lst_skew_gamma.append(skew_gamma)

                features['mm_p_min_alpha'].append(np.mean(lst_p_min_alpha))
                features['mm_p_min_beta'].append(np.mean(lst_p_min_beta))
                features['mm_p_max_alpha'].append(np.mean(lst_p_max_alpha))
                features['mm_p_max_beta'].append(np.mean(lst_p_max_beta))
                features['mm_total_variation'].append(np.mean(lst_total_variation))
                features['mm_mean_zcr'].append(np.mean(lst_mean_zcr))
                features['mm_std_zcr'].append(np.mean(lst_std_zcr))
                features['mm_skew_theta'].append(np.mean(lst_skew_theta))
                features['mm_skew_alpha'].append(np.mean(lst_skew_alpha))
                features['mm_skew_beta'].append(np.mean(lst_skew_beta))
                features['mm_skew_gamma'].append(np.mean(lst_skew_gamma))
                

                    # Target Etiquetes
                features['rt'].append(df_rt_date['rt'].iloc[i]) 
                features['vtc'].append(df_rt_date['vtc'].iloc[i]) 
                features['class'].append(df_rt_date['class'].iloc[i])
                #features['time'].append(df_rt_date.iloc[i,1])
                #features['flag'].append(df_rt_date.iloc[i,3])
    
    features = pivot_channels(pd.DataFrame(features))
    features = pd.DataFrame(features)

    return features

def z_normalization(df):
    df_zscore = (df - df.mean())/df.std()
    return df_zscore

def normalization_zero_to_one(df):
    df_score = (df - df.min())/(df.max() - df.min())
    return df_score

def normalization(df_features):
    for col in list(df_features.columns)[:-3]:
        df_features[col] = normalization_zero_to_one(df_features[col]) 

    return df_features


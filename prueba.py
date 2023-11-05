import os
import pandas as pd
from datetime import datetime
from natsort import os_sorted

entries_eeg = os_sorted(os.listdir('exports/eeg'))
print(entries_eeg)

for path in entries_eeg:
	print(path)
	df_eeg = pd.read_csv('exports/eeg/'+path, sep=',')
	df_eeg['time'] = df_eeg['time'].map(lambda x: datetime.fromtimestamp(x))
	print('exports/eeg_date/'+path)
	df_eeg.to_csv('exports/eeg_date/'+path, index=False)
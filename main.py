from subprocess import Popen, PIPE
import pydbus

# DBus object paths
BLUEZ_SERVICE = 'org.bluez'
ADAPTER_PATH = '/org/bluez/hci0'

# setup dbus
bus = pydbus.SystemBus()
adapter = bus.get(BLUEZ_SERVICE, ADAPTER_PATH)
adapter.Powered = True  

commands = [
    "date; muselsl stream --address 00:55:DA:B7:B5:F2",
    "date; python3 UI_th.py"
    #"date; muselsl record --duration 60",
    #"date; python neurofeedback.py",
    ]

try: 
    p_stream = Popen(commands[0], shell=True)
    p_tkinter = Popen(commands[1], shell=True)
    p_stream.wait()
    p_tkinter.wait()
except KeyboardInterrupt:
    adapter.Powered = False
    
    
print("The End of all")




'''
Currently, an Arduino is converting the analog input from an electret
microphone and sending each sample to the serial port 
'''
#%%
import serial.tools.list_ports as list_ports
import serial
import re

def active_ports():
    ports = list(list_ports.comports())
    for p in ports:
        print(p)

def get_serial_data(br = 9600, duration = 10, f = 17e3):
    '''
    Gets serial data for the defined duration of time. Default is 10 seconds at 17 kHz frequency
    '''
    s = serial.Serial(
        port="COM3", baudrate = br, bytesize = 8,     # Adam's laptop only has one USB port (COM3), change based on your own ports

        )
    
    s.flushInput()
    # setting time restraints
    d = f * duration # n seconds worth of data ish at defined frequency
    counter = 0 
    
    # instantiate list to populate
    data = []
    
    # continue grabbing data until d 
    while(counter <= f):
        try:
            val = s.readline().decode("cp1252")
            counter += 1
            i = int(val)
            data.append(i)
        except:
            continue                

    return data

if __name__ == "__main__":
    active_ports() 
    d = get_serial_data(br = 38400)
    print(len(d))

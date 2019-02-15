import win32gui
import win32ui
import win32con
import threading
from threading import Thread, Lock
from PIL import Image
import PIL.ImageOps
import time
import h5py
import numpy as np
from datetime import datetime
import cv2
import os


############################################################
#  Screen Capture
############################################################

#Asynchronously captures screens of a window. Provides functions for accessing
#the captured screen.
class ScreenCapture:
 
    def __init__(self):
        self.mut = Lock()
        self.hwnd = None
        self.its = None         #Time stamp of last image 
        self.i0 = None          #i0 is the latest image; 
        self.i1 = None          #i1 is used as a temporary variable
        self.cl = False         #Continue looping flag
        #Left, Top, Right, and bottom of the screen window
        self.l, self.t, self.r, self.b = 0, 0, 0, 0
        #Border on left and top to remove
        self.bl, self.bt, self.br, self.bb = 0, 0, 0, 0

    #Begins recording images of the screen
    def Start(self):
        #if self.hwnd is None:
        #    return False
        self.cl = True
        thrd = Thread(target = self.ScreenUpdateT)
        thrd.start()
        return True
        
    #Stop the async thread that is capturing images
    def Stop(self):
        self.cl = False
        
    #Thread used to capture images of screen
    def ScreenUpdateT(self):
        #Keep updating screen until terminating
        while self.cl:
            #t1 = time.time()
            self.i1 = self.GetScreenImg()
            #print('Elapsed: ' + str(time.time() - t1))
            self.mut.acquire()
            self.i0 = self.i1               #Update the latest image in a thread safe way
            self.its = time.time()
            self.mut.release()

    #Gets handle of window to view
    #wname:         Title of window to find
    #Return:        True on success; False on failure
    def GetHWND(self, wname):
        self.hwnd = win32gui.FindWindow(None, wname)
        if self.hwnd == 0:
            self.hwnd = None
            return False
        self.l, self.t, self.r, self.b = win32gui.GetWindowRect(self.hwnd)
        return True
         
    #Get's the latest image of the window
    def GetScreen(self):
        while self.i0 is None:      #Screen hasn't been captured yet
            pass
        self.mut.acquire()
        s = self.i0
        self.mut.release()
        return s
         
    #Get's the latest image of the window along with timestamp
    def GetScreenWithTime(self):
        while self.i0 is None:      #Screen hasn't been captured yet
            pass
        self.mut.acquire()
        s = self.i0
        t = self.its
        self.mut.release()
        return s, t
         
    #Gets the screen of the window referenced by self.hwnd
    def GetScreenImg(self):
        if self.hwnd is None:
            raise Exception("HWND is none. HWND not called or invalid window name provided.")
        self.l, self.t, self.r, self.b = win32gui.GetWindowRect(self.hwnd)
        #Remove border around window (8 pixels on each side)
        #Remove 4 extra pixels from left and right 16 + 8 = 24
        w = self.r - self.l - self.br - self.bl
        #Remove border on top and bottom (31 on top 8 on bottom)
        #Remove 12 extra pixels from bottom 39 + 12 = 51
        h = self.b - self.t - self.bt - self.bb
        wDC = win32gui.GetWindowDC(self.hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, w, h)
        cDC.SelectObject(dataBitMap)
        #First 2 tuples are top-left and bottom-right of destination
        #Third tuple is the start position in source
        cDC.BitBlt((0,0), (w, h), dcObj, (self.bl, self.bt), win32con.SRCCOPY)
        bmInfo = dataBitMap.GetInfo()
        im = np.frombuffer(dataBitMap.GetBitmapBits(True), dtype = np.uint8)
        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())
        #Bitmap has 4 channels like: BGRA. Discard Alpha and flip order to RGB
        #For 1920x1080 images:
        #Remove 12 pixels from bottom + border
        #Remove 4 pixels from left and right + border
        return im.reshape(bmInfo['bmHeight'], bmInfo['bmWidth'], 4)[:, :, -2::-1]

    # returns the left, right, top, and bottom coordinates of the window
    def getWindowBounds(self):
        return [self.l, self.r, self.t, self.b]



class InputRecord():

    def __init__(self, save_path, x, y, interval=0.100):
        self.image_array = []
        self.recording = False
        self.sv = ScreenCapture()
        self.save_path = save_path
        self.recording_session = len(os.listdir(save_path))
        self.rows = y
        self.cols = x
        self.interval = interval
        self.x = int(2560/2-self.cols/2) #configured for 1440p recording
        self.y = int(1440/2-self.rows/2)
        return

    def begin_recording(self):
        start_time = time.time()
        self.recording = True
        self.sv.GetHWND("PLAYERUNKNOWN'S BATTLEGROUNDS ")
        self.sv.Start()
        while self.recording == True:
            if(time.time() - start_time >= self.interval): #ms
                start_time = time.time()
                #record screen
                img = self.sv.GetScreen() 
                img = img[self.y:self.y+self.rows, self.x:self.x+self.cols] #crop the image
                self.image_array.append(img)                      
                #reset
                img = None   
        return


    def stop_recording(self, format='h5'):
        print('Stopping recording session....')
        date = str(datetime.now()).replace(":",".").replace(" ","")
        self.sv.Stop()
        self.recording = False
        #save both arrays as h5py?
        if format == 'h5':
            with h5py.File(f'{self.save_path}/recording_{date}.h5', 'w') as hf:
                hf.create_dataset("Images",  data=self.image_array)
        elif format == 'png':
            recording_path = f'{self.save_path}/{date}/'
            if not os.path.isdir(recording_path):
                os.makedirs(recording_path)
            for i in range(len(self.image_array)):
                img = Image.fromarray(self.image_array[i], 'RGB')
                img.save(f'{recording_path}/{i}.png')
        return
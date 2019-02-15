import win32gui
import win32con
import win32api
import pygame
import config
config = config.Config()
import os
cwd = os.getcwd()
import keyboard
import math
from utilz import utils_capture

colors =	{
  "fuschia": (255, 0, 128),  # Transparency color
  "dark_red": (100, 0, 0),
  "bright_green": (0,230,0)
}


class Client():

    def __init__(self):
        pygame.init()
        pygame.font.init()
        pygame.mixer.init()
        
        #screencapture
        # self.sc = utils_capture.InputRecord(recording_folder, config.capture_mode.get(mode)[0],config.capture_mode.get(mode)[1], interval=interval)

        self.clock = pygame.time.Clock()
        self.myfont = pygame.font.SysFont('Arial', 30)
        self.sound_enemy_lock = pygame.mixer.Sound(cwd+'/resources/enemy_lock.wav')
        self.sound_welcome = pygame.mixer.Sound(cwd+'/resources/welcome.wav')
        self.sound_wide_range = pygame.mixer.Sound(cwd+'/resources/wide_range.wav')
        self.sound_short_range = pygame.mixer.Sound(cwd+'/resources/short_range.wav')
        self.sound_long_range = pygame.mixer.Sound(cwd+'/resources/long_range.wav')

        self.screen = pygame.display.set_mode((config.RESOLUTION_GAME_WIDTH, config.RESOLUTION_GAME_HEIGHT), pygame.NOFRAME) # For borderless, use pygame.NOFRAME
        self.done = False

        keyboard.add_hotkey('U', self.start, suppress=False)
        keyboard.add_hotkey('I', self.stop, suppress=False)
        keyboard.add_hotkey('0', self.wr_mode, suppress=False)
        keyboard.add_hotkey('-', self.sr_mode, suppress=False)
        keyboard.add_hotkey('=', self.lr_mode, suppress=False)

        return

    def start(self, recording=False, mode='sr'):
        self.sound_welcome.play()

        self.mode = mode

        # Set window transparency color
        hwnd = pygame.display.get_wm_info()["window"]
        gamewindow = win32gui.FindWindow(None, "PLAYERUNKNOWN'S BATTLEGROUNDS ")
        posX, posY, width, height = win32gui.GetWindowPlacement(gamewindow)[4]

        windowStyles = win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT


        win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE,
                            win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE) | win32con.WS_EX_LAYERED)
        win32gui.SetLayeredWindowAttributes(hwnd, win32api.RGB(*colors.get("fuschia")), 0, win32con.LWA_COLORKEY)
        win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, posX,posY, 0,0, win32con.SWP_NOSIZE)

        # windowAlpha = 180
        # win32gui.SetLayeredWindowAttributes(hwnd, win32api.RGB(0,0,0),
        # windowAlpha, win32con.LWA_ALPHA)

        longrange_text = self.myfont.render('LONG RANGE MODE', False, (255, 0, 255))
        shortrange_text = self.myfont.render('SHORT RANGE MODE', False, (255, 0, 255))

        i=0
        self.tick = 0
        while not self.done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.done = True
            self.screen.fill(colors.get('fuschia')) #transparent background
            self.draw_hud(mode)
            pygame.display.update()
            if(i == 0):
                i=1
            else:
                i=0

        return

    def stop(self):
        self.done = True
        return

    def draw_hud(self, mode):
        self.hud_cages(mode,tick=0)
        self.hud_fps()
        return

    def wr_mode(self):
        self.sound_wide_range.play()

        return
    
    def sr_mode(self):
        self.sound_short_range.play()
        return

    def lr_mode(self):
        self.sound_long_range.play()
        return
    


    ##### HUD ELEMENTS #####

    def hud_fps(self):
        fps = self.myfont.render(str(int(self.clock.get_fps())), True, (100, 255, 100))
        self.clock.tick(60)
        self.screen.blit(fps, (10, 50))    
        return 

    def hud_scale(self, tick):

        return

    def hud_cages(self, mode, tick):
        if mode == 'wr':
            pass
        elif mode == 'sr':
            pygame.draw.rect(self.screen, colors.get('bright_green'), [2560/2-200, 1440/2-200, 400, 400], 1)
        elif mode == 'lr':
            self.drawRegularPolygon(self.screen, colors.get('bright_green'), 4, 315, 2560/2, 1440/2-200, 250)
            pygame.draw.rect(self.screen, colors.get('dark_red'), [2560/2-100, 1440/2-100, 200, 200], 1)        
        else:
            pygame.draw.rect(self.screen, colors.get('dark_red'), [2560/2-100, 1440/2-100, 200, 200], 1)
            pygame.draw.rect(self.screen, colors.get('dark_red'), [2560/2-640, 1440/2-355, 1280, 720], 1)
            # pygame.draw.rect(self.screen, (0,190+40*i,0), [2560/2-200, 1440/2-200, 400, 400], 1)       

        return

    def drawRegularPolygon(self, surface, color, numSides, tiltAngle, x, y, radius):
        pts = []
        for i in range(numSides):
            x = x + radius * math.cos(tiltAngle + math.pi * 2 * i / numSides)
            y = y + radius * math.sin(tiltAngle + math.pi * 2 * i / numSides)
            pts.append([int(x), int(y)])
        pygame.draw.polygon(surface, color, pts, 1)

    # put array data into a pygame surface
    def put_arr(self, surface, myarr):
        bv = surface.get_buffer()
        bv.write(myarr.tostring(), 0)


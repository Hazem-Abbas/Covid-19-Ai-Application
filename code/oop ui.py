import tkinter as tk
import cv2
from tkinter import filedialog
import numpy as np
import os
from PIL import ImageTk,Image

net = cv2.dnn.readNet("best.pt")
Classes = ["Mask","no mask","incorrect mask"]
layernames = net.getLayerNames()
output_layers = [layernames[i[0]-1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0,255,size = (len(Classes),3))

class other_forms():
    def __init__(self,x,titl="New Window",geom='900x1000'):
        self.window = tk.Tk()
        self.window.tkraise()
        self.window.geometry(geom)
        self.window.title(titl)
        self.window.resizable(False, False)
        self.back_btn2 = tk.Button(self.window, text="Back",
                                   command=lambda: self.swap_gui(x)).place(x=500, y=700)
        self.exit_btn = tk.Button(self.window, text="Exit", bg='Red',
                                  command=lambda: exit()).place(x=500, y=750)

    def swap_gui(self,x):
        x.state = 0
        self.window.destroy()

    def browse1(self,x):
        self.file1 = filedialog.askopenfilename(initialdir = os.getcwd(), title='Choose an image...',
                                                filetypes = (('PNG File','*.png'),('JPG File',"*.jpg"),('jpeg files', '*.jpeg')))
        if self.file1 != None:
            self.path.set(self.file1)
        self.window.tkraise()
        #self.file1=self.file1.split('/')
        #self.file1 = self.file1[-1]
        print(self.file1)
        self.img1=cv2.imread(self.file1)
        print("shape: {}".format(self.img1.shape))
        print("height: {} pixels".format(self.img1.shape[0]))
        print("width: {} pixels".format(self.img1.shape[1]))
        print("channels: {}".format(self.img1.shape[2]))

        cv2.imshow("Image",self.img1)
        cv2.waitKey(0)

        self.window.tkraise()

        self.canvas = tk.Canvas(self.window, bg="Green", width=700, height=577)
        self.canvas.pack()
        #self.img1 = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img1)

    def browse2(self,y):
        self.file1 = filedialog.askopenfilename(initialdir=os.getcwd(), title='Choose a video...',
                                                filetypes=( ("video files", ".mp4"),("video files", ".flv"),("video files", ".avi")))
        if self.file1 != None:
            y.vid_frame.path.set(self.file1)
        self.window.tkraise()
        print(self.file1)
        video = cv2.VideoCapture(self.file1)
        if not video.isOpened():
            print("Cannot open video")
            return
        fps = int(video.get(cv2.CAP_PROP_FPS));
        print("fps : ", fps)
        print("frame count : ", int(video.get(cv2.CAP_PROP_FRAME_COUNT)))
        print("height : ", int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print("width : ", int(video.get(cv2.CAP_PROP_FRAME_WIDTH)))
        print("Duration = frame count/fps : ", int(video.get(cv2.CAP_PROP_FRAME_COUNT))//fps,"Seconds")
        while (video.isOpened()):
            # Capture frame-by-frame
            ret, frame = video.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Error happened while reading video!")
                break
            # Our operations on the frame come here
            frame = cv2.resize(frame, (600, 500))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Display the resulting frame
            cv2.imshow('Video', gray)
            # When everything done, release the capture
            if cv2.waitKey(25) & 0xFF == ord('0'):
                video.set(cv2.CAP_PROP_POS_FRAMES,0)
            if cv2.waitKey(25) & 0xFF == ord('x'):
                video.release()
                cv2.destroyAllWindows()
                self.window.tkraise()
                break


class main_form():
    def __init__(self):
        self.state=0
        self.tkWindow = tk.Tk()
        self.tkWindow.title("Safety Agent")
        self.tkWindow.geometry('900x1000')
        self.tkWindow.configure(bg='Green Yellow')
        self.tkWindow.resizable(False, False)

        self.canvas = tk.Canvas(self.tkWindow, bg="Green", width=900, height=577)
        self.canvas.pack()
        self.img = tk.PhotoImage(file="mask1.png")
        img2 = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img)

        self.Lbl = tk.Label(self.tkWindow, font="Times 11 italic bold", text="Choose a Source : ").place(x=300, y=650)
        self.cam_btn = tk.Button(self.tkWindow, text="Camera", bg="rosy brown",
                                 font="Cooper 11 italic bold",activebackground="light blue", width=15, height=5,
                                 command=self.open_cam).place(x=250, y=690)
        self.img_btn = tk.Button(self.tkWindow, text="Image", bg="rosy brown",
                                 font="Cooper 11 italic bold",activebackground="light blue", width=15, height=5,
                                 command=self.select_img).place(x=400, y=690)
        self.vid_btn = tk.Button(self.tkWindow, text="Video", bg="rosy brown",
                                 font="Cooper 11 italic bold",activebackground="light blue", width=15, height=5,
                                 command=self.select_vid).place(x=550, y=690)
        self.Exit_btn = tk.Button(self.tkWindow, text="Exit", bg='Red',
                                  font="Cooper 11 italic bold", activebackground="light blue",width=15, height=5,
                                  command=lambda: exit()).place(x=400, y=800)
        self.tkWindow.mainloop()

    def open_cam(self):
        if(self.state!=0):
            return
        self.cam_frame = other_forms(self,"Camera Window")
        self.state = 1
        # display video
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
        while (cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Error happened while capture!")
                break
            # Our operations on the frame come here
            frame = cv2.resize(frame,(600,500))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Display the resulting frame
            cv2.imshow('Camera', frame)
            # When everything done, release the capture
            if cv2.waitKey(25) & 0xFF == ord('x'):
                cap.release()
                cv2.destroyAllWindows()
                self.cam_frame.window.tkraise()
                break


    def select_img(self):
        if (self.state != 0):
            return
        self.img_frame = other_forms(self,"Image Window")
        self.state=2
        self.img_frame.path = tk.StringVar()
        self.img_frame.img_enter = tk.Entry(self.img_frame.window, textvariable=self.img_frame.path).pack()
        self.img_frame.select_btn = tk.Button(self.img_frame.window, text="Browse",
                                              command=lambda: self.img_frame.browse1(self)).pack()

    def select_vid(self):
        if (self.state != 0):
            return
        self.vid_frame = other_forms(self, "Video Window")
        self.state = 3
        self.vid_frame.path = tk.StringVar()
        self.vid_frame.vid_enter = tk.Entry(self.vid_frame.window,width = 30,
                                            textvariable=self.vid_frame.path).place(x=290, y=100)
        self.vid_frame.select_btn = tk.Button(self.vid_frame.window, text="Browse",
                                              command=lambda: self.vid_frame.browse2(self)).place(x=490, y=100)


start = main_form()
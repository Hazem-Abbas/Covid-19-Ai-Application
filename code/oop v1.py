import tkinter as tk
import cv2
from tkinter import filedialog
import numpy as np
import os
import torch
#from gtts import gTTS
from audioplayer import AudioPlayer

# Model
width,height =640,640
device1 = torch.device('cpu')
model = torch.hub.load('yolov5', 'custom', path='Resources/weights/best.pt', source='local')  # local repo
model.multi_label = False  # NMS multiple labels per box

#tts = gTTS(text="With Mask")
#tts.save('hello1.mp3')


class other_forms():
    def __init__(self):
        self.file1=""
        self.file2 = "Resources\\audio\hello1.mp3"
        self.file3 = "Resources\\audio\hello2.mp3"
    def browse1(self,x):
        #read an image from files
        self.file1 = filedialog.askopenfilename(initialdir = os.getcwd(), title='Choose an image...',
                                                filetypes = (('PNG File','*.png'),('JPG File',"*.jpg"),('jpeg files', '*.jpeg')))
        if self.file1 == '':
            x.state = 0
            return
        print(self.file1)
        self.img1=cv2.imread(self.file1)[:,:,::-1]
        print("shape: {}".format(self.img1.shape))
        print("height: {} pixels".format(self.img1.shape[0]))
        print("width: {} pixels".format(self.img1.shape[1]))
        print("channels: {}".format(self.img1.shape[2]))
        #cv2.imshow("Image : "+ self.file1,self.img1)
        #cv2.waitKey(0)
        results = model(self.img1)
        results.show()
        print(results)
        results.save("Resources\detections\Images")
        x.state = 0

    def browse2(self,x):
        #read a video from files
        self.file1 = filedialog.askopenfilename(initialdir=os.getcwd(), title='Choose a video...',
                                                filetypes=( ("video files", ".mp4"),("video files", ".flv"),("video files", ".avi")))
        if self.file1 == None:
            x.state = 0
            return
        print(self.file1)
        video = cv2.VideoCapture(self.file1)
        if not video.isOpened():
            print("Cannot open video")
            x.state = 0
            return
        video.set(3, width)  # width
        video.set(4, height)  # height
        #print("fps : ", fps)
        print("frame count : ", int(video.get(cv2.CAP_PROP_FRAME_COUNT)))
        print("height : ", int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print("width : ", int(video.get(cv2.CAP_PROP_FRAME_WIDTH)))
        #print("Duration = frame count/fps : ", int(video.get(cv2.CAP_PROP_FRAME_COUNT))//fps,"Seconds")
        prev_class = 0
        cur_class = 1
        while (video.isOpened()):
            # Capture frame-by-frame
            ret, frame = video.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Error happened while reading video!")
                x.state = 0
                break
            # Our operations on the frame come here
            fps = int(video.get(cv2.CAP_PROP_FPS))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Inference
            results = model(frame, size=640)  # inclu des NMS
            res = results.pandas().xyxy[0]
            # loop through detections and draw them on transparent overlay image
            for i in range(len(res)):
                xmin, ymin, xmax, ymax, confidence, c, name = res.iloc[i, :]
                xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                cur_class = c.item()
                if confidence > 0.6:
                    if(cur_class==1):
                        frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                        frame = cv2.putText(frame, str(name), (xmin, ymin - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                            (0, 255, 0), 2)

                    else:
                        frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
                        frame = cv2.putText(frame, str(name), (xmin, ymin - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                            (255, 0, 0), 2)

            sleep_ms = int(np.round((1 / fps) * 90))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imshow("Video: "+ self.file1, frame)
            if (prev_class != cur_class):
                prev_class = cur_class
                if (cur_class == 1):
                    AudioPlayer(self.file2).play(block=True)
                else:
                    AudioPlayer(self.file3).play(block=True)
            # reset
            if cv2.waitKey(sleep_ms) & 0xFF == ord('0'):
                video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            # close
            if cv2.waitKey(sleep_ms) & 0xFF == ord('q'):
                video.release()
                cv2.destroyAllWindows()
                break
            # When everything done, release the capture
        x.state = 0

    def browse3(self, x):
        # opens the camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            x.state = 0
            return
        cap.set(3, width)  # width
        cap.set(4, height)  # height
        prev_class = 0
        cur_class = 1
        while (cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Error happened while capture!")
                x.state = 0
                break
            # Our operations on the frame come here
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Inference
            results = model(frame, size=640)  # inclu des NMS
            res = results.pandas().xyxy[0]
            # loop through detections and draw them on transparent overlay image
            for i in range(len(res)):
                xmin, ymin, xmax, ymax, confidence, _, name = res.iloc[i, :]
                xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                print(name)
                if confidence > 0.5:
                    frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
                    frame = cv2.putText(frame, str(name), (xmin, ymin - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            sleep_ms = int(np.round((1 / fps) * 1000))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imshow("Camera: " , frame)
            if (prev_class != cur_class):
                prev_class = cur_class
                if (cur_class == 1):
                    AudioPlayer(self.file2).play(block=True)
                else:
                    AudioPlayer(self.file3).play(block=True)
            # close
            if cv2.waitKey(sleep_ms) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break
            # When everything done, release the capture
        x.state = 0


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
        self.img = tk.PhotoImage(file="Resources/Images_used_in_code/mask1.png")
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
        self.cam_frame = other_forms()
        self.state = 1
        self.cam_frame.browse3(self)

    def select_img(self):
        if (self.state != 0):
            return
        self.img_frame = other_forms()
        self.state=2
        self.img_frame.browse1(self)

    def select_vid(self):
        if (self.state != 0):
            return
        self.vid_frame = other_forms()
        self.state = 3
        self.vid_frame.browse2(self)


start = main_form()
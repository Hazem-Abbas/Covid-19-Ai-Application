import tkinter as tk
from functools import partial
import cv2
from tkinter import filedialog
import numpy as np
import os
import torch

class GuiClass():                                    # class for the main window of the program
    def __init__(self):
        self.dt1 = DetectingClass()
        self.state=0                                  # used to disable buttons when one is functioning
        self.tkwindow = tk.Tk()                       # initializes the main window
        self.tkwindow.title("Covid-19 Challenge")
        self.tkwindow.geometry('900x800')
        self.tkwindow.configure(bg='Green Yellow')
        self.tkwindow.resizable(False, False)
        self.canvas = tk.Canvas(self.tkwindow, bg="#1387C6", width=900, height=800)   # used to display image on the window
        self.canvas.pack()
        self.img = tk.PhotoImage(file="Resources\Images_used_in_code\mask.png")
        self.canvas.create_image(450, 0, anchor=tk.NW, image=self.img)
        self.lbl = tk.Label(self.tkwindow, font="Times 22 italic bold",bg="#1387C6",
                            fg="#04537D", text="Choose A Source ").place(x=180, y=250) # used to display text
        self.cam_btn = tk.Button(self.tkwindow, text="Camera", bg="#066CA3",fg="white",
                                 font="Cooper 16  bold",activebackground="light blue", width=10, height=3,
                                 command=partial(self.detect,1)).place(x=50, y=350)             # used to
        self.img_btn = tk.Button(self.tkwindow, text="Image", bg="#066CA3",fg="white",
                                 font="Cooper 16  bold",activebackground="light blue", width=10, height=3,
                                 command=partial(self.detect,2)).place(x=200, y=350)
        self.vid_btn = tk.Button(self.tkwindow, text="Video", bg="#066CA3",fg="white",
                                 font="Cooper 16  bold",activebackground="light blue", width=10, height=3,
                                 command=partial(self.detect,3)).place(x=350, y=350)
        self.exit_btn = tk.Button(self.tkwindow, text="Exit", bg='Red',fg="white",
                                  font="Cooper 16  bold", activebackground="light blue",width=10, height=3,
                                  command=self.tkwindow.destroy).place(x=200, y=500)
        self.tkwindow.mainloop()

    def detect(self, choice):
        if(self.state!=0):
            return
        self.state = 1
        if(choice == 1):
            self.state = self.dt1.detect_cam()
        elif(choice == 2):
            self.state = self.dt1.detect_image()
        elif(choice == 3):
            self.state = self.dt1.detect_video()

class DetectingClass():
    def __init__(self):
        # Model
        self.model = torch.hub.load('yolov5', 'custom', path='Resources/weights/best01.pt', source='local')  # local repo
        self.model.multi_label = False  # NMS multiple labels per box

        self.file1 = ""                 #
        self.width=640
        self.height = 640
        self.prev_class = 0
        self.cur_class = 0
        self.img_res_count = 0  # countes the processed images
        self.video_res_count = 0  # countes the processed videos
        self.webcam_res_count = 0  # countes the processed videos from the webcam
#####################################################################Image######################
    def detect_image(self):
        #read an image from files
        self.file1 = filedialog.askopenfilename(initialdir = os.getcwd(), title='Choose an image...',
                                                filetypes = (('JPG File',"*.jpg"),('PNG File','*.png'),('jpeg files', '*.jpeg'),('jfif files', '*.jfif')))
        if self.file1 == '':
            return 0

        self.img1=cv2.imread(self.file1)
        self.img1 = cv2.cvtColor(self.img1, cv2.COLOR_BGR2RGB)
        print("shape: {}".format(self.img1.shape))
        print("height: {} pixels".format(self.img1.shape[0]))
        print("width: {} pixels".format(self.img1.shape[1]))
        print("channels: {}".format(self.img1.shape[2]))
        results = self.model(self.img1)
        self.img1 = cv2.cvtColor(self.img1, cv2.COLOR_BGR2RGB)
        res = results.pandas().xyxy[0]
        for k in range(len(res)):
            xmin, ymin, xmax, ymax, confidence, c, name = res.iloc[k, :]
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            if confidence > 0.4:
                if (c.item() == 0): #wearing the mask improperly
                    self.img1 = cv2.rectangle(self.img1, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)
                    self.img1 = cv2.putText(self.img1, "incorrect Mask", (xmin, ymin - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                            (0, 255, 255), 2)
                elif (c.item() == 1): #wearing the mask
                    self.img1 = cv2.rectangle(self.img1, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    self.img1 = cv2.putText(self.img1, "With Mask", (xmin, ymin - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                            (0, 255, 0), 2)
                else:                 #not wearing the mask
                    self.img1 = cv2.rectangle(self.img1, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                    self.img1 = cv2.putText(self.img1, "Without Mask", (xmin, ymin - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                            (0, 0, 255), 2)

        # display and save image:
        cv2.imshow("Image : " + self.file1, self.img1)
        cv2.imwrite('Resources/detections/Images/result' + str(self.img_res_count) + '.png',
                                self.img1)
        self.img_res_count = self.img_res_count + 1
        return 0
####################################################################Video####################
    def detect_video(self):
        # read a video from files
        self.file1 = filedialog.askopenfilename(initialdir=os.getcwd(), title='Choose a video...',
                                                filetypes=(("video files", ".mp4"), ("video files", ".flv"),
                                                           ("video files", ".avi")))
        if self.file1 == None:
            return 0

        video = cv2.VideoCapture(self.file1)
        if not video.isOpened():
            print("Cannot open video")
            return 0
        # print("fps : ", fps)
        print("frame count : ", int(video.get(cv2.CAP_PROP_FRAME_COUNT)))
        print("height : ", int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print("width : ", int(video.get(cv2.CAP_PROP_FRAME_WIDTH)))
        # print("Duration = frame count/fps : ", int(video.get(cv2.CAP_PROP_FRAME_COUNT))//fps,"Seconds")

        # initial the output video:
        out_video = cv2.VideoWriter('Resources/detections/videos/result' + str(self.video_res_count) + '.mp4',
                                    cv2.VideoWriter_fourcc(*'mp4v'), 60.0, (self.width, self.height))
        self.video_res_count += 1

        prev_class = 0
        cur_class = 1
        while (video.isOpened()):
            # Capture frame-by-frame
            ret, frame = video.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Error happened while reading video!")
                break
            # Our operations on the frame come here
            fps = int(video.get(cv2.CAP_PROP_FPS))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Inference
            results = self.model(frame, size=640)  # inclu des NMS
            res = results.pandas().xyxy[0]
            # loop through detections and draw them on transparent overlay image
            for i in range(len(res)):
                xmin, ymin, xmax, ymax, confidence, c, name = res.iloc[i, :]
                xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                cur_class = c.item()
                if confidence > 0.4:
                    if (c.item() == 0):   #wearing the mask improperly
                        frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                        frame = cv2.putText(frame, "incorrect Mask", (xmin, ymin - 15),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                                (0, 0, 255), 2)
                    elif (c.item() == 1): #wearing the mask
                        frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                        frame = cv2.putText(frame, "With Mask", (xmin, ymin - 15), cv2.FONT_HERSHEY_SIMPLEX,
                                                0.5,
                                                (0, 255, 0), 2)
                    else:                   #not wearing the mask
                        frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
                        frame = cv2.putText(frame, "Without Mask", (xmin, ymin - 15), cv2.FONT_HERSHEY_SIMPLEX,
                                                0.5,
                                                (255, 0, 0), 2)

            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
            sleep_ms = int(np.round((1 / fps) * 1000))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imshow("Video: " + self.file1, frame)
            # save the frame in the output video:
            out_video.write(frame)
            # close
            if cv2.waitKey(1) & 0xFF == ord('q'):
                video.release()
                out_video.release()
                cv2.destroyAllWindows()
                break
            # reset
            # if cv2.waitKey(1) & 0xFF == ord('0'):
            #    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        video.release()
        out_video.release()
        cv2.destroyAllWindows()
        # When everything done, release the capture
        return 0
#########################################################################Camera##################
    def detect_cam(self):
        # opens the camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            return 0
        cap.set(3, self.width)  # width
        cap.set(3, self.height)  # height
        # initial the output video:
        fourcc2 = cv2.VideoWriter_fourcc(*'MP4V')
        out_webcam = cv2.VideoWriter('Resources/detections/Webcam/result' + str(self.webcam_res_count) + '.mp4',
                                     fourcc2, 10.0, (640, 480))
        self.webcam_res_count += 1

        prev_class = 0
        cur_class = 1
        while (cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Error happened while capture!")
                break
            # Our operations on the frame come here
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Inference
            results = self.model(frame, size=640)  # inclu des NMS
            res = results.pandas().xyxy[0]
            # loop through detections and draw them on transparent overlay image
            for i in range(len(res)):
                xmin, ymin, xmax, ymax, confidence, c, name = res.iloc[i, :]
                xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                cur_class = c.item()
                if confidence > 0.65:
                    if (c.item() == 0):   #wearing the mask improperly
                        frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                        frame = cv2.putText(frame, "incorrect Mask", (xmin, ymin - 15),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                                (0, 0, 255), 2)
                    elif (c.item() == 1): #wearing the mask
                        frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                        frame = cv2.putText(frame, "With Mask", (xmin, ymin - 15), cv2.FONT_HERSHEY_SIMPLEX,
                                                0.5,
                                                (0, 255, 0), 2)
                    else:                 #not wearing the mask
                        frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
                        frame = cv2.putText(frame, "Without Mask", (xmin, ymin - 15), cv2.FONT_HERSHEY_SIMPLEX,
                                                0.5,
                                                (255, 0, 0), 2)

            sleep_ms = int(np.round((1 / fps) * 1000))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imshow("Camera: ", frame)
            # save the frame in the output video:
            out_webcam.write(frame)
            # close
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                out_webcam.release()
                cv2.destroyAllWindows()
                break
            # When everything done, release the capture
        return 0


start = GuiClass()

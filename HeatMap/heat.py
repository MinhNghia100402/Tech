import torch
import numpy as np
import cv2
# import pafy
import time

#===============================

cell_size=40
frame_width = 640
frame_height = 480
n_col = frame_width//cell_size
n_row = frame_height//cell_size
heat_matrix = np.zeros((n_row,n_col))
wei,hei = 0,0
time_skips = 3*10**8
def get_row_col(x,y):
    row = y//cell_size
    col = x//cell_size
    return row,col
#====================================
source = 0

class ObjectDetection:

    """
    Class implements Yolo5 model to make inferences on a youtube video using OpenCV.
    """
    
    def __init__(self):
        """
        Initializes the class with youtube url and output file.
        :param url: Has to be as youtube URL,on which prediction is made.
        :param out_file: A valid output file name.
        """
        self.model = self.load_model()
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("\n\nDevice Used:",self.device)



    def load_model(self):
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model


    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        self.model.conf = 0.25
        self.model.iou = 0.45
        self.model.classes =0
        frame = [frame]
        results = self.model(frame)
     
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        
        return labels, cord


    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

        #=========================================
    def get_row_col(x,y):
        row = y//cell_size
        col = x//cell_size
        return row,col

    #==============================================  
    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        counts = ''
        n = 0
        labels, cord = results
        n = len(labels)
        counts += 'Total : '+ f'{n}' + ' person' 
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 1)
                cv2.putText(frame, counts,(30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                rows,cols = get_row_col(((x1+x2)//2),(y1+y2)//2)
                heat_matrix[rows,cols]+=1
        return frame
    #=========================================
    def get_row_col(x,y):
        row = y//cell_size
        col = x//cell_size
        return row,col

    def __call__(self):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """
        cap = cv2.VideoCapture(source)
        count =0
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        save_result = cv2.VideoWriter('result.avi',fourcc,30.0,(640,480))
        current_time = 0
        while cap.isOpened():
            
            start_time = time.perf_counter()
            ret, frame = cap.read()
            frame = cv2.resize(frame,(640,480))
            if not ret:
                break
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)
            
            # end_time = time.perf_counter()
            # fps = 1 / np.round(end_time - start_time, 3)
            # cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            #=============================
            from skimage.transform import resize
            temp_heat_matrix = heat_matrix.copy()
            temp_heat_matrix = resize(temp_heat_matrix, (frame_height, frame_width))
            temp_heat_matrix = temp_heat_matrix / np.max(temp_heat_matrix)
            temp_heat_matrix = np.uint8(temp_heat_matrix*255)
            image_heat = cv2.applyColorMap(temp_heat_matrix,cv2.COLORMAP_JET )
            cv2.addWeighted(image_heat, 0.5, frame, 0.5, 0,frame)
            #=============================
            
            save_result.write(frame)
            cv2.imshow("img", frame)
            # cv2.imshow('heat',image_heat)
            if time.localtime().tm_min%5==0 and time.localtime().tm_min != current_time: #Automatically take a picture before 5 minutes
                current_time = time.localtime().tm_min
                cv2.imwrite("/home/nghialee/Documents/HIT_Development/HeatMap_Detection/MyHeatMap/custom/runs/frame%d.jpg" % count, frame)    
                count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        save_result.release()
        cv2.destroyAllWindows()

# Create a new object and execute.
detection = ObjectDetection()
detection()
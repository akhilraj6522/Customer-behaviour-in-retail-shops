
# Import necessary packages
import seaborn as sn
import cv2
import matplotlib.pyplot as plt
import csv
import collections
import numpy as np

fig = plt.figure()

# Initialize the videocapture object
cap = cv2.VideoCapture('sample.mp4')
input_size = 320

# Extract sample frame to select ROI
extract_frame = cv2.VideoCapture('sample.mp4')
extract_frame.set(1, 50)
ret, sample_frame = extract_frame.read()
hei, wid, _ = sample_frame.shape
heat_map_reference = np.zeros( (hei, wid) , dtype=np.int64)
final_result = []


frame_skip = int(input("Enter no of frames to skip(eg: 3 for taking 1/3 rd of the video frames) :"))



# Detection confidence threshold
confThreshold =0.1
nmsThreshold= 0.5

font_color = (0, 0, 255)
font_size = 0.5
font_thickness = 2


# Store Coco Names in a list
classesFile = "coco.names"
classNames = open(classesFile).read().strip().split('\n')


# class index for our required detection classes
required_class_index = 0

detected_classNames = []

## Model Files
modelConfiguration = 'yolov3-320.cfg'
modelWeigheights = 'yolov3-320.weights'

# configure the network model
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeigheights)

# Configure the network backend

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Define random colour for each class
np.random.seed(42)


# Function for finding the center of a rectangle
def find_center(x, y, w, h):
    x1=int(w/2)
    y1=int(h/2)
    cx = x+x1
    cy=y+y1
    return cx, cy
    
 

# Function for finding the detected objects from the network output
def postProcess(outputs,img):
    global detected_classNames 
    height, width = img.shape[:2]
    boxes = []
    classIds = []
    confidence_scores = []
    detection = []
    for output in outputs:
        for det in output:
            scores = det[5:]
           # print('scoress:', scores)
            classId = np.argmax(scores)
            #print('clsid:', classId)
            confidence = scores[classId]
            #print('conf:', confidence)
            if classId == required_class_index:
                if confidence > confThreshold:
                    # print(classId)
					
                    # print('scoress:', scores)
                    # print('clsid:', classId)
                    # print('conf:', confidence)
                    print(classId)
                    w,h = int(det[2]*width) , int(det[3]*height)
                    x,y = int((det[0]*width)-w/2) , int((det[1]*height)-h/2)
                    boxes.append([x,y,w,h])
                    classIds.append(classId)
                    confidence_scores.append(float(confidence))
                    print(det[2], det[3])
                    
                    
                    ref_height, ref_width = int(det[2]*wid) , int(det[3]*hei)
                    ref_x, ref_y = int((det[0]*wid)-ref_width/2) , int((det[1]*hei)-ref_height/2)
                    center_x, center_y = find_center(ref_x, ref_y, ref_width, ref_height)
                    print(center_x, center_y)
                    heat_map_reference[center_y][center_x] = heat_map_reference[center_y][center_x] + 1

    # Apply Non-Max Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, confThreshold, nmsThreshold)
    #print('indices:',indices)
    for i in indices:
        x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
        # print(x,y,w,h)

        color = (0, 250, 0)
        name = classNames[classIds[i]]
        detected_classNames.append(name)
        # Draw classname and confidence score 
        cv2.putText(img,f'{name.upper()} {int(confidence_scores[i]*100)}%',
                  (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw bounding rectangle
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
        detection.append([x, y, w, h, 'Person'])
        center = find_center(x, y, w, h)



def realTime():
    frame_cnt = 0
    
    while True:
        success, img = cap.read()
        if success :
            if frame_cnt % frame_skip == 0 :
			
		    
                img = cv2.resize(img,(0,0),None,0.5,0.5)
                ih, iw, channels = img.shape
                blob = cv2.dnn.blobFromImage(img, 1 / 255, (input_size, input_size), [0, 0, 0], 1, crop=False)
        
                # Set the input of the network
                net.setInput(blob)
                layersNames = net.getLayerNames()
                outputNames = [(layersNames[i - 1]) for i in net.getUnconnectedOutLayers()]
                # Feed data to the network
                outputs = net.forward(outputNames)
    
                # Find the objects from the network output
                postProcess(outputs,img)

  

        # Show the frames
                cv2.imshow('Output', img)
        else:
            break
        frame_cnt = frame_cnt + 1

        if cv2.waitKey(1) == ord('q'):
            break

    # plotting the heatmap
    hm = sn.heatmap(data = heat_map_reference)
    
    heatmapshow = None
    heatmapshow = cv2.normalize(heat_map_reference, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
    cv2.imshow('heatmapshow', heatmapshow)
    super_imposed_img = cv2.addWeighted(heatmapshow, 0.8, sample_frame, 0.2, 0)
    cv2.imwrite('cv2_imposed.jpg', super_imposed_img)
    cv2.imshow('super_imposed_img', super_imposed_img)
    cv2.waitKey(0)
      
    plt.savefig('testplot.png')
    # displaying the plotted heatmap
    plt.show()

	
    # Finally realese the capture object and destroy all active windows
    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    realTime()

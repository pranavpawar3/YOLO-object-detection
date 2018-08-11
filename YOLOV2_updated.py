from darkflow.net.build import TFNet
import cv2
import numpy as np
import time

# Configure the model file and weights

options ={
    'model':'cfg/tiny-yolo-voc.cfg',
    'load':'bin/yolov2-tiny-voc.weights',
    'threshold':0.33,
    'gpu' : 1.0
}

#Loading the model.

tfnet= TFNet(options)

# If you wish to perform object detection over images, please uncomment the following code

"""img=cv2.imread('image.png',cv2.IMREAD_COLOR)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
result = tfnet.return_predict(img)
result
length=len(result)




e1 = cv2.getTickCount()
for i in range(0,length):

    tl= (result[i]['topleft']['x'],result[i]['topleft']['y'])
    br=(result[i]['bottomright']['x'],result[i]['bottomright']['y'])
    label= result[i]['label']
    img= cv2.rectangle(img, tl,br,(0,255.0),7)
    img=cv2.putText( img,label,tl,cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
e2 = cv2.getTickCount()
time = (e2-e1)/(cv2.getTickFrequency())
print("One image processed in %f seconds",time)
e3 = cv2.getTickCount()
plt.imshow(img)
plt.show()
e4 = cv2.getTickCount()
time2 = (e4-e3)/(cv2.getTickFrequency())
print("One image plotted in %f seconds",time2)"""





# if you wish to continue the object detection over videos , please provide video_path, and to perform it over webcam just pass 0 as an argument. 

capture =cv2.VideoCapture(Video_Path)
colors = [tuple(255*np.random.rand(3)) for i in range(100)]
check = True
cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
while(check):
    #e1 = cv2.getTickCount()
    #stime = time.time()
    check,frame = capture.read()
    if check == False:
        capture.release()
        cv2.destroyAllWindows()
        break
    frame = np.asarray(frame)
    result = tfnet.return_predict(frame)

    
    person_counter = 0 #Here I have tried to count the number of persons in a particular frame.
    if check:
        for color,result_2 in zip(colors,result):
            tl= (result_2['topleft']['x'],result_2['topleft']['y'])
            br=(result_2['bottomright']['x'],result_2['bottomright']['y'])
            label= result_2['label']
            frame = cv2.rectangle(frame,tl,br,color,7)
            frame=cv2.putText( frame,label,tl,cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
            
        for color,result in zip(colors,result):
            label= result['label']
            if(label == "person"):
                person_counter+=1

        print("\n\n {} persons found ... \n".format(person_counter))
        cv2.imshow('frame',frame)
        #print('FPS {:.1f}'.format(1/(time.time()-stime)))
        if cv2.waitKey(1) == 27:
            capture.release()
            cv2.destroyAllWindows()
            break

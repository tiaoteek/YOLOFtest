import json
import cv2
import numpy as np
import random

def LoadValJson():
    with open('datasets/coco/annotations/instances_val2017.json', 'r') as f:
       datas=json.load(f)
       result={}
       for data in datas['images']:
           result[data['id']]=data['file_name']
       return result

def LoadInferResult():
     with open('output/yolof/R_50_C5_1x/inference/coco_instances_results.json', 'r') as f:
       datas=json.load(f)   
       image_id=[]
       label=[]
       box=[]
       score=[]
       for data in datas:
           image_id.append(data['image_id'])
           label.append(data['category_id'])
           box.append(data['bbox'])
           score.append(data['score'])
       return image_id,label,box,score


def visualise(className,thresold=0.5):
    cols={}
    for i in range(len(className)):
        cols[i+1]=(random.randint(0,255),120,random.randint(0,255))
        
    result=LoadValJson()
    image_ids,labels,boxes,scores=LoadInferResult()
    img = np.ones((1,1),dtype=np.uint8)
    cur_path=''
    for image_id,label,box,score in zip(image_ids,labels,boxes,scores):
        filePath="datasets/coco/val2017/{}".format(result[image_id])
        if(cur_path!=filePath):
            if(img.shape[0]!=1):
                cv2.imwrite('output/yolof/R_50_C5_1x/inference/{}'.format(cur_path.split('/')[-1]),img)
            cur_path=filePath
            img=cv2.imread(cur_path)
        if(float(score)>=thresold):
            cv2.rectangle(img, (int(box[0]),int(box[1])), (int(box[0])+int(box[2]),int(box[1])+int(box[3])), cols[int(label)], 2) 
            cv2.rectangle(img, (int(box[0]),int(box[1])), (int(box[0])+80,int(box[1])+20), cols[int(label)], -1)   
            cv2.putText(img, className[int(label)], (int(box[0]),int(box[1])+20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)
            


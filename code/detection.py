import torch, cv2, time,os

def det(args,det_queue,det_reid_queue):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"The current device is {device}")
    # Model loading and setting
    model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
    model.eval()
    model = model.to(device)
    model.conf = 0.3  # confidence threshold (0-1)
    model.iou = 0.3  # NMS IoU threshold (0-1)
    model.classes = [0]
    print('det:model init done')
    while True:
        print('det:waiting on main')
        video_folder = det_queue.get()
        if video_folder == 'Done':  #when all videos done
            det_reid_queue.put('Done')#pass the message
            break
        
        det_reid_queue.put('START '+video_folder)
        print('det: recv video',video_folder)
        for file in sorted(os.listdir(video_folder+'/img1/')):
            crop_list = []
            imgr = cv2.imread(video_folder+'/img1/'+file)
            imger = cv2.cvtColor(imgr, cv2.COLOR_BGR2RGB)
            results = model(imger)
            bbox = results.xyxy[0]
            for i in bbox:
                crop_data = {}
                img_crop = imgr[int(i[1]):int(i[3]),int(i[0]):int(i[2])]
                crop_data['image'] = img_crop #bgr crop of ith bbox
                crop_data['centre'] = (int((i[0]+i[2])/2),int((i[1]+i[3])/2))
                crop_data['coords'] = (int(i[0]),int(i[1]),int(i[2]),int(i[3]))  #tuple of coords: (xmin,ymin,xmax,ymax)- yolov5 coords in int
                crop_data['conf'] = float(i[4].cpu().numpy())  # det confidence
                crop_data['frame'] = file  # frame number 
                #print(img_crop.shape)
                crop_list.append(crop_data)
            det_reid_queue.put((imgr,crop_list)) #assuming all frames go out in order, no frame info sent to reid 
            #one message per frame
        det_reid_queue.put('video done')
        det_queue.put('Done')#ask main for another video file
    return 0

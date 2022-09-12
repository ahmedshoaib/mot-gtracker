
import os, time,sys
from tracker import Tracker
from tracker_graph import GTracker
import torch
import psutil
#https://kaiyangzhou.github.io/deep-person-reid/user_guide
from torchreid.utils import FeatureExtractor

#https://qr.ae/pv5zwo
#getting color for bounding boxes
import colorsys 
 
def HSVToRGB(h, s, v): 
 (r, g, b) = colorsys.hsv_to_rgb(h, s, v) 
 return (int(255*r), int(255*g), int(255*b)) 
 
def getDistinctColors(value): 
 huePartition = 1.0 / (50) #assuming a max of 50 agents at a  time.
 return (HSVToRGB(huePartition * value, 1.0, 1.0) ) 
            


def reid(args,reid_queue,det_reid_queue, reid_eval_queue,method):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"The current device is {device}")
    #https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO
    
    extractor = FeatureExtractor(
        model_name='osnet_ibn_x1_0',
        model_path=args.reid_folder+'/osnet_ibn_ms_d_c.pth.tar',
        device='cuda'
    )

    #model init
    while True:
        command = det_reid_queue.get()
        if type(command) is str: #if its a meta message
            reid_eval_queue.put(command)
            if command == 'Done':
                reid_eval_queue.put('Done')
                break
            if command == 'video done':
                reid_eval_queue.put('video done')
                continue
            if command[:5] == 'START':
                video_folder = command[6:]
                folder_name = video_folder.split('/')[-1]
                

                if method:
                    tracker = GTracker(video_folder)
                else:
                    tracker = Tracker(folder_name)
                #


                if not os.path.exists(folder_name+"_out"):
                    os.makedirs(folder_name+"_out")
                t0 = time.time()
                t1 = time.time()

                #intialize tracker object
                continue

        #else read crop list
        
        full_frame = command[0]
        command = command[1]
        
        image_list = []
        meta_list = []
        frame_id = int(command[0]['frame'].split('.')[0])
        for crop in command:
            #cv2.imwrite('crop_'+str(i)+crop['frame'],crop['image'])
            image_list.append(crop['image'])
            del crop['image']
            del crop['frame']
            meta_list.append(crop)
        del command
        feature_vector_list = extractor(image_list).cpu().numpy()
        del image_list

        #if frame_id >10:
        #    sys.exit(1)

        assignment_ids = tracker.crop_assign(feature_vector_list,meta_list,frame_id)

        '''
        for i,aid in enumerate(assignment_ids): #draw boxes based on assigned track ids
            box_color = getDistinctColors(aid)
            #print(box_color)
            xyxy = meta_list[i]['coords']
            #print(box_color,xyxy)
            cv2.rectangle(full_frame,(xyxy[0],xyxy[1]),(xyxy[2],xyxy[3]), box_color , 2  )
        cv2.imwrite(folder_name+'_out/'+str(frame_id)+'.jpg',full_frame)
        '''



        reid_eval_queue.put((frame_id,meta_list, assignment_ids))#assignment_ids)) #meta_list, assignment_ids))
        if frame_id%200 == 0:
            
            print('\nStats of frame:',frame_id)
            print('Number of Tracks',len(tracker.tracks.keys()))
            print('cpu : ',psutil.cpu_percent())
            #print('mem: ',psutil.virtual_memory())  # physical memory usage
            print('memory % used:', psutil.virtual_memory()[2])
            print('time from last out: ',time.time()-t1)
            print('time from start: ',time.time()-t0)
            t1 = time.time()












        
        
        
                


        




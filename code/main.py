import time
import argparse,psutil
import os
from multiprocessing import Process, Queue
import detection, reid,eval, monitor

# Command line params settings
parser = argparse.ArgumentParser(description='MOT python')
parser.add_argument('--data', type=str, default='./data', metavar='D',
                    help='dataset folder MOT format')
parser.add_argument('--det_folder', type=str, default='./models/det', metavar='det',
                    help='detection network folder')
parser.add_argument('--reid_folder', type=str, default='./models/reid', metavar='reid',
                    help='ReID network folder')

def main():
    global args
    args = parser.parse_args()
    
    #init queues for communication b/n Main process and detector/reid processes
    det_queue = Queue() #main - det : contains information of which video to start processing
    reid_queue = Queue()  #reid-main : returns detections and track information for all dets in each frame for eval
    det_reid_queue = Queue() #reid-det  : det sends detection crops to reid
    eval_queue = Queue()
    reid_eval_queue = Queue()

    #init det and reid processes
    print('processes starting')
    stat_monitor = Process(target= monitor.monitor,args=(args, det_queue,reid_queue,det_reid_queue, reid_eval_queue,eval_queue  ))
    stat_monitor.start()

    #start processing dataset. Reading video files
    for i in range(2):
        if os.path.exists(args.data):
            for video_folder in os.listdir(args.data):
                print('waiting on det')
                time.sleep(20)
                print('putting on det')
                det_queue.put(args.data+'/'+video_folder)
                t1 = time.time()
                print('start_time:',t1)
                print('main:Sent video to det')
                #reid_queue.put(args.data+'/train/'+video_folder)
                eval_queue.put(args.data+'/'+video_folder)

                det_status =  det_queue.get()
                if det_status == 'Done':
                    video_stats = eval_queue.get()
                    print('Stats for video: ',video_folder)
                    print(video_stats)
                    print('End time:',time.time()-t1)
        det_queue.put('Done')

    stat_monitor.join()
    return



if __name__ == '__main__':
    main()

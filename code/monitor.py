import psutil, time
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue
import detection, reid,eval, monitor

def monitor(args,det_queue,reid_queue,det_reid_queue, reid_eval_queue,eval_queue  ):

    det_cpu_percents = [[],[]]
    reid_cpu_percents = [[],[]]
    eval_cpu_percents = [[],[]]
    det_mem_percents = [[],[]]
    reid_mem_percents = [[],[]]
    eval_mem_percents = [[],[]]

    for i in range(2):

        det_process = Process(target=detection.det, args=(args,det_queue,det_reid_queue))
        reid_process = Process(target=reid.reid, args=(args,reid_queue,det_reid_queue, reid_eval_queue,i))
        eval_process = Process(target=eval.eval, args=(args,eval_queue,reid_eval_queue))

        det_process.start()
        reid_process.start()
        eval_process.start()



        print('...................monitor started',eval_process.is_alive())
        det_pid = psutil.Process(det_process.pid)
        reid_pid = psutil.Process(reid_process.pid)
        eval_pid = psutil.Process(eval_process.pid)
        
        while eval_process.is_alive(): #since this is the last process to die
            #print('...................looping',det_pid.cpu_percent())
            det_cpu_percents[i].append(det_pid.cpu_percent())
            reid_cpu_percents[i].append(reid_pid.cpu_percent())
            eval_cpu_percents[i].append(eval_pid.cpu_percent())
            det_mem_percents[i].append(det_pid.memory_percent())
            reid_mem_percents[i].append(reid_pid.memory_percent())
            eval_mem_percents[i].append(eval_pid.memory_percent())
            time.sleep(1)
        

        det_process.join()
        print('closed det')
        reid_process.join()
        print('closed reid')
        eval_process.join()
        print('closed eval')
        #print(det_cpu_percents,det_mem_percents)


    plt.plot(det_cpu_percents[0],label='Baseline')
    plt.plot(det_cpu_percents[1],label='GTracker')
    #plt.plot(eval_cpu_percents,label='Eval')
    plt.ylabel('CPU usage')
    plt.xlabel('time')
    plt.legend(loc="upper left")
    plt.savefig('cpu_det.png')
    plt.clf()

    plt.plot(reid_cpu_percents[0],label='Baseline')
    plt.plot(reid_cpu_percents[1],label='GTracker')
    #plt.plot(eval_cpu_percents,label='Eval')
    plt.ylabel('CPU usage')
    plt.xlabel('time')
    plt.legend(loc="upper left")
    plt.savefig('cpu_reid.png')
    plt.clf()

    plt.plot(det_mem_percents[0],label='Baseline')
    plt.plot(det_mem_percents[1],label='GTracker')
    #plt.plot(eval_cpu_percents,label='Eval')
    plt.ylabel('Mem usage')
    plt.xlabel('time')
    plt.legend(loc="upper left")
    plt.savefig('mem_det.png')
    plt.clf()

    plt.plot(reid_mem_percents[0],label='Baseline')
    plt.plot(reid_mem_percents[1],label='GTracker')
    #plt.plot(eval_cpu_percents,label='Eval')
    plt.ylabel('Mem usage')
    plt.xlabel('time')
    plt.legend(loc="upper left")
    plt.savefig('mem_reid.png')
    plt.clf()

    return

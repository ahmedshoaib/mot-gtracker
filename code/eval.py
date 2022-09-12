import motmetrics as mm
import pandas as pd
import numpy as np,time

def eval(args,eval_queue,reid_eval_queue):
    
    while True:
        command = reid_eval_queue.get()
        if type(command) is str: #if its a meta message
            reid_eval_queue.get(command)
            if command == 'Done':
                break
            if command == 'video done':
                mh = mm.metrics.create()
                print(mh.compute(acc, metrics=['num_frames', 'mota', 'motp'], name='acc'))
                summary = mh.compute_many(
                    [acc, acc.events.loc[0:1]],
                    metrics=mm.metrics.motchallenge_metrics,
                    names=['full', 'part'])

                strsummary = mm.io.render_summary(
                    summary,
                    formatters=mh.formatters,
                    namemap=mm.io.motchallenge_metric_names
                )
                #print(summary)



                eval_queue.put(strsummary)
                continue
            if command[:5] == 'START':
                video_folder = eval_queue.get()
                folder_name = command[6:]
                acc = mm.MOTAccumulator(auto_id=True)
                gt_file = video_folder+'/gt/gt.txt'
                main_df = pd.read_csv(gt_file,names=["frame_id", "agent_id", "bb_left", "bb_top","bb_width","bb_height",'conf','class','vis'])
                df_clean = main_df[((main_df['class'] == 1) | (main_df['class'] == 7)) & (main_df['conf'] == 1)  & (main_df['vis'] > 0.5) ]
                del main_df
                #intialize tracker object
                continue

        #else eval
        #resolve frame's agent ids
        t1 =time.time()
        frame_id = command[0]
        meta_list = command[1]
        assigned_ids = command[2]
        infer_bbox = []
        for i,d in enumerate(meta_list):
            infer_bbox.append([d['coords'][0],d['coords'][1],d['coords'][2]-d['coords'][0],d['coords'][3]-d['coords'][1]])
        infer_bbox = np.array(infer_bbox)
        df1 = df_clean[df_clean['frame_id']==frame_id]
        gt_bbox = df1[['bb_left','bb_top','bb_width','bb_height']].values
        dist = mm.distances.iou_matrix(gt_bbox, infer_bbox, max_iou=0.5)
        gt_agentid = list(df1['agent_id'].values)
        acc.update(gt_agentid,assigned_ids,list(dist))
        if frame_id%400 == 0:
            print('frame eval time:',time.time()-t1)


        #get ground truth assignments







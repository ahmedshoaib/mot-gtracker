import os
from scipy import spatial
from scipy.optimize import linear_sum_assignment
import numpy as np

'''
          crop1    crop2    crop3   crop4
track 1
track 2
track 3

trackid : {
    'average_fv' : np.array(512,)
    'last_frame_id' : 123
    'last_location' : (12,34,14,35)
    'missing_counter' : 123
    'history' : [
        {
            'fv' : np.array(512,)
            'frame_id' : 123
            'location' : (12,34,14,35)
            'confidence' : 98


        }]}

'''


class Tracker:
    #tracker class 
    def __init__(self,id):
        self.video_name = id
        self.tracks = dict()
        self.last_track_id = 0  # incremental track id values
        self.dist_threshold = 0.4
        #self.dead_tracks = dict()


    def new_track(self,fv,meta,frame_id):
        #can check dead tracks first

        self.last_track_id += 1
        self.tracks[self.last_track_id] = dict()
        self.tracks[self.last_track_id]['average_fv'] = fv
        self.tracks[self.last_track_id]['last_frame_id'] = frame_id
        self.tracks[self.last_track_id]['last_location'] = meta['coords']
        self.tracks[self.last_track_id]['missing_counter'] = 0
        self.tracks[self.last_track_id]['history'] = []
        self.tracks[self.last_track_id]['history'].append({'fv':fv,'frame_id':frame_id,'location':meta['coords'],'confidence':meta['conf']})
        return self.last_track_id


    def add_to_track(self,track_id,fv,meta,frame_id):
        #print(track_id,fv,meta,frame_id)
        #print('pre average score',spatial.distance.cosine(fv, self.tracks[track_id]['average_fv']))
        self.tracks[track_id]['average_fv'] = np.average( np.array([ self.tracks[track_id]['average_fv'], fv ]), axis=0, weights=[0.75, 0.25] )
        #print('post average score',spatial.distance.cosine(fv, self.tracks[track_id]['average_fv']))
        self.tracks[track_id]['last_frame_id'] = frame_id
        self.tracks[track_id]['last_location'] = meta['coords']
        self.tracks[track_id]['missing_counter'] = 0
        self.tracks[track_id]['history'].append({'fv':fv,'frame_id':frame_id,'location':meta['coords'],'confidence':meta['conf']})
        return track_id

    def distance_score(self,track_id, crop_meta, fv):
        cos_dist = spatial.distance.cosine(fv, self.tracks[track_id]['average_fv'])   #lower => closer
        #add other factors as well

        return cos_dist

    
    def crop_assign(self,feature_vector_list,meta_list,frame_id):
        #print('Assign1:',len(meta_list),frame_id,self.tracks.keys())
        assignment_ids = []
        if self.last_track_id == 0: #frist frame
            for i,_ in enumerate(meta_list):
                assignment_ids.append(self.new_track(feature_vector_list[i],meta_list[i],frame_id)) 
            return assignment_ids
        else:
            distance_matrix = []
            #build distance matrix
            for i,track in enumerate(self.tracks): 
                row_distance = []
                for j,crop_meta in enumerate(meta_list):
                    row_distance.append(self.distance_score(track,crop_meta,feature_vector_list[j]))
                distance_matrix.append(row_distance)
            #print('distance matrix:',distance_matrix)

            #assignment
            row_ind,col_ind = linear_sum_assignment(distance_matrix)
            
            
            assignment_ids = [0 for x in meta_list]
            
            #if no. of crops > n.o.tracks, new tracks to all unassigned tracks
            for crop_id in range(len(meta_list)):
                if crop_id not in col_ind:
                    assignment_ids[crop_id] = (self.new_track(feature_vector_list[crop_id],meta_list[crop_id],frame_id))
            


            #iterate through tracks now
            current_track_keys = list(self.tracks.keys())
            #if frame_id%400==0:
            #    print('tracker length:',len(current_track_keys))
            for i,track_id in enumerate(current_track_keys):
                #check if the track's been assigned some crop
                if i in row_ind:
                    assignment_index_linear = list(row_ind).index(i)
                    crop_id = col_ind[assignment_index_linear]
                    if distance_matrix[row_ind[assignment_index_linear]][col_ind[assignment_index_linear]] > self.dist_threshold:
                        #bad assignment => create new track
                        assignment_ids[crop_id] = self.new_track(feature_vector_list[crop_id],meta_list[crop_id],frame_id)
                    else:
                        assignment_ids[crop_id] = self.add_to_track(track_id,feature_vector_list[crop_id],meta_list[crop_id],frame_id)
                        #acceptable assignment => add crop to track
                    #del row_ind[assignment_index_linear]
                    #del col_ind[assignment_index_linear]
                else:
                    self.tracks[track_id]['missing_counter'] += 1
                    if self.tracks[track_id]['missing_counter']  > 400:
                        #send to dead_tracks{}
                        del self.tracks[track_id]
                #print('assignments:',row_ind,col_ind)
                #print(distance_matrix)

    


        return assignment_ids
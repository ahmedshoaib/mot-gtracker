import sys
from scipy import spatial
from scipy.optimize import linear_sum_assignment
import numpy as np
from anytree import NodeMixin, search
import networkx as nx
import itertools

import matplotlib.pyplot as plt
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
class GraphNode(NodeMixin):  # Add Node feature
    def __init__(self, name, x1,x2,x3,y1,y2,y3, parent=None, children=None):
        super(GraphNode, self).__init__()
        self.name = name
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.y1 = y1
        self.y2 = y2
        self.y3 = y3
        
        self.parent = parent
        if children:
            self.children = children
        


class GTracker:
    #tracker class 
    def __init__(self,id):
        self.video_folder = id
        with open(self.video_folder+"/seqinfo.ini", encoding = 'utf-8') as f:
            t = f.readlines()
        self.width = int(t[5][8:-1])
        self.height = int(t[6][9:-1])
        self.tracks = dict()
        self.last_track_id = 0  # incremental track id values
        self.dist_threshold = 0.6  #different from baseline. directly assigns if similarity >= dist_threshold
        
        self.normal_net = nx.DiGraph()  ####normalized weighted graph here -  read-only during runtime
        self.pop_net('root')
        
        

        self.running_net = nx.DiGraph()  #### running graph here
        self.running_net.add_nodes_from(self.normal_net)


        self.graph_root = GraphNode('root',0,self.width/2,self.width,0, self.height/2,self.height) ###weighted network here
        self.node_tracks = dict()  ###shows current tracks in a particular node
        self.populate_dict_keys()
        
        
        #self.dead_tracks = dict()
    
    def populate_dict_keys(self):
        for i in self.normal_net.nodes:
            self.node_tracks[i] = []

    def pop_graph(self,graph_new_node_name):
        parent_node = search.find(self.graph_root, filter_=lambda node: node.name == graph_new_node_name[:-2] )
        p = graph_new_node_name[-1]
        if p == '1':
            GraphNode(graph_new_node_name,parent_node.x1,(parent_node.x1 + parent_node.x2)/2,parent_node.x2,
                parent_node.y1, (parent_node.y1 + parent_node.y2)/2,parent_node.y2, parent=parent_node)
        elif p == '2':
            GraphNode(graph_new_node_name,parent_node.x2,(parent_node.x2 + parent_node.x3)/2,parent_node.x3,
                parent_node.y1, (parent_node.y1 + parent_node.y2)/2,parent_node.y2, parent=parent_node)
        elif p == '3':
            GraphNode(graph_new_node_name,parent_node.x1,(parent_node.x1 + parent_node.x2)/2,parent_node.x2,
                parent_node.y2, (parent_node.y2 + parent_node.y3)/2,parent_node.y3, parent=parent_node)
        elif p == '4':
            GraphNode(graph_new_node_name,parent_node.x2,(parent_node.x2 + parent_node.x3)/2,parent_node.x3,
                parent_node.y2, (parent_node.y2 + parent_node.y3)/2,parent_node.y3, parent=parent_node)
        else:
            print('Graph build failed')
            sys.exit(1)
        pass


        
        
    def pop_net(self,net_node_name):        
        self.normal_net.add_node(net_node_name+'p1')
        self.normal_net.add_node(net_node_name+'p2')
        self.normal_net.add_node(net_node_name+'p3')
        self.normal_net.add_node(net_node_name+'p4')
        lister = [net_node_name+'p1',net_node_name+'p2', net_node_name+'p3', net_node_name+'p4']
        edge_list = [ (pair[0],pair[1],{'weight':0.25}) for pair in itertools.product(lister, repeat=2)]
        self.normal_net.add_edges_from(edge_list)
        
    def rebuild_graphs(self):
        pass    

    def get_node_name(self,x,y):
        node = search.findall(self.graph_root, filter_=lambda node: ((node.x1 <= x <= node.x3) and (node.y1 <= y <= node.y3)))[-1]
        if node.x1 <= x <node.x2:
            if node.y1 <= y < node.y2:
                p='p1'
            else:
                p='p3'
        else:
            if node.y1 <= y < node.y2:
                p='p2'
            else:
                p='p4'
        return node.name+p


    def new_track(self,fv,meta,frame_id):
        #can check dead tracks first
        node = self.get_node_name(meta['centre'][0],meta['centre'][1])
        self.last_track_id += 1
        self.tracks[self.last_track_id] = dict()
        self.tracks[self.last_track_id]['average_fv'] = fv
        self.tracks[self.last_track_id]['last_frame_id'] = frame_id
        self.tracks[self.last_track_id]['last_location'] = meta['centre']
        self.tracks[self.last_track_id]['missing_counter'] = 0
        self.tracks[self.last_track_id]['history'] = []
        self.tracks[self.last_track_id]['history'].append({'fv':fv,'frame_id':frame_id,'location':meta['coords'],'confidence':meta['conf']})
        self.tracks[self.last_track_id]['last_node'] = node
        self.node_tracks[node].append(self.last_track_id)
        #print(self.node_tracks,node)
        return self.last_track_id


    def add_to_track(self,track_id,fv,meta,frame_id,node):
        #print(track_id,fv,meta,frame_id)
        #print('pre average score',spatial.distance.cosine(fv, self.tracks[track_id]['average_fv']))
        #print(self.node_tracks,node)
        self.tracks[track_id]['average_fv'] = np.mean( np.array([ self.tracks[track_id]['average_fv'], fv ]), axis=0 )
        #print('post average score',spatial.distance.cosine(fv, self.tracks[track_id]['average_fv']))
        self.tracks[track_id]['last_frame_id'] = frame_id
        self.tracks[track_id]['last_location'] = meta['centre']
        self.tracks[track_id]['missing_counter'] = 0
        self.tracks[track_id]['history'].append({'fv':fv,'frame_id':frame_id,'location':meta['coords'],'confidence':meta['conf']})
        old_node = self.tracks[track_id]['last_node']
        #print('old,new',old_node,node,track_id)
        if node != old_node:
            self.node_tracks[old_node].remove(track_id)
            self.node_tracks[node].append(track_id)
            self.tracks[track_id]['last_node'] = node
        if self.running_net.has_edge(old_node,node):

            self.running_net.edges[old_node,node]['weight'] += 1   #even if self loop
        else:
            self.running_net.add_edge(old_node,node , weight = 1)   #even if self loop
        return track_id

    def distance_score(self,track_id, crop_meta, fv):
        cos_dist = spatial.distance.cosine(fv, self.tracks[track_id]['average_fv'])   #lower => closer
        #add other factors as well

        return cos_dist

    
    def crop_assign(self,feature_vector_list,meta_list,frame_id):
        #print('Assign1:',len(meta_list),frame_id,self.tracks.keys())
        assignment_ids = []
        if self.last_track_id == 0: #first frame
            #print('first frame')
            for i,_ in enumerate(meta_list):
                assignment_ids.append(self.new_track(feature_vector_list[i],meta_list[i],frame_id)) 
            #print('Ass IDs :)',assignment_ids)
            #print(self.running_net.nodes,self.normal_net.nodes)
            '''subax1 = plt.subplot(111)

            nx.draw(self.normal_net, with_labels=True)
            plt.savefig("net"+str(frame_id)+".png")
            del subax1
            print('After rebuild:',self.node_tracks,list(nx.isolates(self.normal_net)))#self.normal_net.edges,self.normal_net.nodes)               '''

            return assignment_ids


            
        else:
            for i,crop_meta in enumerate(meta_list):
                #print('cont frames',frame_id)

                node = self.get_node_name(crop_meta['centre'][0],crop_meta['centre'][1])
                k = [(edge[0],self.normal_net.edges[edge[0],edge[1]]['weight']) for edge in self.normal_net.in_edges(node)]   # [('node1,12),(node2,11),....]  list of nodes to be searched in - incoming nodes to current node inorder
                #k = sorted(k, key=lambda x: x[1], reverse=True) #sort based on weight (high to low)
                #print('node, incoming nodes',node,k)
                best_match_track  = None
                best_match_score = 0

                for incoming_node,_ in k:  #iterate to find best match
                    for track in self.node_tracks[incoming_node]:
                        score = 1 - self.distance_score(track,crop_meta,feature_vector_list[i])
                        #print(track,score)
                        if score > best_match_score:
                            best_match_track = track
                            best_match_score = score
                #print('best',best_match_track,best_match_score)
                if best_match_score > self.dist_threshold:  #if best match is better than threshold - > assign to track
                    assignment_ids.append(self.add_to_track(best_match_track,feature_vector_list[i],meta_list[i],frame_id,node))
                    #print('adding to track')
                else:       #else make a new track
                    assignment_ids.append(self.new_track(feature_vector_list[i],meta_list[i],frame_id))
                    #print('new_track')
                #print('Ass IDs',assignment_ids)

            #iterate through tracks to mark dead ones
            current_track_keys = list(self.tracks.keys())
            #if frame_id%400==0:
            #    print('tracker length:',len(current_track_keys))#,frame_id,self.node_tracks.keys())
            for i,track_id in enumerate(current_track_keys):
                #check if the track's been assigned some crop for this frame
                if self.tracks[track_id]['last_frame_id'] != frame_id:
                    self.tracks[track_id]['missing_counter'] += 1
                    if self.tracks[track_id]['missing_counter']  > 400:
                        #send to dead_tracks{}
                        self.node_tracks[self.tracks[track_id]['last_node']].remove(track_id)
                        del self.tracks[track_id]
                        
                        #print('**********Deleting track',track_id)
                        
        
        
            #every n frames, rebuild graphs
            if frame_id%600==0:
                print('Rebuilding graphs')
                nodes_to_pop = []
                for node_name in self.running_net.nodes:
                    if not self.running_net.has_edge(node_name,node_name):
                        continue
                    total_in = self.running_net.in_degree(node_name,weight = 'weight')
                    self_in = self.running_net.edges[node_name,node_name]['weight']
                    if (self_in > total_in*0.8) and (self_in > 100) :
                        #print('self loop node:',node_name, self_in, total_in)
                        nodes_to_pop.append(node_name)
                for node_name in nodes_to_pop:
                    self.normal_net.remove_node(node_name)
                    self.pop_net(node_name)
                    self.pop_graph(node_name)
                    
                self.running_net = nx.DiGraph()  #### reset running graph here
                self.running_net = self.normal_net
                self.node_tracks = dict()   #reset node_track dictionary
                self.populate_dict_keys()

                #loop through tracks to assign weights

                current_track_keys = list(self.tracks.keys())
                for track_id in current_track_keys:
                    last_node = self.get_node_name(self.tracks[track_id]['last_location'][0],self.tracks[track_id]['last_location'][1])
                    self.node_tracks[last_node].append(track_id)
                    self.tracks[track_id]['last_node'] = last_node
                    for i,obs in enumerate(self.tracks[track_id]['history']):
                        if i==0:
                            continue
                        xyxy = self.tracks[track_id]['history'][i-1]['location']
                        previous_node = self.get_node_name(int((xyxy[0]+xyxy[2])/2),int((xyxy[1]+xyxy[3])/2))
                        current_node = self.get_node_name(int((obs['location'][0]+obs['location'][2])/2),int((obs['location'][1]+obs['location'][3])/2))
                        if self.normal_net.has_edge(previous_node,current_node):
                            self.normal_net.edges[previous_node,current_node]['weight'] += 1   #even if self loop
                        else:
                            self.normal_net.add_edge(previous_node,current_node , weight = 1)   #even if self loop
                        #self.normal_net.edges[previous_node,current_node]['weight'] += 1   #even if self loop

                for node_name in self.normal_net.nodes:   #normalize normal network
                    total_in = self.normal_net.in_degree(node_name,weight = 'weight')
                    k = [(edge[0],self.normal_net.edges[edge[0],edge[1]]['weight']) for edge in self.normal_net.in_edges(node_name)]
                    for n,w in k:
                        self.normal_net.edges[n,node_name]['weight'] = w/total_in
                '''print('After rebuild:',self.node_tracks,nx.number_connected_components(nx.to_undirected(self.normal_net)))#self.normal_net.edges,self.normal_net.nodes)               
                subax1 = plt.subplot(111)

                nx.draw(self.normal_net, with_labels=True)
                plt.savefig("net"+str(frame_id)+".png")
                del subax1'''



                        
        return assignment_ids


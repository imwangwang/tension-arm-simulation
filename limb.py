# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 18:15:01 2017

@author: Rohit
"""

# -*- coding: utf-8 -*-
"""
@author: Rohit
"""
from random import randint
from vpython import *
import numpy as np
import pandas as pd
import datetime
from pandas.io.common import EmptyDataError 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from ga_class_1 import GenticAlgorithm

"""
Custom Classes
"""
    
class GUI(object):
    def __init__(self,model):
        self.__angles = model['angles']
        self.__limits = model['limits']
        self.__s_id = []
        self.__b_id = []
        self.__claw = 0 #open
        self.__state = True #running
    
    def s_callback(self,s):
        i = s.id
        u = self.__limits['u'][i]
        l = self.__limits['l'][i]
        self.__angles[i] = self.__s_id[i].value*(u-l) + l
        
    def b_callback(self,b):
        if b.id == 0:
            if self.__state == True:
                self.__b_id[0].text ='<b>Run</b>' 
            else:
                self.__b_id[0].text ='<b>Pause</b>' 
            self.__state = not self.__state
        else:
            if self.__claw == 0:
                self.__b_id[1].text = '<b>Release</b>'
                self.__claw = 1
            else:
                self.__b_id[1].text = '<b>Grab</b>'
                self.__claw = 0
            
    def build(self):
        self.__b_id.append(button(text='<b>Pause</b>',bind=self.b_callback, id=0))
        scene.append_to_caption('\t')
        self.__b_id.append(button(text='<b>Grab</b>',bind=self.b_callback, id=1))
        scene.append_to_caption('\n\n')
        for i in range(len(self.__angles)):
            caption = '\tJoint ' + str(i+1)
            self.__s_id.append( slider(length=250, bind = self.s_callback, id=i))
            scene.append_to_caption(caption + '\n\n')
   
    def read_values(self):
        return self.__angles, self.__claw

    def read_state(self):
        return self.__state
    
class Arm_segment(object):
    def __init__(self, pos, length):
        self.pos = pos
        self.l = length
    """
    Generate a single arm segement (ball+cylinder)
    Needed to be run during model building/initialization
    """
    def generate(self,angle,phi,pos=None):
        new_pos, axis = self.calculate_pos(angle,phi,pos)
        arm = cylinder(pos=self.pos,axis=axis,radius=0.5)
        joint = sphere(pos=self.pos,radius=0.7,color=color.cyan)
        limb = {'arm':arm, 'joint':joint}
        return limb, new_pos
    
    """
    Calulate end points of arm segments
    """
    def calculate_pos(self,angle,phi,pos=None):
        new_pos = vector(0,0,0)
        if pos != None:
            self.pos = pos
        angle = np.radians(angle)
        new_pos.x = self.pos.x + self.l*np.cos(angle)*np.cos(np.radians(phi))
        new_pos.z = self.pos.z + self.l*np.cos(angle)*np.sin(np.radians(phi))
        new_pos.y = self.pos.y + (self.l*np.sin(angle))
        axis = self.l * (new_pos - self.pos).norm() 
        return new_pos, axis

"""
Dataset creation
"""
class Dataset(object):
    
    def __init__(self,filename,cols=None):
        self.__dataset = None
        self.__cols = cols
        if cols == None:
            self.__cols =  ['Date'] + [str(i) for i in range(75)]
        self.__filename = filename
        try:
            self.__dataset = pd.read_csv(self.__filename)
            self.__cols = self.__dataset.columns
        except (EmptyDataError, FileNotFoundError)  : 
            self.__dataset = pd.DataFrame(columns=self.__cols)   
    
    def update_dataset(self,new_data):
        data_row = dict()
        for i in range(len(new_data)):
            data_row.update({self.__cols[i]: new_data[i]})
        self.__dataset = self.__dataset.append(data_row, ignore_index=True)
    
    def save_dataset(self):
        self.__dataset = self.__dataset.set_index(self.__dataset.columns[0])
        self.__dataset.to_csv(self.__filename)
"""
Setup vpython scene, camera position
create axes and colors
"""

def create_scene(height=None):
    if height==None:
        scene.height = 600
    else:
        scene.height = height
    scene.width = 800
    scene.forward = vector(0.052912, -0.738750, -1.836577)
    size = 30
    x_axis = cylinder(pos=vector(0,0,0),axis=vector(size,0,0),radius=0.05, color=color.red)
    L_x_axis = label(pos=x_axis.axis,text='x',xoffset=1,yoffset=1, box=False)
    y_axis = cylinder(pos=vector(0,0,0),axis=vector(0,size,0),radius=0.05, color=color.red)
    L_y_axis = label(pos=y_axis.axis,text='y',xoffset=1,yoffset=1, box=False)
    z_axis = cylinder(pos=vector(0,0,0),axis=vector(0,0,size),radius=0.05, color=color.red)
    L_z_axis = label(pos=z_axis.axis,text='z',xoffset=1,yoffset=1, box=False)
    base = box(pos=vector(0,0,0), length=size, height=0.1, width=size) 
    base.color = color.green
    base.opacity = 0.4
    return
   
"""
Calulates angles with refernce to x-z axis
"""     
def update_angles(angles,limits):
    for i in range(len(angles)):
        if angles[i] < limits['l'][i] :
            angles[i] = limits['l'][i]
        elif angles[i] > limits['u'][i]:
            angles[i] = limits['u'][i]
        else:
            pass
    total_angles = [angles[0],angles[1]]
    for i in range(2,len(angles)):
        total_angles.append(angles[i] + total_angles[i-1])
    return total_angles

"""
Repositions arm segments
"""   
def update(model):
    seg, arm, angles, pos = model['seg'],model['arm'],model['angles'],model['pos']
    limits = model['limits']
    total_angles = update_angles(angles,limits)
    claw_state = model['claw_state']
    joint = model['joint']
    for i in range(len(arm)):
        arm[i].pos = pos[i]
        joint[i].pos = pos[i]
        new_pos, axis = seg[i].calculate_pos(total_angles[i+1],phi=total_angles[0],pos=pos[i])
        arm[i].axis = axis
        pos[i+1] = new_pos
    joint[-1].pos = new_pos
    if claw_state == 1:
        joint[-1].color = color.red
    else:
        joint[-1].color = color.blue
    return pos[-1]    

"""
Build/Initialize the arm
Creates a dictionary with arm properties
length, limits, current position of arm segments
current angles, claw/grabber state
"""
       
def init_model(lengths=None, limits=None):
    if lengths == None:
        lengths = [3,6,6]
    if limits == None:
        limits = {'l':[0,0,0],'u':[1,180,180]}
    angles = [0,45,30]
    total_angles = update_angles(angles,limits)
    """
    Base joint creations
    """
    a = cylinder(pos=vector(0,0,0),axis=vector(0,1,0),length=lengths[0],radius=0.5,color=color.white)
    a1_joint = sphere(pos=vector(0,0,0),radius=0.7,color=color.cyan)  
    pos=[a.axis]
    seg = []
    arm = []
    joint = []
    j = 0
    for i in range(1,len(total_angles)):
        seg.append(Arm_segment(pos=pos[j],length=lengths[i]))
        limb, new_pos = seg[j].generate(total_angles[i],phi=total_angles[0])
        pos.append(new_pos)
        arm.append(limb['arm'])
        joint.append(limb['joint'])
        j+=1
    joint.append(sphere(pos=pos[-1],radius=0.7,color=color.blue))
    model = {'pos':pos, 'seg':seg, 'arm':arm,'joint':joint,'angles':angles}
    model.update({'lengths':lengths,'limits':limits, 'claw_state':0})
    return model

"""
Testing model movement
"""
def rand_move_model(model):
    limits = model['limits']
    angles = model['angles']
    for i in range(len(angles)):
        if angles[i] < limits['l'][i] :
            angles[i] += randint(0,5)
        elif angles[i] > limits['u'][i]:
            angles[i] -= randint(0,5)
        else:
            angles[i] += randint(-5,5)
    model['angles'] = angles
    return model
  

"""
create specific path
Generates random target location using lengths of arm segment
"""
class TargetGenerator(object):
    def __init__(self,model,path='circle'):
        self.__total_l = sum(model['lengths'])
        self.__path = path
        pos = vector(0,0,0)
        self.__target = sphere(pos=pos,radius=0.5,color=color.red)
        self.__target.visible = False
        self.__curr_iter = 0
    
    def __move_target(self):
        r = 2
        cx, cy = 0, 9
        angle = np.radians(self.__curr_iter)
        self.__target.pos.x = r*np.cos(angle) + cx
        self.__target.pos.y = r*np.sin(angle) + cy
        self.__target.pos.z = 0 
        self.__curr_iter += 1
        if self.__curr_iter > 360:
            self.__curr_iter = 0
        
    def __random_target(self):
        pos = vector(0,0,0)
        max_range = int(0.8*self.__total_l )
        pos.x = randint(int(0.1*self.__total_l ),max_range)
        pos.y = randint(int(0.1*self.__total_l ),max_range)
        pos.z = 0
        self.__target.pos = pos
        
    def get_target(self):
        if self.__path == 'circle':
            self.__move_target()
        elif self.__path == 'random':
            self.__random_target()
        self.__target.visible = True
        return self.__target
        
def create_target_path(target=None):
    pos = vector(0,0,0)
    
    if(target==None):
        target = sphere(pos=pos,radius=0.75,color=color.red)
    else:
        target.pos=pos
        
    return target


"""
Create a dataset using random angles
"""
def generate_random_angles(model):
    limits = model['limits']
    angles = model['angles']
    for i in range(len(angles)):
        angles[i] = randint(limits['l'][i],limits['u'][i])
        
    return angles

def generate_dataset(size=None):
    if size==None:
        size = 5000
    create_scene(400)
    model = init_model()
    model.update({'reach':vector(0,0,0)})
    data = Dataset()
    for i in range(size):
        model['angles'] = generate_random_angles(model)
        reach = update(model)
        data.update_dataset(model,reach)
    data.save_dataset()    
    return

"""
visualization
"""

def vis(cam_enable=False):
    create_scene(400)
    model = init_model()
    model.update({'reach':vector(0,0,0)})
    gui = GUI(model)
    gui.build()
    
    while(True):
        if(gui.read_state()):
            
            model['angles'], _= gui.read_values()
            model['reach'] = update(model)
        rate(200)
    return

"""
Tension model simulation via Gentic Algorithm
"""
"""
http://robotics.usc.edu/~aatrash/cs545/CS545_Lecture_9.pdf
"""
def analytic_soln(model,x,y):
    l0,l1,l2 = model['lengths']
    y = y-l0
    l = np.sqrt((x*x) + (y*y))
    angle1 = np.arctan2(y,x) - np.arccos(((l*l) + (l1*l1) - (l2*l2))/(2*l1*l))
    angle2 =  np.arctan2((y - l1*np.sin(angle1)),(x - l1*np.cos(angle1))) - angle1
    return list(np.degrees([angle1,angle2]))

def tension_to_angles(tension):
    R = np.array([[-2,0],[0,-1.5]])
    pred_angles = np.dot(R,np.array(tension))
    return list(pred_angles)

def ga_sim(save_file='plot_data.csv'):
    create_scene(400)
    model = init_model()
    model.update({'reach':vector(0,0,0)})
    
    ga_model = GenticAlgorithm()
    tension = ga_model.constrained_individual()
    target_gen = TargetGenerator(model,'circle')
    
    #Data to be saved and plotted
    new_data = []
    
    cols = ['Index','Target Position x','Target Position y','Pred Position x','Pred Position y']
    cols += ['Target Angle q1','Target Angle q2','Pred Angle q1','Pred Angle q2']
    cols += ['Excursion s{}'.format(i) for i in range(1,3)] + ['Time']
    data = Dataset(save_file,cols)
    
    i=0    
    while(i < 360):
        target = target_gen.get_target()
        target_angles = analytic_soln(model,target.pos.x,target.pos.y)
        start_time = datetime.datetime.now()
        tension = ga_model.run(tension,target_angles)
        stop_time = datetime.datetime.now()
        pred_angles = tension_to_angles(tension)
        model['angles'] = [0] + pred_angles
        model['reach'] = update(model)
        
        time_diff = (stop_time - start_time).total_seconds()
        print('target {} prediction {}'.format(target_angles,pred_angles))
        
#        error = np.linalg(mode)
        
        new_data =[i]+[target.pos.x,target.pos.y]+[model['reach'].x,model['reach'].y]
        new_data += target_angles+pred_angles
        new_data += tension+[time_diff]
        data.update_dataset(new_data)
        i+=1
        print('Iteration {} Time taken = {} '.format(i,time_diff))
    
    data.save_dataset()


def load_plot(filename = 'plot_data.csv'):
    
    dataset = pd.read_csv(filename)
    target_position_cols = [col for col in dataset.columns if 'Target Position' in col]
    pred_position_cols = [col for col in dataset.columns if 'Pred Position' in col]
    target_angle_cols = [col for col in dataset.columns if 'Target Angle' in col]
    pred_angle_cols = [col for col in dataset.columns if 'Pred Angle' in col]
    excursion_cols = [col for col in dataset.columns if 'Excursion' in col]
    time_cols = [col for col in dataset.columns if 'Time' in col]
    target_positions = dataset[target_position_cols].values
    pred_positions = dataset[pred_position_cols].values
    target_angles = dataset[target_angle_cols].values
    pred_angles = dataset[pred_angle_cols].values
    excursion = dataset[excursion_cols].values
    time_taken = dataset[time_cols].values
    print(dataset.describe())
    graphing(target_positions,pred_positions,target_angles,pred_angles,excursion,time_taken)

def graphing(target_positions,pred_positions,target_angles,pred_angles,excursion,time_taken):
    target_positions = np.array(target_positions)
    pred_positions = np.array(pred_positions)
    target_angles = np.array(target_angles)
    pred_angles = np.array(pred_angles)
    excursion = np.array(excursion)
    time_taken = np.array(time_taken)
    
    #Plotting graphs
    fig = plt.figure(1)
    fig.suptitle('Positions')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    x1 = target_positions[:,0]
    y1 = target_positions[:,1]
    x2 = pred_positions[:,0]
    y2 = pred_positions[:,1]
    plt.plot(x1,y1,label='Target positions')
    plt.plot(x2,y2,label='Predition positions')
    plt.legend()
    
    fig = plt.figure(2)
    fig.suptitle('Joint Angles')
    plt.xlabel('Angle 1 (q1)')
    plt.ylabel('Angle 2 (q2)')
    x1 = target_angles[:,0]
    y1 = target_angles[:,1]
    x2 = pred_angles[:,0]
    y2 = pred_angles[:,1]
    plt.plot(x1,y1,label='Target angles')
    plt.plot(x2,y2,label='Predition angles')
    plt.legend()
    
    fig = plt.figure(3)
    ax = fig.gca(projection='3d')
    fig.suptitle('Excursion Values')
    ax.set_xlabel('S1')
    ax.set_ylabel('S2')
#    ax.set_zlabel('S3')
    # s3 = 
    x = excursion[:,0]
    y = excursion[:,1]
#    z = excursion[:,2]
    ax.plot(x,y)
    
    fig = plt.figure(4)
    fig.suptitle('Time taken per iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Time taken')
    x = [i for i in range(time_taken.shape[0])]
    y = time_taken
    plt.plot(x,y)
    
    plt.show()
    

"""
main function
"""
def main():
#    vis()
#    generate_dataset(10000)
#    ga_sim(save_file = 'plot_data3.csv')
    load_plot(filename = 'plot_data3.csv')
    
if __name__ == '__main__':
    main()
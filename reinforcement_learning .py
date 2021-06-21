# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 11:25:32 2021

@author: OPTLAB
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import gc 
import cv2
import sys
import time
import h5py
import slmpy

import numpy as np

from pypylon import pylon
from pypylon import genicam
from datetime import datetime
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn import Linear
import pynvml

def GPU_check():
    """ check if GPU is available or not.
        if not set 'device=CPU'',
        else set 'device=GPU' and print the current information of GPU"""
        
    print('torch version:',torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('available device:',device)
    print('\n')
    if device==torch.device('cuda'):
        GPU_num=torch.cuda.device_count()
        for i in range (GPU_num):
            print('-----------GPU',str(i),':',torch.cuda.get_device_name(i),'-----------')
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            ratio = 1024**3
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total = meminfo.total/ratio
            used = meminfo.used/ratio
            free = meminfo.free/ratio
            print("total memory: ", total,'GB')
            print("used memory: ", used,'GB')
            print("free memory: ", free,'GB')
            print('\n')
            pynvml.nvmlShutdown()
    else :
        print('No GPU is available!, use cpu')
    
    return device

def image_resize(data,resized_shape):
    
    def _resize(data,resized_shape):
        # resize_data=np.resize(data,resized_shape)
        resize_data=np.zeros(resized_shape)
        data_shape=np.shape(data)
        
        size=int(resized_shape[0]/data_shape[0])
        for i in range(data_shape[0]):
            for j in range(data_shape[1]):
                resize_data[i*size:(i+1)*size,j*size:(j+1)*size]=data[i,j]
        return resize_data
    data_shape=np.shape(data)
    if len(data_shape)==3:
        resized_data=np.zeros((data_shape[0],)+resized_shape)
        for i in range(data_shape[0]):
            resized_data[i]=_resize(data[i],resized_shape)
    if len(data_shape)==2:
        resized_data=np.zeros(resized_shape)
        resized_data=_resize(data,resized_shape)
    return resized_data
def total_normalization( data,ymin,ymax ):
    xmax=np.max(data)
    xmin=np.min(data)
    data=(ymax-ymin)*(data-xmin)/(xmax-xmin) + ymin
    return data

class Basler_Camera(pylon.ConfigurationEventHandler):
    
    def __init__(self,camera_name,if_Hardware_trigger,Width,Height,OffsetX,OffsetY,ExposureTime,PixelFormat):
        self.camera_name=camera_name
        self.if_Hardware_trigger = if_Hardware_trigger
        self.Width = Width
        self.Height = Height
        self.OffsetX = OffsetX
        self.OffsetY = OffsetY
        self.ExposureTime = ExposureTime
        self.PixelFormat = PixelFormat
    
    def Config(self):
        tl_factory = pylon.TlFactory.GetInstance()
        devices = tl_factory.EnumerateDevices()
        print('find devices num:',len(devices))
        
        name_list=[]
        for i,device in enumerate(devices):
            print(i,':',device.GetFriendlyName())
            name_list.append(device.GetFriendlyName())
        
        if self.camera_name in name_list:
            index=name_list.index(self.camera_name)
            print("Using device:",self.camera_name)
            camera= pylon.InstantCamera()
            camera.Attach(tl_factory.CreateDevice(devices[index]))
            print('Camera Attached:',camera.IsPylonDeviceAttached())
            camera.Open()
            print('Camera Opened:',camera.IsOpen())
            print('Camera max size:','height:',camera.HeightMax.GetValue(),'width:',camera.WidthMax.GetValue())
        else:
            raise Exception("wrong camera name")
        camera.Width = self.Width
        camera.Height = self.Height
        if genicam.IsWritable(camera.OffsetX):
            camera.OffsetX.SetValue(self.OffsetX)
        if genicam.IsWritable(camera.OffsetY):
            camera.OffsetY.SetValue(self.OffsetY)
        camera.ExposureTime.SetValue(self.ExposureTime)
        camera.PixelFormat = self.PixelFormat
        camera.AcquisitionFrameRateEnable.SetValue(False)
        print('allow camera max AcquisitionFrameRate:',camera.ResultingFrameRate.GetValue())
        
        # set hardware trigger camera parameters
        camera.MaxNumBuffer = 5 #count of buffers allocated for grabbing
        if self.if_Hardware_trigger:
            camera.TriggerSelector.SetValue('FrameStart')
            camera.TriggerMode.SetValue('On') #hardware trigger
            camera.TriggerSource.SetValue('Line1')
            camera.TriggerActivation.SetValue('RisingEdge')
            # print(is Hardware trigger setup of camera complete?)
            print("Hardware trigger setup of camera ", self.camera_name, " complete.")
            
        return camera
    def Close(self,camera):
        if self.if_Hardware_trigger:
            camera.TriggerMode.SetValue('Off')
            camera.AcquisitionMode.SetValue('Continuous')
            camera.TriggerSelector.SetValue('FrameStart')
        camera.Close()


def makedir(path):
    isExists=os.path.exists(path)
    # 判断结果
    if not isExists:
    
        os.makedirs(path) 
     
        print (path+' 创建成功')
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print (path+' 目录已存在')
        
class FCL(nn.Module):
    def __init__(self,in_size,out_size,bias,activation_functuion):
        super(FCL, self).__init__()
        self.loss_func = torch.nn.MSELoss() 
        self.fc=torch.nn.Linear(in_size,out_size,bias = bias)
        self.Sigmoid= torch.nn.Sigmoid()
        self.activation_functuion=activation_functuion
    def forward(self,x):
        x_out= self.fc(x)
        if self.activation_functuion:
            x_out=self.Sigmoid(x_out)
        return x_out

def train_forward_model(model,model_path,epochs,lr,batch_size,is_log,bias,activation_functuion,x_path,y_path):
    
    print('-----------start train forward model-----------')
    x=np.load(x_path)[0:desired_num]
    x=total_normalization(x,0,1).reshape(-1,image_size**2).astype('float32')
    

    y=np.load(y_path)
    y=total_normalization(y,0,1).reshape(-1,speckle_size**2).astype('float32')
    
    
    
    model.to(device)
    optimizer= torch.optim.Adam(model.parameters(), lr=lr)
    loss_func= torch.nn.MSELoss(reduction='mean')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',verbose=True,patience=5)
    dataset=torch.utils.data.TensorDataset(torch.from_numpy(x),torch.from_numpy(y))
    trainset, valset = torch.utils.data.random_split(dataset=dataset,lengths=[int(0.8*len(dataset)), int(0.2*len(dataset))],generator=torch.Generator().manual_seed(0))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size,shuffle=True)
    
    
    #train model     
    start = time.perf_counter()
    print('start trainning!')
    train_loss_list=[]
    val_loss_list=[]
    lr_list=[]
    for epoch in range(epochs):
        train_loss = 0.
        val_loss=0.
        for batch_idx, sample in enumerate(train_loader):
            data, target = sample[0].to(device),sample[1].to(device)
            optimizer.zero_grad()
            output= model(data)
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss=train_loss/(batch_idx+1)
        train_loss_list.append(train_loss)
        # test per epoch
        with torch.no_grad():  
            for batch_idx, sample in enumerate(val_loader):
                data, target = sample[0].to(device),sample[1].to(device)
                output = model(data)
                loss = loss_func(output, target)
                val_loss+=loss.item() 
        val_loss=val_loss/(batch_idx+1)
        val_loss_list.append(val_loss)
        
        scheduler.step(val_loss)
        current_lr=optimizer.state_dict()['param_groups'][0]['lr']
        lr_list.append(current_lr)
        print('Train Epoch:{:4}  lr:{:.8f}:    train_loss: {:.6f} val_loss: {:.6f}'.format(epoch+1,current_lr,train_loss,val_loss)) 
    if is_log:
        np.save(model_path+'/train_loss.npy',train_loss_list)
        np.save(model_path+'/val_loss.npy',val_loss_list)
        np.save(model_path+'/lr.npy',lr_list)
        
        plt.figure()
        plt.plot(train_loss_list[2:],'b-.',lw=2)
        plt.plot(val_loss_list[2:],'y-.',lw=2)
        plt.title('loss curve')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()
        plt.savefig(model_path+'loss.png',bbox_inches='tight', pad_inches=0)
    
    print("backward_model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    
    torch.save(model.state_dict(), model_path+'model_state_dict.pkl')#save parammeter of the model
    print('saved model parameters successfully!')
    print('train finished!','train time:',time.perf_counter() - start)
    print("\n")
def train_backward_model(model,model_path,epochs,lr,batch_size,is_log,bias,activation_functuion,x_path,forward_model):
    print('-----------start train backward model-----------')
    with h5py.File(x_path,'r') as f:
        x=f[name][0:desired_num]
    x=image_resize(x,(target_size,target_size))
    paddding_length=int((speckle_size-target_size)/2)
    x=np.pad(x, ((0,0),(paddding_length,paddding_length), (paddding_length, paddding_length)), 'constant',constant_values=0)
    x=total_normalization(x,0,1).reshape(-1,speckle_size**2).astype('float32')

    model.to(device)
    optimizer= torch.optim.Adam(model.parameters(), lr=lr)
    loss_func= torch.nn.MSELoss(reduction='mean')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',verbose=True,patience=5)
    dataset=torch.utils.data.TensorDataset(torch.from_numpy(x))
    trainset, valset = torch.utils.data.random_split(dataset=dataset,lengths=[int(0.8*len(dataset)), int(0.2*len(dataset))],generator=torch.Generator().manual_seed(0))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size,shuffle=True)
    
    
    #train model     
    start = time.perf_counter()
    print('start trainning!')
    train_loss_list=[]
    val_loss_list=[]
    lr_list=[]
    for epoch in range(epochs):
        train_loss = 0.
        val_loss=0.
        for batch_idx, sample in enumerate(train_loader):
            data= sample[0].to(device)
            optimizer.zero_grad()
            mask= model(data)
            forward_model.eval()
            output=forward_model(mask)
            loss = loss_func(output, data)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss=train_loss/(batch_idx+1)
        train_loss_list.append(train_loss)
        # test per epoch
        with torch.no_grad():  
            for batch_idx, sample in enumerate(val_loader):
                data= sample[0].to(device)
                mask= model(data)
                forward_model.eval()
                output=forward_model(mask)
                loss = loss_func(output, data)
                val_loss+=loss.item() 
        val_loss=val_loss/(batch_idx+1)
        val_loss_list.append(val_loss)
        
        scheduler.step(val_loss)
        current_lr=optimizer.state_dict()['param_groups'][0]['lr']
        lr_list.append(current_lr)
        print('Train Epoch:{:4}  lr:{:.8f}:    train_loss: {:.6f} val_loss: {:.6f}'.format(epoch+1,current_lr,train_loss,val_loss)) 
    if is_log:
        np.save(model_path+'/train_loss.npy',train_loss_list)
        np.save(model_path+'/val_loss.npy',val_loss_list)
        np.save(model_path+'/lr.npy',lr_list)
        
        plt.figure()
        plt.plot(train_loss_list[2:],'b-.',lw=2)
        plt.plot(val_loss_list[2:],'y-.',lw=2)
        plt.title('loss curve')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()
        plt.savefig(model_path+'loss.png',bbox_inches='tight', pad_inches=0)
    
    print("backward_model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    
    torch.save(model.state_dict(), model_path+'model_state_dict.pkl')#save parammeter of the model
    print('saved model parameters successfully!')
    print('train finished!','train time:',time.perf_counter() - start)
    print("\n")

def collect_data():
    print('-----------start collect data-----------')
    cam=Basler_Camera(camera_name,if_Hardware_trigger,Width,Height,OffsetX,OffsetY,ExposureTime,PixelFormat)
    camera=cam.Config()
    slm = slmpy.SLMdisplay(isImageLock = True)
    
    #采集数据
    phase_data=np.load(random_pattern_192_path)[0:desired_num]
    image_num,size_y,size_x=phase_data.shape
    print('total data_number',image_num)
    result=np.zeros(shape=(image_num,Height,Width),dtype = 'uint8')
    start1= time.perf_counter()
    for i in range(image_num):
        if i%1000==0:
            print(i)
        mask_temp=np.pad(phase_data[i], ((a,a), (b, b)), 'constant',constant_values=0)
        mask_temp=((mask_temp+WFC)%256).astype('uint8')
        slm.updateArray(mask_temp,sleep=0.15)
        time.sleep(0.1)
        grabResult= camera.GrabOne(100)
        grabResult_array=grabResult.Array
        result[i]=grabResult_array
    end1= time.perf_counter()
    print('fps:',image_num/(end1-start1))
    print('time:',(end1-start1)/60,'min')
    print('\n')
    slm.close()
    cam.Close(camera)
    
    #存储数据
    np.save(speckle_path,result)
    
    plt.figure()
    plt.imshow(result[5],cmap='gray')
    plt.show()

def Parameter_recording():
    parameter=[
    '-----experemental parameter record-----',
    '\n',
    
    '-----basic parameter-----',
    {'experemental person ':experemental_person},
    {'data time ':date_time},
    {'experemental purpose ':experemental_purpose},
    {'laser ':laser},
    {'experemental_sample ':experemental_sample},
    '\n',
    
    '-----camera parameter-----',
    {'camera device ':camera_name},
    {'camera OffsetX ':OffsetX},
    {'camera OffsetY ':OffsetY},
    {'camera Width ':Width},
    {'camera Height':Height},
    {'camera collected data ExposureTime ':ExposureTime},
    '\n',
    
    '-----data parameter-----',
    {'data name ': name},
    {'image size ': image_size},
    {'speckle size ': speckle_size},
    {'data_num ': desired_num},
    {'collected time ':str(collected_time)+' S'},

    '\n',
    '-----slm parameter-----',
    {'slm display_shape ':display_shape},
    {'slm_shape ':slm_shape},
    {'micropixel size ': micropixel_size},
    
    '\n',
    '-----projection parameter-----',
    {'ExposureTime_list ':ExposureTime_list},
    
    '\n',
    '-----experemental result and analyze-----'
    '\n',
    '结果不错'
    ]
    file = open(save_root_path+'/'+'experimental_parameter.txt', 'w')
    for i in range(len(parameter)):
        s = str(parameter[i])+'\n'
        file.write(s)
    file.close()

def projection(name,target_size,projection_num):
    print('-----------start projection-----------')
    with h5py.File(image_path,'r') as f:
        target=f[name][0:projection_num]
    target=image_resize(target,(target_size,target_size))
    paddding_length=int((speckle_size-target_size)/2)
    target=np.pad(target, ((0,0),(paddding_length,paddding_length), (paddding_length, paddding_length)), 'constant',constant_values=0)
    target=total_normalization(target,0,1).reshape(-1,speckle_size**2).astype('float32')
    
    with torch.no_grad():
        target_tensor=(torch.from_numpy(target)).to(device)
        mask=backward_model(target_tensor)
        predict=forward_model(mask).cpu().numpy().reshape(-1,speckle_size,speckle_size)
        
    predict=(total_normalization(predict,0,255)).astype('uint8')
    predict_result_path=save_root_path+'/predict_result/'
    makedir(predict_result_path)
    
    for i in range(len(predict)):
        cv2.imwrite(predict_result_path+'/'+str(i+1)+'.png',predict[i])
    
    
    mask=(mask.cpu().numpy()).reshape(-1,image_size,image_size)
    mask=(total_normalization(mask,0,255)).astype('uint8')
    image_num,size_y,size_x=mask.shape
    
    
    mask_phase_name='all_mask_phase'
    image_target_name='all_image_target'
    with h5py.File(mask_path,'w') as f:
        f.create_dataset(mask_phase_name,data=mask.reshape(-1,image_size,image_size))
        f.create_dataset(image_target_name,data=target.reshape(-1,speckle_size,speckle_size))


    for ExposureTime in ExposureTime_list:
        result=np.zeros(shape=(image_num,Height,Width),dtype = 'uint8')
        cam=Basler_Camera(camera_name,if_Hardware_trigger,Width,Height,OffsetX,OffsetY,ExposureTime,PixelFormat)
        camera=cam.Config()
        slm = slmpy.SLMdisplay(isImageLock = True)
        for i in range(image_num):
            mask_temp=image_resize(mask[i],display_shape).astype('uint8')
            mask_temp=np.pad(mask_temp, ((a,a), (b, b)), 'constant',constant_values=0)
            mask_temp=((mask_temp+WFC)%256).astype('uint8')
            slm.updateArray(mask_temp,sleep=0.15)
            time.sleep(0.1)
            grabResult= camera.GrabOne(100)
            grabResult_array=grabResult.Array
            result[i]=grabResult_array
        slm.close()
        camera.Close()
        name='projection_ET'+str(ExposureTime)
        projection_result_path=save_root_path+'/projection_result/'
        makedir(projection_result_path)
        makedir(projection_result_path+name)
        np.save(projection_result_path+name+'.npy',result)
        for i in range(len(result)):
            cv2.imwrite(projection_result_path+name+'/'+str(i+1)+'.png',result[i])

device=GPU_check()

'''
实验基本参数
'''
experemental_person='luwenjian'
experemental_purpose='强化学习，光传播控制'
experemental_sample='graded-index MMF'
laser='633nm HeNe'
micropixel_size=3
'''
相机参数
camera_name:
    Basler acA2000-165umNIR (22450980)
    Basler acA1300-200um (22911411)
    Basler acA1300-200um (22911414)
'''
camera_name='Basler acA1300-200um (22911414)'
if_Hardware_trigger=False
OffsetX=560
OffsetY=370
Width=128
Height=128
ExposureTime=8000
PixelFormat="Mono8"

print('-----------start test camera and SLM-----------')
cam=Basler_Camera(camera_name,if_Hardware_trigger,Width,Height,OffsetX,OffsetY,ExposureTime,PixelFormat)
camera=cam.Config()
cam.Close(camera)

'''
初始化SLM参数
'''
slm = slmpy.SLMdisplay(isImageLock = True)
resX, resY= slm.getSize()
print('slm found and open')
print('slm shape:',resX, resY)
slm.close()
print('slm close')
print('\n')
display_shape=(192,192)
slm_shape=(resY, resX)
a=int(slm_shape[0]/2-display_shape[0]/2)
b=int(slm_shape[1]/2-display_shape[1]/2)
WFC=cv2.imread('F:/lwj/SLM/Test_SLM/slm5022.bmp',0)

'''
数据参数
'''
name='MNIST'
date_time='20210621'
image_size=64
speckle_size=128
desired_num=20000
save_root_path='./data/'+date_time
makedir(save_root_path)

image_path='F:/lwj/SLM/raw_image_data/'+name+'_64_50000.h5'
random_pattern_192_path='./data/random_pattern/phase192.npy'
random_pattern_64_path='./data/random_pattern/phase64.npy'
speckle_path=save_root_path+'/speckle.npy'
'''
训练参数
'''
epochs=50
lr=1e-4
batch_size =100
is_log=True
bias=False
activation_functuion=True

forward_model_path=save_root_path+'/forward_model/'
makedir(forward_model_path)
backward_model_path=save_root_path+'/backward_model/'
makedir(backward_model_path)
forward_model=FCL(image_size**2,speckle_size**2,bias,activation_functuion)
backward_model=FCL(speckle_size**2,image_size**2,bias,activation_functuion)

'''
投影参数
'''
target_size=96
projection_num=50
ExposureTime_list=[1000,5000]
mask_path=save_root_path+'/'+'projection_data.h5'
#%%
start= time.perf_counter()
collect_data()
end= time.perf_counter()
collected_time=end-start

train_forward_model(forward_model,forward_model_path,epochs,lr,batch_size,is_log,bias,activation_functuion,random_pattern_64_path,speckle_path)

train_backward_model(backward_model,backward_model_path,epochs,lr,batch_size,is_log,bias,activation_functuion,image_path,forward_model)

projection(name,target_size,projection_num)

Parameter_recording()
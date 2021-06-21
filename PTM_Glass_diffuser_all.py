# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 16:27:26 2021

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
# from functions import *
from pypylon import pylon
from pypylon import genicam
from datetime import datetime
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import Linear

""" check if GPU is available or not.
    if not set 'device=CPU'',
    else set 'device=GPU' and print the type and num of GPU"""
    
print('torch version:',torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('available device:',device)
if device==torch.device('cuda'):
    GPU_num=torch.cuda.device_count()
    for i in range (GPU_num):
        print('GPU',str(i),':',torch.cuda.get_device_name(i))
else :
    print('No GPU is available!')
    
    
'''
基本函数
'''
def makedir(path):
    isExists=os.path.exists(path)
    # 判断结果
    if not isExists:
    
        os.makedirs(path) 
     
        print (path+' 创建成功')
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print (path+' 目录已存在')
def complex_normalization(data):
   amp=total_normalization(np.abs(data),0,1)
   # phase=every_normalization(np.angle(data),-np.pi,np.pi)
   phase=np.angle(data)
   return amp*np.exp(1j*phase)

def total_normalization( data,ymin,ymax ):
    xmax=np.max(data)
    xmin=np.min(data)
    data=(ymax-ymin)*(data-xmin)/(xmax-xmin) + ymin
    return data
def channels_to_complex_np(X):
    return X[..., 0] + 1j * X[..., 1]

def LG_filter(data,D0):
    
    def paddedsize(size):
        row=2*size[0]
        column=2*size[1]
        return row,column
    
    def dftuv(M,N):
        
        u=np.linspace(0,(M-1),M)
        v=np.linspace(0,(N-1),N)
        idx=u>M/2
        u[idx]=u[idx]-M
        idx=v>N/2
        v[idx]=v[idx]-N
        [V, U] = np.meshgrid(v, -u)
        
        return[U,V]
        
    def lp_filter(M,N):
        [U, V] = dftuv(M,N)
        D = np.sqrt(np.square(U) + np.square(V))
        H = (U+1j*V)*(np.exp(-(np.square(D))/(D0*D0)))
        return H
    #show picture
    image1=data[0]
    shape=np.shape(data[0])
    PQ  = paddedsize(shape)
    lg_filter=lp_filter(PQ[0],PQ[1])
    # image1_fft=np.fft.fft2(image1,s=(PQ[0],PQ[1]))
    
    # plt.figure()
    # plt.imshow(np.abs(np.fft.fftshift(image1_fft)))
    # plt.title('image1_fft')
    
    # plt.figure()
    # plt.imshow(np.abs(np.fft.fftshift(lg_filter)))
    # plt.title('lg_filter')
    
    
    # FrefigofSpk=np.abs(np.fft.fftshift(image1_fft))
    # FrefigofFilt=np.abs(np.fft.fftshift(lg_filter))
    
    # plt.figure()
    # plt.plot(FrefigofSpk[shape[1],:]/np.max(FrefigofSpk[shape[1],:]))

    # plt.plot(FrefigofFilt[shape[1],:]/np.max(FrefigofFilt[shape[1],:]))
    
    Virtual_data=np.zeros((len(data),shape[0],shape[1]),dtype=np.complex64) 
    for i in range(len(data)):
         fft_data=np.fft.fft2(data[i],s=(PQ[0],PQ[1]))
         filter_data=lg_filter*fft_data
         recover_data=np.fft.ifft2(filter_data)
         Virtual_data[i]=recover_data[0:(shape[0]),0:(shape[1])]
    return Virtual_data

def complex_to_channels_np(Z):
    RE = np.real(Z)
    IM = np.imag(Z)

    if Z.shape[-1] == 1:
        RE = np.squeeze(RE, (-1))
        IM = np.squeeze(IM, (-1))

    return np.stack([RE, IM], axis=-1)

def real_to_channels_np(X):
    # Create complex with zero imaginary part
    X_c = X + 0.j
    return complex_to_channels_np(X_c)

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

def collect_data():
    cam=Basler_Camera(camera_name,if_Hardware_trigger,Width,Height,OffsetX,OffsetY,ExposureTime,PixelFormat)
    camera=cam.Config()
    slm = slmpy.SLMdisplay(isImageLock = True)
    #采集数据
    phase_data=np.load(phase_data_path+'phase192.npy')[0:desired_num]
    image_num,size_y,size_x=phase_data.shape
    print(image_num)
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
    print('time:',(end1-start1)/60)
    slm.close()
    cam.Close(camera)
    
    #存储数据
    np.save(save_root_path+'/'+'speckle.npy',result)
    
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
    {'data_num ': data_num},
    {'D0 ': D0},
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
    
    
def data_process():
    phase=(np.load(phase_data_path+'phase64.npy'))[0:desired_num]
    amplitude=np.sqrt(np.load('./grab_data/amp.npy'))
    speckle=np.load(save_root_path+'/'+'speckle.npy')
    speckle=np.sqrt(speckle)
    phase=total_normalization(phase.reshape(-1,image_size*image_size),0,(2*np.pi))
    
    amplitude=total_normalization((amplitude).reshape(-1,image_size*image_size),0.1,1)
    complex_image=amplitude*np.exp(1j*(phase))
    image_ch=complex_to_channels_np(complex_image)
    del complex_image
    del amplitude
    del phase
    gc.collect
    
    speckle=total_normalization(speckle.reshape(-1,speckle_size,speckle_size),0,1)
    pseudo_speckle=LG_filter(speckle,D0=D0)
    pseudo_speckle=pseudo_speckle.reshape(-1,speckle_size*speckle_size)
    pseudo_speckle=complex_normalization(pseudo_speckle)
    speckle_ch=complex_to_channels_np(pseudo_speckle)
        
    # pseudo_speckle=speckle.reshape(-1,speckle_size*speckle_size)
    # speckle_ch=pseudo_speckle*np.exp(1j)
    # speckle_ch=complex_to_channels_np(speckle_ch)
    # speckle_ch=real_to_channels_np(pseudo_speckle)
    del pseudo_speckle
    del speckle
    gc.collect
    data_length=len(image_ch)
    with h5py.File(dataset_path,'w') as f:
        train=f.create_group("train")
        val=f.create_group("val")
        test=f.create_group("test")
        train.create_dataset('image',data=image_ch[0:int((data_length/10)*8)])
        train.create_dataset('speckle',data=speckle_ch[0:int((data_length/10)*8)])
        val.create_dataset('image',data=image_ch[int((data_length/10)*8):int((data_length/10)*9)])
        val.create_dataset('speckle',data=speckle_ch[int((data_length/10)*8):int((data_length/10)*9)])
        test.create_dataset('image',data=image_ch[int((data_length/10)*9):data_length])
        test.create_dataset('speckle',data=speckle_ch[int((data_length/10)*9):data_length])
    del image_ch
    del speckle_ch
    gc.collect
    with h5py.File(dataset_path,'r') as f:
        image_ch=f['train/image'][0:100]
        speckle_ch=f['train/speckle'][0:100]
        complex_image=channels_to_complex_np(image_ch).reshape(-1,image_size,image_size)
        complex_speckle=channels_to_complex_np(speckle_ch).reshape(-1,speckle_size,speckle_size)

    plt.figure()
    plt.imshow(np.angle(complex_image[0]),cmap='gray')
    plt.axis('off')
    plt.show()
    plt.savefig(save_root_path+'/'+'image_phase.png',bbox_inches='tight',pad_inches = 0,dpi=600)
    
    plt.figure()
    plt.imshow(np.abs(complex_image[0]),cmap='gray')
    plt.axis('off')
    plt.show()
    plt.savefig(save_root_path+'/'+'image_amp.png',bbox_inches='tight',pad_inches = 0,dpi=600)
    
    plt.figure()
    plt.imshow((np.angle(complex_speckle[0])),cmap='gray')
    plt.axis('off')
    plt.show()
    plt.savefig(save_root_path+'/'+'speckle_pseudo_phase.png',bbox_inches='tight',pad_inches = 0,dpi=600)
    
    plt.figure()
    plt.imshow(np.abs(complex_speckle[0]),cmap='gray')
    plt.axis('off')
    plt.show()
    plt.savefig(save_root_path+'/'+'speckle_pseudo_amp.png',bbox_inches='tight',pad_inches = 0,dpi=600)
    
class Mydataset(Dataset):
    def __init__(self,dataset_path,is_train,transform=None):
        """
        Args:
            dataset_path: the path of dataset
            is_train: True: train dataset.  False:test dataset
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataset_path=dataset_path
        self.is_train=is_train
        self.transform = transform
        with h5py.File(self.dataset_path,'r') as f:
            if self.is_train:
                  self.image=f['train/image'][()].astype('float32')
                  self.speckle=f['train/speckle'][()].astype('float32')
                  self.length=len(self.image)
            else:
                    self.image=f['val/image'][()].astype('float32')
                    self.speckle=f['val/speckle'][()].astype('float32')
                    self.length=len(self.image)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        speckle=self.speckle[idx]
        image=self.image[idx]
        sample= {'speckle': speckle,'image': image}
        if self.transform:
            sample= self.transform(sample)
        return sample
class ComplexLinear(nn.Module):

    def __init__(self, in_features, out_features,bias=False):
        super(ComplexLinear, self).__init__()
        self.fc_r = Linear(in_features, out_features,bias=bias)
        self.fc_i = Linear(in_features, out_features,bias=bias)

    def forward(self,input_r, input_i):
        return self.fc_r(input_r)-self.fc_i(input_i), \
                self.fc_r(input_i)+self.fc_i(input_r)
class ComplexNet(nn.Module):
    
    def __init__(self,speckle_flatten_size,image_flatten_size,bias):
        super(ComplexNet, self).__init__()
        self.fc= ComplexLinear(speckle_flatten_size,image_flatten_size,bias)
        self.loss_func = torch.nn.MSELoss() 
    def forward(self,x,y=None):
        xr = x[:,:,0]
        xi = x[:,:,1]
        xr,xi = self.fc(xr,xi)
        
        # take the absolute value as output
        # x = torch.sqrt(torch.pow(xr,2)+torch.pow(xi,2))
        
        #concatenate xr,xi in the last dim
        x_out=torch.stack((xr,xi),dim=-1)
        
        #calculate loss(solve the multi-GPU imbalanced problem)
        if y is not None:
            loss=self.loss_func(x_out,y)
            return loss
        else:
            return x_out
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, speckle = sample['image'], sample['speckle']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'speckle': torch.from_numpy(speckle)}

def train():
    if is_log:
        makedir(save_root_path+'/model'+'/logs')      
        logs_path=save_root_path+'/model'+'/logs/'+'{0:%Y-%m-%dT%H-%M-%S}'.format(datetime.now())
    #load dataset
    trans=transforms.Compose([ToTensor()])
    train_dataset= Mydataset(dataset_path=dataset_path,is_train=True,transform=trans)
    train_loader =DataLoader(train_dataset, batch_size= batch_size, shuffle=True)
    val_dataset= Mydataset(dataset_path=dataset_path,is_train=False,transform=trans)
    val_loader =DataLoader(val_dataset, batch_size= batch_size, shuffle=True)
    
    #define model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ComplexNet(speckle_flatten_size,image_flatten_size,bias)
    # if os.path.exists(model_path):
    #     model.load_state_dict(torch.load(model_path))
    #     print('load model successfully !')
    model.to(device)
    model=torch.nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=0)
    # optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum=0.9)
    loss_func = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',verbose=True,patience=5)
    if is_log:
        tb_writer = SummaryWriter(logs_path,comment=name)
    
    
    
    #run training
    start = time.process_time()
    print('start training!')
    for epoch in range(epochs):
        train_loss = 0.
        val_loss=0.
        for batch_idx, sample in enumerate(train_loader):
            data, target = sample['speckle'].to(device),sample['image'].to(device)
            optimizer.zero_grad()
            output= model(data)
            if is_phase_optimization:
                loss=loss_func(torch.atan2(output[:,:,1],output[:,:,0]),torch.atan2(target[:,:,1],target[:,:,0]))
            else:
                loss = loss_func(output,target)
            # output= model(data,target)  #reduce the main GPU calculate loss time
            # loss=output.mean()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss=train_loss/(batch_idx+1)
          
        # val per epoch
        with torch.no_grad():  
            for batch_idx, sample in enumerate(val_loader):
                data, target = sample['speckle'].to(device),sample['image'].to(device)
                output = model(data)
                
                if is_phase_optimization:
                    loss=loss_func(torch.atan2(output[:,:,1],output[:,:,0]),torch.atan2(target[:,:,1],target[:,:,0]))
                else:
                    loss = loss_func(output,target)
                
                val_loss+=loss.item() 
        val_loss=val_loss/(batch_idx+1)
        current_lr=optimizer.state_dict()['param_groups'][0]['lr']
        
        scheduler.step(val_loss)
        print('Train Epoch:{:4}  lr:{:.8f}:    train_loss: {:.6f} val_loss: {:.6f}'.format(epoch+1,current_lr,train_loss,val_loss)) 
    
        if is_log:
            
            tb_writer.add_scalars('MSE_loss',{'train_loss':train_loss,'val_loss':val_loss},epoch)
            tb_writer.add_scalar('lr', current_lr, epoch) 
    
        # print('Train Epoch:{:4}  lr:{:.8f}:    train_loss: {:.6f}'.format(epoch+1,current_lr,train_loss)) 
        # if is_log:
            
        #     tb_writer.add_scalar('train_loss',train_loss,epoch)
        #     # tb_writer.add_scalar('test_loss',test_loss,epoch)
        #     tb_writer.add_scalar('lr', current_lr, epoch) 
    if is_log:
        tb_writer.close()
        
        
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    
    torch.save(model.module.state_dict(), model_path)#save parammeter of the model
    # torch.save(model, model_path)#save the whole model
    
    real_matrix=model.module.fc.fc_r.weight.data.cpu().numpy()
    imag_matrix=model.module.fc.fc_i.weight.data.cpu().numpy()
    
    complex_weight_matrix=real_matrix+1j*(imag_matrix)
    np.save(complex_weight_matrix_path,complex_weight_matrix)
    
    print('saved model or weight_matrix successfully!')
    print('train finished!','train time:',time.process_time() - start)
    torch.cuda.empty_cache()
def test():
    #test
    with h5py.File(dataset_path,'r') as f:
        image=f['test/image'][0:100].astype('float32')
        speckle=f['test/speckle'][0:100].astype('float32')
    complex_speckle=speckle[..., 0] + 1j * speckle[..., 1]
    complex_image=image[..., 0] + 1j * image[..., 1]
    
    complex_weight_matrix=np.load(complex_weight_matrix_path)
    predict=(np.matmul(complex_weight_matrix,(complex_speckle).T)).T
        
    plt.figure()
    for i in range(1,17):
        plt.subplot(4,4,i) 
        plt.imshow((np.abs(np.angle(predict[i-1]))).reshape(image_size,image_size),cmap='gray')
        plt.axis('off')
        plt.title('predict phase')
    plt.show()
    
    plt.figure()
    for i in range(1,17):
        plt.subplot(4,4,i)
        plt.imshow(np.abs(predict[i-1]).reshape(image_size,image_size),cmap='gray')
        plt.axis('off')
        plt.title('predict amplitude')
    plt.show()
    
    plt.figure()
    for i in range(1,17):
        plt.subplot(4,4,i)
        plt.imshow((np.angle(complex_image[i-1])).reshape(image_size,image_size),cmap='gray')
        plt.axis('off')
        plt.title('original phase')
    plt.show()
    
    plt.figure()
    for i in range(1,17):
        plt.subplot(4,4,i)
        plt.imshow(np.abs(complex_image[i-1]).reshape(image_size,image_size),cmap='gray')
        plt.axis('off')
        plt.title('original amplitude')
    plt.show()

def projection():
    
    image_target=np.load(target_path)
    
    
    pseudo_target=LG_filter(image_target.reshape(-1,speckle_size,speckle_size),D0=D0).reshape(-1,speckle_size*speckle_size)
    pseudo_target=complex_normalization(pseudo_target)
    
    complex_weight_matrix=np.load(complex_weight_matrix_path)
    mask=(np.matmul(complex_weight_matrix,pseudo_target.T).T)
    
    mask_phase=np.angle(mask)
    mask_amp=np.abs(mask)
    
    
    mask_phase_name='all_mask_phase'
    image_target_name='all_image_target'
    

    with h5py.File(mask_path,'w') as f:
        f.create_dataset(mask_phase_name,data=mask_phase.reshape(-1,image_size,image_size))
        f.create_dataset(image_target_name,data=image_target.reshape(-1,speckle_size,speckle_size))
    
    with h5py.File(mask_path,'r') as f:
        mask=f['all_mask_phase'][()]
    mask[mask<0]=mask[mask<0]+2*np.pi
    mask=(total_normalization(mask,0,255)).astype('uint8')
    image_num,size_y,size_x=mask.shape
    
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
    
'''
实验基本参数
'''
experemental_person='luwenjian'
experemental_purpose='哈哈哈哈'
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
ExposureTime=1500
PixelFormat="Mono8"

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

display_shape=(192,192)
slm_shape=(resY, resX)
a=int(slm_shape[0]/2-display_shape[0]/2)
b=int(slm_shape[1]/2-display_shape[1]/2)
WFC=cv2.imread('F:/lwj/SLM/Test_SLM/slm5022.bmp',0)
'''
数据参数
'''
name='random'
date_time='20210620'
image_size=64
speckle_size=128
image_flatten_size=image_size*image_size
speckle_flatten_size=speckle_size*speckle_size
data_num=30000
desired_num=20000
D0=50
phase_data_path='./raw_input_data/random_64_192_3/'
save_root_path='./grab_data/'+date_time
dataset_path=save_root_path+'/'+'dataset.h5'
makedir(save_root_path)

'''
训练参数
'''
epochs=50
lr=1e-4
batch_size =400
is_log=True
is_phase_optimization=False
bias=False
model_path=save_root_path+'/model/'+'model_state_dict.pkl'
complex_weight_matrix_path=save_root_path+'/model/'+'weight_matrix.npy'
makedir(save_root_path+'/model')
'''
投影参数
'''
ExposureTime_list=[60,100,200,300]
mask_path=save_root_path+'/'+'projection_data.h5'
target_path='raw_input_data/target.npy'
mask_path=save_root_path+'/projection_data.h5'
#%%
'''
数据采集
'''
start= time.perf_counter()
collect_data()
end= time.perf_counter()
collected_time=end-start
#%%
'''
实验参数记录
'''
Parameter_recording()
#%%
'''
数据处理
'''
data_process()

#%%
'''
训练
'''
train()
# test()
#%%
'''
投影数据
'''
projection()

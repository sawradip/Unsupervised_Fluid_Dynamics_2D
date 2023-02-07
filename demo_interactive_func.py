import os
import cv2
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

import get_param
from setups import Dataset
from pde_cnn import get_Net
from Logger import Logger
from derivatives import params
from derivatives import rot_mac, vector2HSV
from derivatives import toCuda,toCpu
from derivatives import normal2staggered,staggered2normal



save_movie = params.save_movie #False#True#
n_time_steps = params.average_sequence_length
print(n_time_steps)

screen_update_freq = 10




    

def show_demo(w = params.width, 
            h = params.height, 
            mu = 0.4,
            rho = 4,
            dt = 4,
            type="magnus", 
            image="wing",
            background_image="empty",
            average_sequence_length = n_time_steps,
            ):
    # params.mu = mu
    # params.rho = rho
    # params.dt = dt

    dataset = Dataset(w = w, 
                    h = h, 
                    batch_size = 1, 
                    dataset_size = 1, 
                    interactive = True, 
                    average_sequence_length = average_sequence_length,
                    max_speed = params.max_speed, 
                    dt = dt, 
                    types=[type], 
                    images=[image],
                    background_images=[background_image]
                    )

    # Mouse interactions:
    def mousePosition(event,x,y,flags,param):
        # global dataset
        if (event==cv2.EVENT_MOUSEMOVE or event==cv2.EVENT_LBUTTONDOWN) and flags==1:
            dataset.mousex = x
            dataset.mousey = y


    # load fluid model:
    logger = Logger(get_param.get_hyperparam(params),use_csv=False,use_tensorboard=False)
    fluid_model = toCuda(get_Net(params))
    date_time,index = logger.load_state(fluid_model,None,datetime=params.load_date_time,index=params.load_index)
    fluid_model.eval()
    print(f"loaded {params.net}: {date_time}, index: {index}")

    if save_movie:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        movie_p = cv2.VideoWriter(  filename =  f'plots/p_{get_param.get_hyperparam(params)}.avi', 
                                    fourcc = fourcc, 
                                    fps =  20.0 , 
                                    frameSize = (w,  h)
                                    )
        movie_v = cv2.VideoWriter(  filename = f'plots/v_{get_param.get_hyperparam(params)}.avi', 
                                    fourcc = fourcc, 
                                    fps =  20.0, 
                                    frameSize = (w-3,  h-3)
                                    )
        movie_a = cv2.VideoWriter(  filename = f'plots/a_{get_param.get_hyperparam(params)}.avi', 
                                    fourcc = fourcc, 
                                    fps =  20.0 , 
                                    frameSize = (w,  h)
                                    )
    # setup opencv windows:
    cv2.namedWindow('Legend',cv2.WINDOW_NORMAL) # legend for velocity field
    contour_vector = torch.cat([torch.arange(-1, 1, 0.01).unsqueeze(0).unsqueeze(2).repeat(1,1,200),
                        torch.arange(-1,1,0.01).unsqueeze(0).unsqueeze(1).repeat(1,200,1)]).cuda()
    contour_image = vector2HSV(contour_vector)
    contour_image = cv2.cvtColor(contour_image,cv2.COLOR_HSV2BGR)
    cv2.imshow('Legend',contour_image)

    cv2.namedWindow('p',cv2.WINDOW_NORMAL)
    cv2.namedWindow('v',cv2.WINDOW_NORMAL)
    cv2.namedWindow('a',cv2.WINDOW_NORMAL)

    cv2.setMouseCallback("p",mousePosition)
    cv2.setMouseCallback("v",mousePosition)
    cv2.setMouseCallback("a",mousePosition)


    FPS = 0
    FPS_Counter = 0
    last_time = time.time()

    #simulation loop:
    for t in range(n_time_steps):
        v_cond, cond_mask, flow_mask, a_old, p_old = toCuda(dataset.ask())

        # convert v_cond,cond_mask,flow_mask to MAC grid:
        v_cond = normal2staggered(v_cond)
        cond_mask_mac = (normal2staggered(cond_mask.repeat(1,2,1,1))==1).float()
        flow_mask_mac = (normal2staggered(flow_mask.repeat(1,2,1,1))>=0.5).float()

        # MOST IMPORTANT PART: apply fluid model to advace fluid state
        with torch.no_grad():
            a_new,p_new = fluid_model(a_old,p_old,flow_mask,v_cond,cond_mask)   
            v_new = rot_mac(a_new)

        # Appendix B: normalize mean of p and a, to keep these fields well-defined and prevent drifting offset values
        p_new = (p_new-torch.mean(p_new,dim=(1,2,3)).unsqueeze(1).unsqueeze(2).unsqueeze(3))
        a_new = (a_new-torch.mean(a_new,dim=(1,2,3)).unsqueeze(1).unsqueeze(2).unsqueeze(3))

        if t % screen_update_freq == 0:
            # Show p on screen and save frame:
            p = flow_mask[0,0]*p_new[0,0].clone()
            p = p-torch.min(p)
            p = p/torch.max(p)
            p = toCpu(p).unsqueeze(2).repeat(1,1,3).numpy()
            if save_movie:
                movie_p.write((255*p).astype(np.uint8))
            cv2.imshow('p',p)

            # Show v on screen and save frame:
            v_new = flow_mask_mac * v_new + cond_mask_mac * v_cond
            vector = staggered2normal(v_new.clone())[0,:,2:-1,2:-1]
            image = vector2HSV(vector)
            image = cv2.cvtColor(image,cv2.COLOR_HSV2BGR)
            if save_movie:
                movie_v.write((255*image).astype(np.uint8))
            cv2.imshow('v',image)

            
            # Show a on screen and save frame:
            a = a_new[0,0].clone()
            a = a-torch.min(a)
            a = toCpu(a/torch.max(a)).unsqueeze(2).repeat(1,1,3).numpy()
            if save_movie:
                movie_a.write((255*a).astype(np.uint8))
            cv2.imshow('a',a)

            print(contour_image.shape)
            print(image.shape)
            print(p.shape)
            print(a.shape)
            print('--------------')
            # keyboard interactions:
            key = cv2.waitKey(1)
            
            if key==ord('x'): # increase flow speed
                dataset.mousev+=0.1

            if key==ord('y'): # decrease flow speed
                dataset.mousev-=0.1
            
            if key==ord('s'): # increase angular velocity
                dataset.mousew+=0.1
            if key==ord('a'): # decrease angular velocity
                dataset.mousew-=0.1
            
            if key==ord('n'): # start new environmet
                break
            
            if key==ord('p'): # print image
                flow = staggered2normal(v_new.clone())[0,:,2:-1,2:-1]
                image = vector2HSV(flow)
                flow = toCpu(flow).numpy()
                fig, ax = plt.subplots()
                Y,X = np.mgrid[0:flow.shape[1],0:flow.shape[2]]
                linewidth = image[:,:,2]/np.max(image[:,:,2])
                ax.streamplot(X, Y, flow[1], flow[0], color='k', density=1,linewidth=2*linewidth)
                palette = plt.cm.gnuplot2
                palette.set_bad('k',1.0)
                pm = np.ma.masked_where(toCpu(cond_mask).numpy()==1, toCpu(p_new).numpy())
                plt.imshow(pm[0,0,2:-1,2:-1],cmap=palette)
                plt.axis('off')
                os.makedirs("plots",exist_ok=True)
                name = dataset.env_info[0]["type"]
                if name=="image":
                    name = name+"_"+dataset.env_info[0]["image"]
                plt.savefig(f"plots/flow_and_pressure_field_{name}_t_{t}.png", bbox_inches='tight')
                plt.show()
        
        if key==ord('q'): # quit simulation
            # quit=True
            break

        FPS_Counter += 1
        if time.time()-last_time>=1:
            last_time = time.time()
            FPS=FPS_Counter
            FPS_Counter = 0

        if t % 10==0: # print out results only at every 10th iteration
            print(f"t:{t} (FPS: {FPS})")

        dataset.tell(toCpu(a_new),toCpu(p_new))

    if save_movie:
        movie_p.release()
        movie_v.release()
        movie_a.release()
        print('Plots Saved')

if __name__ == "__main__":
    # types = 'magnus', 'box', 'pipe', 'image'
    # images = "cyber", "fish", "smiley", "wing"
    # backgrounds = "empty","cave1","cave2"

    show_demo(w = 600,
            h = 200,
            type="magnus", 
            image="wing",
            background_image="empty")

import torch
import matplotlib.pyplot as plt
# Calculate total link length for given sample, it is assumed the base of robot is at (0,0). Input is the position of joints in one timestamp
def base_to_ee_distance(input):
  
    ## Turn data to sets of 2 so that we can reach each joint seperately.
    different_view = input.view(-1,2)
    
    ## TOOOODOOOOOOOOOOOOO in order to scale this without problem I need to do calculations for each of the joints    
    first_joint = different_view[0]
    
    ## Last element of the different_view is the end effector
    ee_pose = different_view[-1]
    base = torch.zeros(1,2).cuda()
    
    ## by calculating distance from first joint to base and end effector to first_joint we can calculate overall link distances 
    dist = ((first_joint - base)*(first_joint - base))+ ((ee_pose-first_joint)*(ee_pose-first_joint))
    dist = torch.sqrt(dist.sum())
    return dist

# First input is the data in "i"th time, second input is the "i+1" time data, and the last one is the total link length for the sampled robot
def past_to_present_dist(past,present,normalizer):
  
    ## For "i" data get its end effector position
    past_different_view = past.view(-1,2)
    past_ee_pose = past_different_view[1]
    
    ## For "i+1" data get its end effector position
    present_different_view = present.view(-1,2)
    present_ee_pose = present_different_view[1]
    
    ## L1 distance between past and present end effector positions
    diff = present_ee_pose - past_ee_pose
    diff = torch.abs(diff)
    
    ## Sum over x and y variable
    diff = diff.sum()
    
    ## Divide it to link length of the sampled robot
    diff = diff/normalizer
    return diff

  
# It takes one input and it is sample from one of the batches provided to similarity function
def ee_history(input):
  
    ## Length of the timestamped sequence
    length = input.shape[0]  
    
    ## Create a torch to keep  the outcome of calculations. There is a minus one because total history will be one less than length of the sample.
    ## The reason for that is we are calculating relative trajectory which will follow the actions of the end effector.
    ## For example first element of this torch will be the vector from timestamp0 to timestamp1
    keeper = torch.zeros(length-1,1).cuda()
    for i in range(1,length):
        
        ### Calculate the total link length for this sample
        dist = base_to_ee_distance(input[i])
        
        ### Calculate the vector from past timestamp to current one
        diff = past_to_present_dist(input[i-1],input[i],dist)
        
        ### Place it to keeper
        keeper[i-1] = diff
    return keeper

  
# It takes two inputs. First one is the "real" which is one of the robots. Second one is the "fake" which is the outcome of generator given "real".
def similarity(real,fake):
  
    ## Create variable to sum calculated differences over
    total_difference = torch.FloatTensor([0]).cuda()
    
    ## Get the batch size of the given input.
    batch_size = real.shape[0]
    
    ## For each sample duality(one from "real", one from "fake") from batch calculate similarity. 
    for i in range(batch_size):
      
        ### Calculate trajectory(relative positions ee were in) of the end effector for sample from "real" and put it in a torch 
        real_hist = ee_history(real[i])
        
        ### Calculate trajectory(relative positions ee were in) of the end effector for sample from "fake" and put it in a torch         
        fake_hist = ee_history(fake[i])
        
        ### Get difference of these trajectories for samples
        temp = real_hist - fake_hist
        
        ### Take absolute value of the difference 
        temp = torch.abs(temp)
        
        ### Sum the difference for each timestamp
        temp = temp.sum()
        
        ### Add it to total difference for the two batch
        total_difference += temp
        
    ### Return the total_difference    
    return total_difference


def write_to_file(m_data,des_file):
    resstr= []
    for i in m_data:

        for x in i:
            resstr.append(x)
    # print resstr
    # print len(resstr)
    str2 = ""
    for  e in resstr:
        str2 += "," + str(e)
    str2 = str2[1:]
    # print len(str2.split(","))
    # for e in str2.split(","):
        # print e
    with open(des_file,'w') as f:
        f.write(str2)
    del resstr

def write_res(m_data,des_file):
    
    with open(des_file,'w') as f:
        f.write(str(m_data))


def save_model(config, sensor1_gen, sensor2_gen, sensor1_dis, sensor2_dis,
               optimizer_sensor1_gen, optimizer_sensor2_gen, optimizer_sensor1_dis, optimizer_sensor2_dis, epoch):

    torch.save({
        'sensor1_gen': sensor1_gen.state_dict(),
        'sensor2_gen': sensor2_gen.state_dict(),
        'sensor1_dis': sensor1_dis.state_dict(),
        'sensor2_dis': sensor2_dis.state_dict(),
        'optimizer_sensor1_gen': optimizer_sensor1_gen.state_dict(),
        'optimizer_sensor2_gen': optimizer_sensor2_gen.state_dict(),
        'optimizer_sensor1_dis': optimizer_sensor1_dis.state_dict(),
        'optimizer_sensor2_dis': optimizer_sensor2_dis.state_dict()
    }, config.TRAIN.SAVE_WEIGHTS+str(epoch))


def save_vanilla_model(config, sensor1_gen, sensor1_dis, optimizer_sensor1_gen, optimizer_sensor1_dis, epoch):

    torch.save({
        'sensor1_gen': sensor1_gen.state_dict(),
        'sensor1_dis': sensor1_dis.state_dict(),
        'optimizer_sensor1_gen': optimizer_sensor1_gen.state_dict(),
        'optimizer_sensor1_dis': optimizer_sensor1_dis.state_dict()
    }, config.TRAIN.SAVE_WEIGHTS+str(epoch)+".pth")

def save_generator(config, sensor1_gen, optimizer_sensor1_gen, epoch):

    torch.save({
        'sensor1_gen': sensor1_gen.state_dict(),
        'optimizer_sensor1_gen': optimizer_sensor1_gen.state_dict()
    }, config.TRAIN.SAVE_WEIGHTS+str(epoch))



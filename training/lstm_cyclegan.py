from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals





import torch
print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

from Discriminator import Discriminator
from Generator import EncDec

from utils.helpers.helpers import similarity
from utils.helpers.helpers import write_to_file
from utils.helpers.helpers import write_res
from utils.data.Dataloader import lstm_cycle_dataloader
from utils.core.config import config
from utils.core.config import load_config


import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
import optparse


parser = optparse.OptionParser()

parser.add_option('-c', '--config', dest="config",
                  help="load this config file", metavar="FILE")

def train(dataloader, config, device):

	torch.manual_seed(1)




	# GPU device
	device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

	# How much of the action sequence we will take in to consideration
	target_length = config.TRAIN.TARGET_LENGTH







	first_discriminator = Discriminator(4,10,1,0).to(device)
	first_generator = EncDec(4,12,12,3,0).to(device)
	second_discriminator = Discriminator(4,10,1,0).to(device)
	second_generator = EncDec(4,12,12,3,0).to(device)


	# Initialize BCELoss function

	## This function is more stable version of BCELoss, it contains its own sigmoid and can take reduction arguments for Batchs with("none|mean|sum")
	criterion = nn.BCEWithLogitsLoss(reduction=config.TRAIN.DISCRIMINATOR_CRITERION_REDUCTION)

	cycleLossCalculation = nn.MSELoss(reduction=config.TRAIN.DISCRIMINATOR_CRITERION_REDUCTION)

	real_label = 1
	main_label = 0

	# Setup Adam optimizers for both G and D
	optimizerFD = optim.Adam(first_discriminator.parameters(), lr=config.DISCRIMINATOR.BASE_LR)
	optimizerFED = optim.Adam (first_generator.parameters(), lr=config.GENERATOR.BASE_LR)
	optimizerSD = optim.Adam (second_discriminator.parameters(), lr=config.DISCRIMINATOR.BASE_LR)
	optimizerSED = optim.Adam (second_generator.parameters(), lr=config.GENERATOR.BASE_LR)




	Fexample_outputs= []
	Fexample_inputs= []
	Sexample_outputs= []
	Sexample_inputs= []
	FG_losses = []
	SG_losses = []
	FD_losses = []
	SD_losses = []
	CycleLoss = []
	FirstReconstructedHolder = []
	SecondReconstructedHolder = []
	iters = 0

	pytorch_total_params = sum(p.numel() for p in first_generator.parameters())
	print("First Generator ",pytorch_total_params)
	pytorch_total_params = sum(p.numel() for p in first_discriminator.parameters())
	print("First Discriminator ",pytorch_total_params)
	Fdiscr_dec = 0
	Sdiscr_dec = 0

	if config.TRAIN.START_EPOCH>0:
	    print("loading previous model")
	    checkpoint = torch.load(load_path)
	    first_generator.load_state_dict(checkpoint['first_generator'])
	    second_generator.load_state_dict(checkpoint['second_generator'])
	    first_discriminator.load_state_dict(checkpoint['first_discriminator'])
	    second_discriminator.load_state_dict(checkpoint['second_discriminator'])
	    optimizerFD.load_state_dict(checkpoint['optimizerFD'])
	    optimizerFED.load_state_dict(checkpoint['optimizerFED'])
	    optimizerSED.load_state_dict(checkpoint['optimizerSED'])
	    optimizerSD.load_state_dict(checkpoint['optimizerSD'])


	    first_generator.train()
	    second_generator.train()
	    first_discriminator.train()
	    second_discriminator.train()
	    print("done")


	for epoch in range(start_epoch, config.TRAIN.MAX_EPOCH):
	    errFD = 0
	    errFG = 0
	    errFD_real=0
	    errFD_fake = 0
	    FD_G_z1 = 0.0
	    errSD = 0
	    errSG = 0
	    errSD_real=0
	    errSD_fake = 0
	    SD_G_z1 = 0.0



	    for i, (second_robot_data,first_robot_data) in enumerate(dataloader, 0):
	        first_robot_sample = first_robot_data[0:config.TRAIN.BATCH_SIZE].to(device)
	        #print first_robot_sample.size()
	        #first_robot_sample = first_robot_sample.view(config.TRAIN.BATCH_SIZE,-1,4)
	        b_size = first_robot_sample.size(0)

	        label_fake = torch.full((b_size,), real_label, device=device)
	        label_real = torch.full((b_size,), real_label, device=device)
	        
	        label_real = label_real.cuda()
	        label_fake = label_fake.cuda()
	#####################################################
	####################  Zero Gradients
	###################################################3
	        first_generator.zero_grad()
	        first_discriminator.zero_grad()
	        second_generator.zero_grad()
	        second_discriminator.zero_grad()

	        optimizerFD.zero_grad()
	        optimizerSD.zero_grad()
	        optimizerFED.zero_grad()
	        optimizerSED.zero_grad()
	###################################################################################
	#################           First Discriminator
	##################################################################################
	        # if(i % 2 == 0) :
	        second_robot_sample = second_robot_data[0:config.TRAIN.BATCH_SIZE].to(device)

	        # Forward pass real batch through D
	        output = first_discriminator(second_robot_sample)

	        output = output.view(-1)

	        label_real.fill_(real_label)

	        # Calculate loss on all-real batch
	        errFD_real = criterion(output, label_real)
	        # Calculate gradients for D in backward pass
	        errFD_real.backward()

	        FD_x = output.mean().item()
	        

	        fakeDec = first_generator(first_robot_sample)

	        label_fake.fill_(main_label)
	        # Classify all fake batch with D
	        output = first_discriminator(fakeDec.detach())
	        output = output.view(-1)
	        # Calculate D's loss on the all-fake batch
	        errFD_fake = criterion(output, label_fake)
	        # Calculate the gradients for this batch
	        errFD_fake.backward()

	        FD_G_z1 = output.mean().item()
	        # Add the gradients from the all-real and all-fake batches
	        errFD = (errFD_real + errFD_fake) 

	        optimizerFD.step()
	        
	         
	################################################################################
	#################################### First Generator
	################################################################################
	        
	        fakeDec = first_generator(first_robot_sample)

	        label_real.fill_(real_label)  # fake labels are real for generator cost
	        # Since we just updated D, perform another forward pass of all-fake batch thro"ugh D
	        output= first_discriminator(fakeDec)

	        output = output.view(-1)
	        # Calculate G's loss based on this output

	        errFG = criterion(output, label_real)
	        # Calculate gradients for G
	        errFG.backward()

	        FD_G_z2 = output.mean().item()
	        # Update G


	        


	###################################################################################
	#################           Second Discriminator
	##################################################################################
	        # if(i % 2 == 0) :
	        second_robot_sample = second_robot_data[0:config.TRAIN.BATCH_SIZE].to(device)


	        # Forward pass real batch through D
	        output = second_discriminator(first_robot_sample)

	        output = output.view(-1)

	        label_real.fill_(real_label)

	        # Calculate loss on all-real batch
	        errSD_real = criterion(output, label_real)
	        # Calculate gradients for D in backward pass
	        errSD_real.backward()

	        SD_x = output.mean().item()
	        

	        fakeDec = second_generator(second_robot_sample)

	        label_fake.fill_(main_label)
	        # Classify all fake batch with D
	        output = second_discriminator(fakeDec.detach())
	        output = output.view(-1)
	        # Calculate D's loss on the all-fake batch
	        errSD_fake = criterion(output, label_fake)
	        # Calculate the gradients for this batch
	        errSD_fake.backward()

	        SD_G_z1 = output.mean().item()
	        # Add the gradients from the all-real and all-fake batches
	        errSD = (errSD_real + errSD_fake) 

	        optimizerSD.step()
	        
	         
	################################################################################
	#################################### Second Generator
	################################################################################
	        
	        fakeDec = second_generator(second_robot_sample)

	        label_real.fill_(real_label)  # fake labels are real for generator cost
	        # Since we just updated D, perform another forward pass of all-fake batch thro"ugh D
	        output= second_discriminator(fakeDec)

	        output = output.view(-1)
	        # Calculate G's loss based on this output
	        errSG = criterion(output, label_real)
	        # Calculate gradients for G
	        errSG.backward()

	        SD_G_z2 = output.mean().item()

	        
	        
	        
	################################################################################
	#################################  Zero Gradients of Discriminators
	################################################################################
	        first_discriminator.zero_grad()
	        second_discriminator.zero_grad()

	        optimizerFD.zero_grad()
	        optimizerSD.zero_grad()


	################################################################################
	#################################  Cycle Loss
	################################################################################
	        
	        ### Currently using Mean Squared Error
	        Err1 = cycleLossCalculation( first_robot_sample, second_generator(first_generator(first_robot_sample)) )
	        Err2 = cycleLossCalculation( second_robot_sample, first_generator(second_generator(second_robot_sample)) )

	        Res = config.TRAIN.CYCLE_LAMBDA*(Err1+Err2)
	        Res.backward()

	################################################################################
	#################################  Similarity Loss
	################################################################################

	        
	        # first_sim_error = similarity(first_robot_sample,first_generator(first_robot_sample))
	        # second_sim_error = similarity(second_robot_sample,second_generator(second_robot_sample))
	        # 
	        # SimErr = (first_sim_error+second_sim_error)*config.TRAIN.SIMILARITY_RATE
	        # SimErr.backward()


	        optimizerFED.step()
	        optimizerSED.step()
	        
	        optimizerFD.step()
	        optimizerSD.step()
	        
	        
	        











	        if (i != 0) and (i % 5 == 0):
	            with open(config.OUTPUT_DIR+"output.txt","a+") as f:
	                f.write('[%d/%d][%d/%d]\t\t First GAN Loss_DF/R: %.4f/ %.4f \tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f \n'
	                 % (epoch, config.TRAIN.MAX_EPOCH, i, len(dataloader),
	                    errFD_fake.item(), errFD_real.item(), errFG.item(), FD_x, FD_G_z1, FD_G_z2))
	                f.write('[%d/%d][%d/%d]\t\t Second GAN Loss_DF/R: %.4f/ %.4f \tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f \n'
	                 % (epoch, config.TRAIN.MAX_EPOCH, i, len(dataloader),
	                    errSD_fake.item(), errSD_real.item(), errSG.item(), SD_x, SD_G_z1, SD_G_z2))
	                f.write('[%d/%d][%d/%d]\t\t Cycle Consistency Loss First->Second Error/Second->First Error: %.4f/ %.4f \t applied %.4f \n'
	                     % (epoch, config.TRAIN.MAX_EPOCH, i, len(dataloader),
	                        Err1.item(),Err2.item(),Res.item()))
	                #f.write('[%d/%d][%d/%d]\t\t similarity Loss First Gan: %.4f \t Second Gan: %.4f \t Overall: %.4f \n'
	                #     % (epoch, config.TRAIN.MAX_EPOCH, i, len(dataloader),
	                #       first_sim_error.item(),second_sim_error.item(),SimErr.item()))
	            print('[%d/%d][%d/%d]\t\t First GAN Loss_DF/R: %.4f/ %.4f \tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f \n'
	                 % (epoch, config.TRAIN.MAX_EPOCH, i, len(dataloader),
	                    errFD_fake.item(), errFD_real.item(), errFG.item(), FD_x, FD_G_z1, FD_G_z2))
	            print('[%d/%d][%d/%d]\t\t Second GAN Loss_DF/R: %.4f/ %.4f \tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f \n'
	             % (epoch, config.TRAIN.MAX_EPOCH, i, len(dataloader),
	                errSD_fake.item(), errSD_real.item(), errSG.item(), SD_x, SD_G_z1, SD_G_z2))
	            print('[%d/%d][%d/%d]\t\t Cycle Consistency Loss First->Second Error/Second->First Error: %.4f/ %.4f \t applied %.4f \n'
	                 % (epoch, config.TRAIN.MAX_EPOCH, i, len(dataloader),
	                    Err1.item(),Err2.item(),Res.item()))
	            #print('[%d/%d][%d/%d]\t\t similarity Loss First Gan: %.4f \t Second Gan: %.4f \t Overall: %.4f \n'
	            #     % (epoch, config.TRAIN.MAX_EPOCH, i, len(dataloader),
	            #        first_sim_error.item(),second_sim_error.item(),SimErr.item()))
	            
	           

	           # Save Losses for plotting later
	        FG_losses.append(errFG.item())
	        FD_losses.append(errFD.item())
	        SG_losses.append(errSG.item())
	        SD_losses.append(errSD.item())
	        CycleLoss.append(Res.item())
	        # Check how the generator is doing by saving G's output 

	        if (iters % 10 == 0) or ((epoch == config.TRAIN.MAX_EPOCH - 1) and (i == len(dataloader) - 1)):
	           with torch.no_grad():
	               SecondRobotFake = first_generator(first_robot_sample.detach())
	               FirstDiscResponse = first_discriminator(SecondRobotFake.detach())
	               FirstRobotFake = second_generator(second_robot_sample.detach())
	               SecondDiscResponse = second_discriminator(FirstRobotFake.detach())
	               FirstReconstructed = second_generator(first_generator(first_robot_sample.detach()))
	               SecondReconstructed = first_generator(second_generator(second_robot_sample.detach()))
	           
	           Fexample_outputs = SecondRobotFake[0].cpu().numpy()
	           Fexample_inputs = first_robot_sample[0].cpu().numpy()
	           Fdiscr_dec = FirstDiscResponse[0].cpu().numpy()
	           Sexample_outputs = FirstRobotFake[0].cpu().numpy()
	           Sexample_inputs = second_robot_sample[0].cpu().numpy()
	           Sdiscr_dec = SecondDiscResponse[0].cpu().numpy()

	           FirstReconstructedHolder = FirstReconstructed[0].cpu().numpy()
	           SecondReconstructedHolder = SecondReconstructed[0].cpu().numpy()
	           del SecondRobotFake
	           del FirstRobotFake

	        iters += 1
	       
	        del first_robot_sample,second_robot_sample
	        del label_real,label_fake
	        del first_robot_data,second_robot_data
	        del fakeDec
	    print(epoch)

	    if (epoch != 0) and (epoch %1 == 0):
	        plt.figure(figsize=(20,14))
	        plt.title("GAN Losses  During Training")
	        plt.plot(FG_losses,label="FGen")
	        plt.plot(FD_losses,label="FDisc")
	        plt.plot(SG_losses,label="SGen")
	        plt.plot(SD_losses,label="SDisc")
	        
	        plt.xlabel("iterations")
	        plt.ylabel("Loss")
	        plt.legend()
	        plt.savefig(config.TRAIN.GRAPH_SAVE_PATH+'graphs/Gans'+str(epoch)+'.png')
	        plt.close()

	        plt.figure(figsize=(20,14))
	        plt.title("Cycle Consistency During Training")
	        plt.plot(CycleLoss,label="CycleLoss")
	        plt.xlabel("iterations")
	        plt.ylabel("Loss")
	        plt.legend()
	        plt.savefig(config.TRAIN.GRAPH_SAVE_PATH+'Cycle'+str(epoch)+'.png')
	        plt.close()
	        write_to_file(Fexample_outputs,config.TRAIN.GRAPH_SAVE_PATH+str(epoch)+'FRes')
	        write_to_file(Fexample_inputs,config.TRAIN.GRAPH_SAVE_PATH+str(epoch)+'FInp')
	        write_res(Fdiscr_dec,config.TRAIN.GRAPH_SAVE_PATH+str(epoch)+'FDecision.txt')
	        write_to_file(Sexample_outputs,config.TRAIN.GRAPH_SAVE_PATH+str(epoch)+'SRes')
	        write_to_file(Sexample_inputs,config.TRAIN.GRAPH_SAVE_PATH+str(epoch)+'SInp')
	        write_res(Sdiscr_dec,config.TRAIN.GRAPH_SAVE_PATH+str(epoch)+'SDecision.txt')
	        write_to_file(FirstReconstructedHolder,config.TRAIN.GRAPH_SAVE_PATH+str(epoch)+'FirstReconstructed')
	        write_to_file(SecondReconstructedHolder,config.TRAIN.GRAPH_SAVE_PATH+str(epoch)+'SecondReconstructed')

	        torch.cuda.empty_cache() 
	    if (epoch %config.TRAIN.SAVE_AT == 0):
	        torch.save({
	                    'first_generator': first_generator.state_dict(),
	                    'second_generator': second_generator.state_dict(),
	                    'first_discriminator': first_discriminator.state_dict(),
	                    'second_discriminator': second_discriminator.state_dict(),
	                    'optimizerFD': optimizerFD.state_dict(),
	                    'optimizerFED': optimizerFED.state_dict(),
	                    'optimizerSED': optimizerSED.state_dict(),
	                    'optimizerSD' : optimizerSD.state_dict()
	                    }, config.TRAIN.SAVE_WEIGHTS+'saved'+str(epoch)+'.pt')


	torch.save({
	            'first_generator': first_generator.state_dict(),
	            'second_generator': second_generator.state_dict(),
	            'first_discriminator': first_discriminator.state_dict(),
	            'second_discriminator': second_discriminator.state_dict(),
	            'optimizerFD': optimizerFD.state_dict(),
	            'optimizerFED': optimizerFED.state_dict(),
	            'optimizerSED': optimizerSED.state_dict(),
	            'optimizerSD' : optimizerSD.state_dict()
	            }, config.TRAIN.SAVE_WEIGHTS+'saved.pt')
	plt.figure(figsize=(20,14))
	plt.title("GAN Losses  During Training")
	plt.plot(FG_losses,label="FGen")
	plt.plot(FD_losses,label="FDisc")
	plt.plot(SG_losses,label="SGen")
	plt.plot(SD_losses,label="SDisc")

	plt.xlabel("iterations")
	plt.ylabel("Loss")
	plt.legend()
	plt.savefig(config.TRAIN.GRAPH_SAVE_PATH+'GansResult.png')
	plt.close()

	plt.figure(figsize=(20,14))
	plt.title("Cycle Consistency During Training")
	plt.plot(CycleLoss,label="CycleLoss")
	plt.xlabel("iterations")
	plt.ylabel("Loss")
	plt.legend()
	plt.savefig(config.TRAIN.GRAPH_SAVE_PATH+'CycleResult.png')
	plt.close()

	# str1 = ','.join(str(e) for e in example_outputs[-1].cpu().numpy()[0][0])

	write_to_file(Fexample_outputs,config.TRAIN.GRAPH_SAVE_PATH+'EndFRes')
	write_to_file(Fexample_inputs,config.TRAIN.GRAPH_SAVE_PATH+'EndFInp')
	write_res(Fdiscr_dec,config.TRAIN.GRAPH_SAVE_PATH+'EndFDecision.txt')
	write_to_file(Sexample_outputs,config.TRAIN.GRAPH_SAVE_PATH+'EndSRes')
	write_to_file(Sexample_inputs,config.TRAIN.GRAPH_SAVE_PATH+'EndSInp')
	write_res(Sdiscr_dec,config.TRAIN.GRAPH_SAVE_PATH+'EndSDecision.txt')
	write_to_file(FirstReconstructedHolder,config.TRAIN.GRAPH_SAVE_PATH+'EndFirstReconstructed')
	write_to_file(SecondReconstructedHolder,config.TRAIN.GRAPH_SAVE_PATH+'EndSecondReconstructed')


def main(opts):
    load_config(opts.config)
    dataloader = lstm_cycle_dataloader(config)
    device = torch.device("cuda:0" if (torch.cuda.is_available() and config.NUM_GPUS > 0) else "cpu")
    train(dataloader, config, device)


if __name__ == "__main__":
    options, args = parser.parse_args()
    main(options)


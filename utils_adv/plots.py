import matplotlib.pyplot as plt
from pylab import genfromtxt
import os


def plot_indi_loss():
    n_epochs = []
    total_loss = []
    det_loss = []
    nps_loss = []
    tv_loss = []

    for line in open('/content/output/loss_over_epochs.txt', 'r'):
        values = [float(s) for s in line.split(' ')]
        n_epochs.append(values[0])
        total_loss.append(values[1])
        det_loss.append(values[2])
        nps_loss.append(values[3])
        tv_loss.append(values[4])

    # Plotting and saving each subplot separately
    plt.figure(figsize=(8, 6))  # Adjust the figure size as needed

    # Total Loss
    plt.plot(n_epochs, total_loss, label='Total Loss', color='black')
    plt.xlabel('Epochs')
    plt.ylabel('Total Loss')
    plt.title('Total Loss over Epochs')
    plt.savefig('/content/output/total_loss.png')
    plt.clf()  # Clear the figure for the next plot

    # Object Detection Loss
    plt.plot(n_epochs, det_loss, label='Objectness Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Object Detection Loss')
    plt.title('Object Detection Loss over Epochs')
    plt.savefig('/content/output/det_loss.png')
    plt.clf()

    # NPS Loss
    plt.plot(n_epochs, nps_loss, label='NPS Loss', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('NPS Loss')
    plt.title('NPS Loss over Epochs')
    plt.savefig('/content/output/nps_loss.png')
    plt.clf()

    # TV Loss
    plt.plot(n_epochs, tv_loss, label='TV Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('TV Loss')
    plt.title('TV Loss over Epochs')
    plt.savefig('/content/output/tv_loss.png')
    plt.clf()

def plot_sep_loss():
    #print("Filepath: ",filepath)
    n_epochs = []
    total_loss = []
    det_loss = []
    nps_loss = []
    tv_loss = []

    for line in open('./outputs/Adv/rtdetr/loss_over_epochs.txt', 'r'):
        values = [float(s) for s in line.split(' ')]
        n_epochs.append(values[0])
        total_loss.append(values[1])
        det_loss.append(values[2])
        nps_loss.append(values[3])
        tv_loss.append(values[4])

    # Plotting
    plt.figure(figsize=(16, 12))  # Adjust the figure size as needed
    plt.title("Losses over epochs")

    plt.subplot(221)
    plt.plot(n_epochs, total_loss, label='Total Loss', color='black')
    plt.xlabel('Epochs')
    plt.ylabel('Total Loss')
    
    plt.subplot(222)
    plt.plot(n_epochs, det_loss, label='Objectness Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Object Detection Loss')

    plt.subplot(223)
    plt.plot(n_epochs, nps_loss, label='NPS loss', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('NPS Loss')

    plt.subplot(224)
    plt.plot(n_epochs, tv_loss, label='TV loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('TV Loss')

    # save the plot to a file (e.g., PNG, PDF, etc.)
    plt.savefig('./outputs/Adv/rtdetr/sep_losses.png')
    # plt.savefig('./loss_tracking/yolo/complete_dataset/experiment/comparison/test_4/sep_losses.png')
    # Show the plot or save it to a file
    # plt.ion()  # Turn on interactive mode
    # plt.show()

def plot_ens_loss():
    n_epochs = []
    total_loss = []
    total_yolo_loss = []
    total_mbnt_loss = []
    det_loss = []
    nps_loss = []
    tv_loss = []
    yolo_det_loss = []
    mbnt_det_loss = []

    for line in open('./loss_tracking/ensemble/experiment/loss_over_epochs.txt', 'r'):
        values = [float(s) for s in line.split(' ')]
        n_epochs.append(values[0])
        total_loss.append(values[1])
        det_loss.append(values[2])
        nps_loss.append(values[3])
        tv_loss.append(values[4])
        total_yolo_loss.append(values[5])
        total_mbnt_loss.append(values[6])
        yolo_det_loss.append(values[7])
        mbnt_det_loss.append(values[5])


    # Plotting
    plt.figure(figsize=(12, 10))  # Adjust the figure size as needed
    plt.title("Ensemble Losses over epochs")

    plt.subplot(221)
    plt.plot(n_epochs, total_loss, label='Total Loss', color='black')
    plt.plot(n_epochs, total_yolo_loss, label='Total YOLO Loss', color='orange')
    plt.plot(n_epochs, total_mbnt_loss, label='Total Mobilenet Loss', color='purple')
    plt.xlabel('Epochs')
    plt.ylabel('Total Loss')
    plt.legend(loc="upper right")
    
    plt.subplot(222)
    plt.plot(n_epochs, det_loss, label='Objectness Loss', color='black')
    plt.plot(n_epochs, yolo_det_loss, label='YOLO Det Loss', color='orange')
    plt.plot(n_epochs, mbnt_det_loss, label='Mobilenet Det Loss', color='purple')
    plt.xlabel('Epochs')
    plt.ylabel('Object Detection Loss')
    plt.legend(loc="upper right")

    plt.subplot(223)
    plt.plot(n_epochs, nps_loss, label='NPS loss', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('NPS Loss')

    plt.subplot(224)
    plt.plot(n_epochs, tv_loss, label='TV loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('TV Loss')

    # save the plot to a file (e.g., PNG, PDF, etc.)
    plt.savefig('./loss_tracking/ensemble/experiment/ens_losses.png')
    # plt.savefig('./loss_tracking/yolo/complete_dataset/experiment/comparison/test_4/sep_losses.png')
    # Show the plot or save it to a file
    plt.show()

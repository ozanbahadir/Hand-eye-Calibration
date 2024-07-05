import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from pathlib import Path

mypath = Path().absolute().parent

def plot(loss_stats,count,n,b,l):
    train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(
        columns={"index": "epochs"})
    plt.figure(figsize=(15, 8))
    yy = 4

    sns.lineplot(data=train_val_loss_df, x="epochs", y="value", hue="variable").set_title('Train-Test Loss/Epoch')
    # plt.legend(['Train','Validation','Test'])

    textstr = '\n'.join((
        'Epochs: %s' % (n,),
        'Batch_size: %s' % (b,),
        'Learning_rate: %s' % (l,)
        # 'Noise: Normal(mean=0,std= %s)' %(std_i,)
    ))
    '''
    #props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(
        'Layer: %s' % (yy,),
        'Hidden_size: %s' % (hindex + 1,),
        'Epochs: %s' % (nindex + 1,),
        'Batch_size: %s' % (bindex + 1,),
        'Learning_rate: %s' % (lindex + 1,),
        horizontalalignment='center',
        verticalalignment='top',
    )
    '''
    # leg=plt.legend(['Layer: %s' %yy,'Hidden_size: %s' %(hindex+1),'Epochs: %s' %(nindex+1),'Batch_size: %s' %(bindex+1),'Learning_rate: %s' %(lindex+1)])
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # place a text box in upper left in axes coords
    #1.0 * (loss_stats['train'][0])
    plt.text(0.5 * n, 1.0 * (loss_stats['train'][0]), textstr, fontsize=14, verticalalignment='top',
             horizontalalignment='center', bbox=props)
    plt.xlabel("Noise_level")
    plt.ylabel("MSE_loss(mm)")
    # plt.xlim(0, 300)
    # plt.ylim(0, 2000)
    plt.savefig(str(mypath) + "/Result/Plot/" +"line_plot%s.png" % count)
    #plt.show()
def plot_noise(loss_stats,count,n,b,l):
    train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(
        columns={"index": "epochs"})
    plt.figure(figsize=(15, 8))
    yy = 4

    sns.lineplot(data=train_val_loss_df, x="epochs", y="value", hue="variable").set_title('Train-Test Loss/Epoch with Gausian Noise (0,std)')
    # plt.legend(['Train','Validation','Test'])

    textstr = '\n'.join((
        'Epochs: %s' % (n,),
        'Batch_size: %s' % (b,),
        'Learning_rate: %s' % (l,)
        # 'Noise: Normal(mean=0,std= %s)' %(std_i,)
    ))
    '''
    #props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(
        'Layer: %s' % (yy,),
        'Hidden_size: %s' % (hindex + 1,),
        'Epochs: %s' % (nindex + 1,),
        'Batch_size: %s' % (bindex + 1,),
        'Learning_rate: %s' % (lindex + 1,),
        horizontalalignment='center',
        verticalalignment='top',
    )
    '''
    # leg=plt.legend(['Layer: %s' %yy,'Hidden_size: %s' %(hindex+1),'Epochs: %s' %(nindex+1),'Batch_size: %s' %(bindex+1),'Learning_rate: %s' %(lindex+1)])
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # place a text box in upper left in axes coords
    #1.0 * (loss_stats['trainG(0,0.001)'][0]
    plt.text(0.5 * n, 100, textstr, fontsize=14, verticalalignment='top',
             horizontalalignment='center', bbox=props)
    plt.xlabel("Epoch")
    plt.ylabel("MSE_loss(mm)")
    # plt.xlim(0, 300)
    plt.ylim(0, 100)
    plt.savefig(str(mypath) + "/Calibration_image/Result/Plot/" +"line_plot%s.png" % count)
    #plt.show()
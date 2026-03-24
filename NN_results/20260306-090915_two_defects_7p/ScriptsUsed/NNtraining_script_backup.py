import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import shutil, datetime, sys, os, glob
import joblib, h5py, json
from sklearn.preprocessing import StandardScaler
import numpy as np

def par_fit_transform(par_mat):
    '''
    par_fit_transform : perform StandardScaler transformation of the input parameters into the space required by the Neural Network. 
                        before aplying StandardScaler transformation a natural log transform is done on all the inputs.
    
    Parameters
    ----------
    par_mat : numpy.array of all parameter combinations in their original form. 
    n_var : number of variable parameters

    Returns
    -------
    par_norm : standardnorm of par_mat
    scaler : scaler value for normalization 
    ''' 
    
    scaler = StandardScaler()   # define scaler object
    par_norm = scaler.fit_transform(par_mat) # transform data with the standard scaler
    return par_norm, scaler

def log_trans(paras, n_var, ln_idx):
    '''
    log_trans : take the logarithm of a certain subset of data that is varied over several orders of magnitude  
                !!! number of parameters that scales linearly must be defined here !!!
    
    Parameters
    ----------
    paras : numpy.array of data of which a subset is supposed to be transformed. one dimension must be of the size n_var 
    n_var : number of variable parameters
    ln_idx : index of the start of logscale parameter

    Returns
    -------
    mod : data with the natural logarithm of some data
    ''' 
    if np.shape(paras)[0] == n_var:  # check which dimension contains the n_var parameters
        mod = np.concatenate((paras[:ln_idx],np.log(paras[ln_idx:])),axis=0)    # take logarithm of some parameters along that axis
    elif np.shape(paras)[1] == n_var:    # check which dimension contains the n_var parameters
        mod = np.concatenate((paras[:,:ln_idx],np.log(paras[:,ln_idx:])),axis=1)     # take logarithm of some parameters along that axis
    return mod

def plot_sim(y, dir,fname):
    '''
    plot_sim : plot's input vs arbitrary output

    Parameters 
    ----------
    y : numpy.array of y-axis values. Each row is new result.
    dir : directory where it saves the plot
    fname : filename of the saved plot 

    Returns
    -------
    Nothing
    '''
    num = y.shape[0]    # find number of subplots needed

    fig,ax = plt.subplots(num,1, figsize=[10,15])   # define figure
    for i, yslice  in enumerate(y): # loop through the individual simulations
        ax[i,].plot(yslice) # plot current simulation with arbitrary x axis
    plt.xlabel('x (a.u.)')
    plt.ylabel('y (a.u.)')
    # plt.show()
    fig.savefig(os.path.join(dir,fname))  # save figure
    plt.close

def network256(x, y, training_folder, training_name, lrate, batchsize):
    '''
    network256 : NN model used for training surrogate model
           
    Parameters 
    ----------
    x : numpy.array of inputs to NN. Material parameters in their standard normalized form(par_norm). Axis = 0 should be equal to axis = 0 of y.
    y : numpy.array of y-axis values. Each row is new result.  Axis = 0 should be equal to axis = 0 of x. axis = 1 must have 512 points
    training folder : folder where the training results will be stored after completion of training.
    training_name : name of hdf file in which the weights and biases will be stored

    Saves
    -------
    reg : regression model
    reg_name : path where the regression model is stored
    x_test : the x values used for testing
    y_test : the y values used for testing
    x_train : the x_values used for training and validation
    y_train : the y values used for training and validation
    '''

    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.10, shuffle=False)    # split training data set to keep certain data set unknown to the NN
    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

    # Network Parameters
    max_filter = 256
    strides = [2,2,2,2]
    kernel = [2,2,2,2]
    map_size = 16

    nParams = X_train.shape[1]

    # define layer of the NN
    z_in = tf.keras.layers.Input(shape=(nParams,))
    z1 = tf.keras.layers.Dense(max_filter)(z_in)
    z1 = tf.keras.activations.swish(z1)
    z1 = tf.keras.layers.Dense(max_filter*map_size)(z1) #256 * 32
    z1 = tf.keras.activations.swish(z1)
    z1 = tf.keras.layers.Reshape((map_size,max_filter))(z1) # 32 by 256
    z2 = tf.keras.layers.Conv1DTranspose( max_filter//2, kernel[3], strides=strides[3], padding='SAME')(z1) # 64 by 128
    z2 = tf.keras.activations.swish(z2)
    z3 = tf.keras.layers.Conv1DTranspose(max_filter//4, kernel[2], strides=strides[2],padding='SAME')(z2) # 128 by 64
    z3 = tf.keras.activations.swish(z3)
    z4 = tf.keras.layers.Conv1DTranspose(max_filter//8, kernel[1], strides=strides[1],padding='SAME')(z3) # 256 by 32
    z4 = tf.keras.activations.swish(z4)
    z5 = tf.keras.layers.Conv1DTranspose(1, kernel[0], strides=strides[0],padding='SAME')(z4) # 512 by 1
    decoded_Y = tf.keras.activations.swish(z5)
    decoded_Y = tf.keras.layers.Reshape((Y_train.shape[1],))(decoded_Y)
    
    # in case the learning rate is not constant, define the scheduler here
    def scheduler(epoch, lr):
        if epoch <250:  # number of epoches, where the first learning rate is valid
            lr = 0.001
            return lr
        elif ((epoch>=250) & (lr>lrate)) or ((epoch>=2000) & (lr>0.0001)) :
            lr = lr*0.99 # decrease learning rate with every step
            return lr        
        else  :
            return lr
    lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)   # create the tensorflow object for the scheduler 
      
    log_folder = os.path.join(training_folder ,  training_name )# define folder where data for tensorboard and checkpoint are stored
    print(log_folder)   # print folder name so it can be copied for tensorboard
    tf_callbacks = [TensorBoard(log_dir=log_folder,     # allow tensorflow to store information for tensorboard
                        histogram_freq=1,
                        update_freq='epoch',
                         profile_batch=(2,10))]
    os.system("tensorboard --logdir " + log_folder + " --host 0.0.0.0 --port 6006 &")
    # to activate tensorboard to observe the training losses, activate the right environment in conda and use the command 'tensorboard --logdir log_folder'

    # currently, checkpoints are saved after each iteration, I might uncomment this. To load the checkpoint, use 'reg.load_weights(checkpoint_path)'
    # checkpoint_path = log_folder / ("cp.ckpt")  # define paths where the checkpoints are stored
    # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True,  verbose=1)  # define how checkpoints are made

    reg = Model(z_in,decoded_Y) # create a model with the layers defined before
    reg.summary()   # print information on model
    
    # the neural network can stop before the full number of epoches are run. Here, the criterion for this is the validation loss. 
    # If it has not varied by the min_delta, it will run another 'patience' epochs before terminating the training. Then the best 
    # weights will be used based on the mode.
    # es = EarlyStopping(monitor='val_loss', min_delta=5e-8, mode='min', verbose=1, patience=2000, restore_best_weights=True) 
    
    reg.compile(loss='mse',optimizer='adam', metrics=['mae', 'msle'])   # mean square error is used for training with the optimizer 'adam'

    # run the training of the neural network.
    reg.fit(X_train,Y_train,shuffle=False, batch_size=batchsize, epochs = 2000,
            validation_split=0.4,  callbacks=[lr_callback] + tf_callbacks, verbose = 0)
        
    reg_name = os.path.join(training_folder, "%s_trained_model.h5" % training_name)   # create path to store the neural network
    # save the neural network
    reg.save(reg_name)  
    reg.save(os.path.join(training_folder, 'model.keras'))
    return reg, reg_name, X_test, Y_test, X_train, Y_train

def check_nn(reg1, x1_test, y1_test, dir, fname):
    '''
    check_nn  : plots some of the test data of training along the prediction of the neural network

    Parameters 
    ----------
    reg1 : results of the first neural network training
    reg2 : results of the second neural network training
    train_path : file path to training data file
    dir : directory of this optimization run
    fname : name for this type of plot

    returns
    -------
    Nothing

    '''
    num = 5
    idx = np.random.randint(0, x1_test.shape[0],num)   # select random indices for plotting
    
    # idx = numpy.array([13076])
    y1_predict = reg1(x1_test[idx,:])   # let the first neural network predict data using the parameter combinations at the indices idx
    plot_sim_predict(y1_test[idx,:], y1_predict, dir, fname)   # plot the prediction together with the real simulation
    
def plot_sim_predict(y, y_predict, dir, fname):
    '''
    plot_sim : plot's y_norm and predited y

    Parameters 
    ----------
    y : numpy.array of y-axis values. Each row is new result.
    y_predict : numpy array of y-axis of predicted output
    dir : the directory where the plot is stored
    fname : the filename of the saved plot

    Returns
    -------
    Nothing
    '''
    num = y.shape[0]    # find number of subplots needed

    fig,ax = plt.subplots(num,1, figsize=[10,15])   # define figure

    for i in range(num): # loop through the individual simulations
        ax[i,].plot(y[i,:]) # plot current simulation with arbitrary x axis
        ax[i,].plot(y_predict[i,:],'--') # plot predicted values with arbitrary x axis
    plt.xlabel('x (a.u.)')
    plt.ylabel('y (a.u.)')
    # plt.show()
    fig.savefig(os.path.join(dir,fname))  # save figure
    plt.close('all')

def main(NN_name, dataset_name, lr, batch_size):

    tf.config.list_physical_devices('GPU')

    # Create result path
    cwd = os.getcwd()
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    res_dir = os.path.join(cwd, "NN_results", timestamp + '_' + NN_name)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    # Back up scripts
    backup_path = os.path.join(res_dir,"ScriptsUsed") # create folder path to copy the python script version of this run to 
    if not os.path.exists(backup_path):
        os.mkdir(backup_path)   # create folder
    current_script = sys.argv[0]
    shutil.copy2(current_script, os.path.join(backup_path, 'run_script_backup.py'))
    shutil.copy2(os.path.join(cwd, 'NNtraining.py'), os.path.join(backup_path, 'NNtraining_script_backup.py'))

    # Load input output file
    dataset_path = os.path.join('/content/drive/MyDrive/NNtraining_temp/', 'combined_SCLCdata.h5')
    with h5py.File(dataset_path, 'r') as hf:
        input_all = hf['input_all'][:]
        nn_outputs_norm = hf['nn_outputs_scaled'][:]      # normalized
        
    # Load other variables
    json_path = os.path.join(cwd, 'Datagen_results', dataset_name, 'simulation_metadata.json')
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    varied_params = metadata['varied_parameters']
    log_indices = metadata['log_indices']
    shutil.copy2(json_path, os.path.join(res_dir, 'simulation_metadata.json'))

    # Log-transform input parameters 
    input_all_log = log_trans(input_all, len(varied_params), min(log_indices))
    print(input_all_log[999,:])
    input_all_norm, scaler = par_fit_transform(input_all_log)
    scaler_name = os.path.join(res_dir,timestamp + '_' + NN_name + "_scaler.joblib")  # create filname for the scaler used
    joblib.dump(scaler, scaler_name)    # save scaler at the path 'scaler_name'

    # select some random points and plot to see if you have loaded the correct datasets.
    dir1 = os.path.join(res_dir, 'y1')
    os.mkdir(dir1)  # create directory for the first neural network
    idx = np.random.randint(0, nn_outputs_norm.shape[0],9)   # get random indices for test ploting of the training data
    plot_sim(nn_outputs_norm[idx,:], dir1, 'y1_norm.png')   # plot test subset of the training data

    # train neural network
    name1 = timestamp + '_' + NN_name + '_y1'    # define name for training folder of this neural network
    reg1, reg_name1, x1_test, y1_test, x1_train, y1_train = network256(input_all_norm, nn_outputs_norm, dir1, name1, lr, batch_size) # run neural network

    dir_nn = os.path.join(dir1, 'nn_check')
    os.mkdir(dir_nn)
    fname = "nn_check.png"
    check_nn(reg1, x1_test, y1_test, dir_nn, fname)
    

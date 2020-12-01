########## Direct Data input in model.fit version
import sys
print(sys.executable)
from F_modules import *
#from F_functions import *
from F_plotting import *
from F_utils import *
from F_models import *

# %%     init paths parameters
Save_pictures_path,local_dataset_path,Keras_models_path = F_init_paths()
Record_data_flag='read_data_time_domain2'

# Load pretrained CNN weights=0,train CNN=1
flag_train_model = 1
flag_single_gpu=0
if flag_single_gpu==0:
    import horovod.tensorflow.keras as hvd
    hvd.init()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
#  Start logging to file
log_save_const = F_calculate_log_number(Save_pictures_path, 'log', '')
Save_pictures_path = Save_pictures_path + '/log' + str(log_save_const)
F_create_folder(Save_pictures_path)
logname = '/log' + str(log_save_const) + '.txt'
logpath = Save_pictures_path
f = open(logpath + logname, 'w')
original = sys.stdout
sys.stdout = Tee(sys.stdout, f)
print('Writing log to ' + logpath + logname)
T1 = datetime.datetime.now()
print(T1)
print('Record_data_flag=', Record_data_flag)

# %%   dataset loading
dataset_to_load_name='dataset'
dataset_to_load_name='dataset_vl_gen'
# dataset_to_load_name='dataset_4'
data_path=local_dataset_path+dataset_to_load_name
Nn=-1
# Nn=10000
# Nn=30
if os.path.exists(data_path):
    print('Start loading of dataset ' + data_path)       
    files_all=fnmatch.filter(os.listdir(data_path),'*.npz')
    files=sorted(files_all)
    N=len(files)
    print('Files number=',N)
    data=np.load(data_path+'/'+files[0],allow_pickle=True)
    dx=data['dx']
    dz=dx
    data.close()
    if flag_train_model==0:
        Nn=30
        train_frac = 0.6
        val_frac = 0.4
    else:
        train_frac= 0.9
        val_frac= 0.1
    if Nn>N or Nn==-1:
        N=N
    else:
        N=Nn
    ################# dataset splitting, no shuffling
    # Train_models = np.arange(0,N * train_frac, dtype=int)   
    # Valid_models = np.setdiff1d(np.arange(0, N, 1), Train_models)
    ################# dataset splitting, shuffling
    all_models=np.arange(0,N, dtype=int).tolist()
    Train_models=random.sample(all_models,int(N*train_frac))
    Valid_models =list(set(all_models)-set(Train_models))
    Train_models=np.array(Train_models)
    Valid_models=np.array(Valid_models)
    Test_models=[]
    Test_models.append(  files_all.index('augmented_marmousi_10_it__Overthrust.npz') )
    Test_models.append(  files_all.index('augmented_marmousi_10_it__Marmousi.npz') )
    ###############################################################
    for ii in Test_models:
        if [ii] in Train_models:
            Train_models=np.delete(Train_models,np.where(ii==Train_models))
        if [ii] in Valid_models:
            Valid_models=np.delete(Valid_models,np.where(ii==Valid_models))
    ######## Overfitting case for four testing models
    list_train=[]
    for i in range(len(Train_models)):
        list_train.append(data_path+'/'+files[Train_models[i]])
    list_valid=[]
    for i in range(len(Valid_models)):
        list_valid.append(data_path+'/'+files[Valid_models[i]])
    list_test=[]
    for i in range(len(Test_models)):
        list_test.append(data_path+'/'+files_all[Test_models[i]])
    list_all=list_train+list_valid+list_test
    N=len(list_all)
else:
    print('dataset does not exist')
    exit()
print('Models for training:',  len(Train_models), ' out of ',N, 'Overall models')
print('Models for validation:',len(Valid_models), ' out of ',N, 'Overall models')
print('Models for testing:',   len(Test_models), ' out of ',N, 'Overall models')
# %%   CNN hyperparameters
batch_size=2
epochs=2
test_status=0
#   data loader for training, validation
training_generator=DataLoader(list_train,batch_size,shuffle=True,   to_fit=True)
validation_generator=DataLoader(list_valid,batch_size,shuffle=True, to_fit=True)
testing_generator=DataLoader(list_test,batch_size=1,shuffle=True,to_fit=True)

pars_h={'epochs':epochs,'batch_size':batch_size,'learning_rate':0.001,
          'training_type': 2, 'loss': 'mse',
          'patch_sz_x':training_generator.Nx_in,'patch_sz_z':training_generator.Nz_in, 
          'strides_x':training_generator.Nx_in, 'strides_z': training_generator.Nz_in,
          'reg_value': 0.0, 'Nm_to_invert': 200,
          'val_shuffle': True, 'flag_parallel': False}
save_best_only=True
# %%     callbacks for training
filename=Keras_models_path + '/pars_' + str(log_save_const) + '.hdf5'
early_stopping = EarlyStopping(monitor='val_coeff_determination', patience=14, min_delta=0.000010, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', min_delta=0.001, factor=0.5, patience=15, min_lr=1e-8, verbose=1,mode='auto')
terminate = TerminateOnNaN()
log_dir = "logs/profile/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1,profile_batch='10, 15')
checkpoint = ModelCheckpoint(Keras_models_path + '/weights_best' + str(log_save_const) + '.hdf5', verbose=1,
    monitor='val_loss',mode='min',save_best_only=save_best_only, save_weights_only=True,save_freq='epoch')
# Callbacks = [terminate,checkpoint,early_stopping]  #add early_stopping??
# Callbacks = [terminate,checkpoint,early_stopping,tensorboard_callback]  #add early_stopping??
Callbacks = [terminate,checkpoint]  #add early_stopping??
print(pars_h);  print(Callbacks)
# %%     DL MODEL initialization
input_shape=(training_generator.Nx_in,training_generator.Nz_in,1)
print('input picture shape=',input_shape)
model = auto_encoder2_deep4(input_shape)
model.summary()

flag_input_pipeline='dataloader'
flag_input_pipeline='tfdataset'

data_train=training_generator.__getdataset__()
x_train=data_train[0]
t_train=data_train[1]
data_valid=validation_generator.__getdataset__()
x_valid=data_valid[0]
t_valid=data_valid[1]
data_test=testing_generator.__getdataset__()
x_test=data_test[0]
t_test=data_test[1]
flag_do_scaling=1
if flag_do_scaling==1:
    # from sklearn.externals.joblib import dump,load
    from joblib import dump,load
    if flag_train_model==1:
        x_train_,scaler_x=scaling_data(x_train)
        x_valid_=transforming_data(x_valid,scaler_x)
        x_test_=transforming_data(x_test,scaler_x)
        t_train_,scaler_t=scaling_data(t_train)
        t_valid_=transforming_data(t_valid,scaler_t)
        t_test_=transforming_data(t_test,scaler_t)
        dump(scaler_x,Keras_models_path+'/scaler_x_'+str(log_save_const)+'.bin',compress=True)
        dump(scaler_t,Keras_models_path+'/scaler_t_'+str(log_save_const)+'.bin',compress=True)
    else:
        scaler_x=load(Keras_models_path+'/scaler_x_'+str(Model_to_load_const)+'.bin')
        scaler_t=load(Keras_models_path+'/scaler_t_'+str(Model_to_load_const)+'.bin')
        x_train_=transforming_data(x_train,scaler_x)
        x_valid_=transforming_data(x_valid,scaler_x)
        x_test_=transforming_data(x_test,scaler_x)
        t_train_=transforming_data(t_train,scaler_t)
        t_valid_=transforming_data(t_valid,scaler_t)
        t_test_=transforming_data(t_test,scaler_t)
else:
    x_train_=x_train
    x_valid_=x_valid
    x_test_=x_test
    t_train_=t_train
    t_valid_=t_valid
    t_test_=t_test
       
if flag_train_model == True:
    metrics=['MAPE',coeff_determination]
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train_,t_train_))
    valid_dataset = tf.data.Dataset.from_tensor_slices((x_valid_,t_valid_))
    train_dataset=train_dataset.shuffle(Nn+2000).batch(batch_size)
    valid_dataset=valid_dataset.shuffle(Nn+2000).batch(batch_size)
    if flag_single_gpu==1:
        a=1
    else:
        #########multi gpu-version
        # Horovod: pin GPU to be used to process local rank (one GPU per process)
        # gpus = tf.config.experimental.list_physical_devices('GPU')
        # for gpu in gpus:
        #     tf.config.experimental.set_memory_growth(gpu, True)
        # if gpus:
        #     tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
        # Optimizer
        scaled_lr = pars_h['learning_rate'] * hvd.size()
        opt = tf.optimizers.Nadam(scaled_lr)
        opt = hvd.DistributedOptimizer(opt)
        # Horovod: Specify `experimental_run_tf_function=False` to ensure TensorFlow
        # uses hvd.DistributedOptimizer() to compute gradients.
        model.compile(loss='mse',
                    optimizer=opt,
                    # metrics=['accuracy'],
                    metrics=metrics,
                    experimental_run_tf_function=False)
        callbacks = [
        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        # Horovod: average metrics among workers at the end of every epoch.
        # Note: This callback must be in the list before the ReduceLROnPlateau,
        # TensorBoard or other metrics-based callbacks.
        hvd.callbacks.MetricAverageCallback(),
        # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
        # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
        # the first three epochs. See https://arxiv.org/abs/1706.02677 for details.
        hvd.callbacks.LearningRateWarmupCallback(initial_lr=scaled_lr, warmup_epochs=3, verbose=1),]

        # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
        if hvd.rank() == 0:
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(Save_pictures_path+'/checkpoint-{epoch}.h5'))
        # Horovod: write logs on worker 0.
        verbose = 1 if hvd.rank() == 0 else 0
        # Train the model.
        # Horovod: adjust number of steps based on number of GPUs.
        # model.fit(dataset, steps_per_epoch=500 // hvd.size(), callbacks=callbacks, epochs=24, verbose=verbose)
        print('start of training')
        T3 = datetime.datetime.now()
        history=model.fit(x=train_dataset,validation_data=valid_dataset,steps_per_epoch=500 // hvd.size(), callbacks=callbacks, epochs=pars_h['epochs'], verbose=verbose)
        T4 = datetime.datetime.now()
        print('Training time', T4 - T3)
        # model.fit(x=train_dataset,
        #     validation_data=valid_dataset,
        #     epochs=pars_h['epochs'],
        #     verbose=2,shuffle=True,callbacks=callbacks)
    history = history.history
    print(history)
    F_save_history_to_file(history,Keras_models_path,log_save_const)
    Model_to_load_const=log_save_const
else:
    fname=Keras_models_path + '/weights_best' + str(Model_to_load_const)+'.hdf5';print(fname)
    model.load_weights(fname,by_name=1)
    history = F_load_history_from_file(Keras_models_path,Model_to_load_const)
    predict_using_model=1

print('predictions')
if flag_do_scaling==1:
    t_predicted_train_ =model.predict(x_train_,verbose=1)
    t_predicted_valid_ =model.predict(x_valid_,verbose=1)
    t_predicted_test_ = model.predict(x_test_,verbose=1)
    t_predicted_train=transforming_data_inverse(t_predicted_train_,scaler_t)
    t_predicted_valid=transforming_data_inverse(t_predicted_valid_,scaler_t)
    t_predicted_test=transforming_data_inverse(t_predicted_test_,scaler_t)
else:
    t_predicted_train_ =model.predict(x_train_,verbose=1)
    t_predicted_valid_ =model.predict(x_valid_,verbose=1)
    t_predicted_test_ = model.predict(x_test_,verbose=1)
    t_predicted_train=t_predicted_train_
    t_predicted_valid=t_predicted_valid_
    t_predicted_test=t_predicted_test_

print('calculate misfits')
misfit_stats,idx_train_best,idx_train_worst,train_misfits,valid_misfits,\
    test_misfits,train_best_val,train_worst_val,train_best_file,train_worst_file=   \
F_calculate_misfits_across_files(list_all,list_test,list_train,list_valid,
    t_predicted_train,t_predicted_valid,t_predicted_test,
    1,Train_models,Test_models,Valid_models,print_flag=1,test_status=test_status)
print('calculating misfits finished')

# %%  plot losses
Plot_loss(history, Title='log' +      str(log_save_const) + 'loss_mse', Save_pictures_path=Save_pictures_path, Save_flag=1)
Plot_accuracy2(history, Title='log' + str(log_save_const) + 'r2accuracy', Save_pictures_path=Save_pictures_path,Save_flag=1)
# %%    Plotting pictures
Flag_plot_sections=1
predict_using_model=1
if Flag_plot_sections == 1:
    val = 2
    if N < val:
        val = N
    list_all=list_train[0:val]+list_valid[0:val]+list_test
    list_all.append(train_best_file)
    list_all.append(train_worst_file)
    # list_all=list_test
    t_predicted_all=np.concatenate((t_predicted_train[0:val], \
             t_predicted_valid[0:val],t_predicted_test,       \
             t_predicted_train[idx_train_best],t_predicted_train[idx_train_worst]),axis=0)
    t_predicted_all_=np.concatenate((t_predicted_train_[0:val], \
             t_predicted_valid_[0:val],t_predicted_test_,       \
             t_predicted_train_[idx_train_best],t_predicted_train_[idx_train_worst]),axis=0)
    if flag_input_pipeline=='tfdataset' and flag_do_scaling==1:
        x_all_=np.concatenate((x_train_[0:val], \
             x_valid_[0:val],x_test_,       \
             x_train_[idx_train_best],x_train_[idx_train_worst]),axis=0)
        t_all_=np.concatenate((t_train_[0:val], \
             t_valid_[0:val],t_test_,       \
             t_train_[idx_train_best],t_train_[idx_train_worst]),axis=0)
    i_x=0
    flag_not_split_models=1
    for NAME in list_all:
        with open(NAME, 'rb') as f:
            data=np.load(f)
            M0=data['models'][0,:,:,0]
            dz=data['dz']
            dx=data['dx']
            if test_status==0:
                Minit=data['models_init'][0,:,:,0]
                M1=data['input_data']
                if len(M1.shape)==3:
                    M1=np.expand_dims(M1,axis=0)
                M1=M1[0,:,:,0]
                M2=data['output_data'][0,:,:,0]
            data.close()
        if flag_input_pipeline=='tfdataset' and flag_do_scaling==1:
            M1=x_all_[i_x, :, :, 0]
            M2=t_all_[i_x, :, :, 0]
            M3=t_predicted_all_[i_x, :, :, 0]
        else:
            M3=t_predicted_all[i_x, :, :, 0]
        if (NAME==list_test[0] or NAME==list_test[1]) and predict_using_model==0 and flag_train_model==0:
            # number=192
            # number=134
            number=Model_to_load_const
            if NAME==list_test[0]:
                tmp='./Pictures/log'+str(number)+'/_10_it__Overthrust_weights_'+str(number)+'.npz'
            if NAME==list_test[1]:
                tmp='./Pictures/log'+str(number)+'/_10_it__Marmousi_weights_'+str(number)+'.npz'
            print(tmp)
            with open(tmp,'rb') as f:
                data=np.load(f)
                M0=data['models']
                Minit=data['models_init']
                M1=data['input_data']
                M2=data['output_data']
                M3=data['predicted_update']
                dx=data['dx']
                dz=data['dz']
                data.close()
            print('Attention')
            Models_init=Minit
            Predicted_update=M3
            M3=imresize(M3,M2.shape)
            Plot_image4(Models_init.T,Show_flag=0,Save_flag=1,Aspect='equal',
            Save_pictures_path=Save_pictures_path,Title='',fname='init')
            Plot_image4(M1.T,Show_flag=0,Save_flag=1,Aspect='equal',
            Save_pictures_path=Save_pictures_path,Title='',fname='input_data')
            Plot_image4(M2.T,Show_flag=0,Save_flag=1,Aspect='equal',
            Save_pictures_path=Save_pictures_path,Title='',fname='output_data')
            Plot_image4(Predicted_update.T,Show_flag=0,Save_flag=1,Aspect='equal',
            Save_pictures_path=Save_pictures_path,Title='',fname='Predicted_update')
            # exit()
        else:
            Predicted_update=t_predicted_all[i_x, :, :, 0]
            Predicted_update=imresize(Predicted_update,M1.shape)
            Models_init=Minit
        testing_model=Models_init+Predicted_update
        ################### Crop testing models for better visualization
        print(NAME)
        # if NAME in list_test:
        if NAME in list_test:
            if NAME==list_test[0]:
                sz_x = (850*20//dx)
            elif NAME==list_test[1]:
                sz_x = (900*20//dx)
            else:
                sz_x = (900*20//dx)
            Nx=M0.shape[0]
            edges=math.floor((Nx-sz_x)/2)
            ix1=edges;      ix2=Nx-edges
            print('M0=',M0.shape)
            M0_show=M0[ix1:ix2,:]
            print('M0_show=',M0_show.shape)
            testing_model=testing_model[ix1:ix2,:]
            water=np.ones((M0_show.shape[0],18))*1500
            M0_show=np.concatenate([water,M0_show],axis=1)
            testing_model=np.concatenate([water,testing_model],axis=1)
        else:
            M0_show=M0
        inp_orig_sizes=[M1,M2,M3,testing_model,M0_show]

        saving_name=NAME.split('augmented_marmousi_10_it')[-1]
        saving_name=saving_name.split('.npy')[0]
        # Plot_image(testing_model.T,Show_flag=1,Save_flag=1,Title='testing_model1'+saving_name,Aspect='equal',Save_pictures_path=Save_pictures_path)
        ####
        Prediction_accuracy=F_r2(M3,M2)
        R2val=F_r2(testing_model,M0_show)
        if flag_not_split_models == 1 and NAME in list_train:
            data_type='Train'
            if NAME==train_best_file:
                data_type='Train_best_score'
            elif NAME==train_worst_file:
                data_type='Train_worst_score'
        elif flag_not_split_models == 1 and NAME in list_test:
            data_type='Test'
            files[i_x]
            print(files[i_x])
            tmp2=NAME.split('augmented_marmousi')
            if flag_train_model==1:
                path =Save_pictures_path +'/'+ tmp2[1][0:-4]+'_weights_'+str(Model_to_load_const)
                print('Saving weights to=',path)
                np.savez(path,input_data=M1,output_data=M2,
                models_init=Models_init,models=M0,predicted_update=Predicted_update,dx=dx,dz=dz)
        elif flag_not_split_models == 1 and NAME in list_valid:
            data_type='Valid'
        tmp=NAME.split('augmented_marmousi_10_it')[-1]
        tmp=tmp.split('.npy')[0]
        print('tmp=',tmp)

        if NAME in list_test:
            data_type = '_' + data_type+tmp+'_'+numstr(R2val)
            title=data_type + ', R2(prediction, target) = ' + numstr(Prediction_accuracy)
            title='Prediction, R2(prediction, target) = ' + numstr(Prediction_accuracy)
        else:
            data_type = '_' + data_type+tmp+'_'+numstr(R2val)
            title=data_type + ', R2(prediction, target) = ' + numstr(Prediction_accuracy)
            title='Prediction, R2(prediction, target) = ' + numstr(Prediction_accuracy)
        Name=Save_pictures_path + '/' + 'log' + str(log_save_const) + data_type+'.png'
        print(Save_pictures_path)
        print(Name)
        # exit()
        # title2=', R2 accuracy curve train/test='+numstr(history['coeff_determination'][-1])+'/'+numstr(history['val_coeff_determination'][-1])
        # title2=', R2 accuracy curve train/test='+numstr( history['coeff_determination'][len(history['coeff_determination'])-1] )+'/'+numstr(history['val_coeff_determination'][len(history['val_coeff_determination'])-1])
        title2=''
        print('dx=',dx)
        print('dz=',dz)
        PLOT_ML_Result4(inp_orig_sizes,numstr(R2val),history_flag=0,history=None,Boundaries=[],save_file_path=Name,
            dx=dx, dy=dz, Title=title, Title2=title2, Save_flag=1)
            # Title=title, Title2=title2, Save_flag=1)        #dx,dz
        i_x=i_x+1

T2 = datetime.datetime.now()
print('Program execution time', T2 - T1)
print("FINISH!")
# %tensorboard --logdir {log_dir}

#single gpu-version
# history=model.fit(x=x_train_,y=t_train_,
#     validation_data=(x_valid_,t_valid_),
#     batch_size=batch_size,
#     epochs=pars_h['epochs'],
#     verbose=2,shuffle=True,callbacks=Callbacks)
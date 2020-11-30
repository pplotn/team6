#   ML  utils
from F_modules import *
from F_utils import *
from F_plotting import *
from tensorflow.keras import Model, Input
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
# %%    networks to try
def auto_encoder2_deep4(input_shape):
    print(input_shape)
    model = tf.keras.models.Sequential()
    ##########  Convolution block   ############
    model.add(layers.Conv2D(128,(8,8),strides=(2,2),activation='elu', input_shape=(input_shape[0], input_shape[1], 1), padding='valid'))
    model.add(BatchNormalization())
    # model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(64,(7, 7),strides=(2,1),activation='elu', padding='valid'))
    model.add(BatchNormalization())
    # model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(32,(6,6) ,strides=(2,1),activation='elu', padding='valid'))
    model.add(BatchNormalization())
    # model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(2,(5,5),strides=(1,1),activation='elu', padding='valid'))
    model.add(BatchNormalization())
    # model.add(layers.Dropout(0.2))
    ##########  Dense block   ############
    a=(model.layers[-1])
    nx=a.output_shape[1];   nz=a.output_shape[2]
    model.add(layers.Flatten())
    l1_reg=1e-3
    l2_reg=1e-3
    Dropout_val=0.0
    print('L1 reg value=',l1_reg)
    print('L2 reg value=',l2_reg)
    print('Dropout_val=',Dropout_val)
    # model.add(layers.Dense(nx*nz))
    # model.add(layers.Dense(nx*nz,kernel_regularizer=regularizers.l2(0.3)))      # l1_l2(l1=1e-5, l2=1e-4)
    model.add(layers.Dense(nx*nz,kernel_regularizer=regularizers.l1_l2(l1=l1_reg,l2=l2_reg)))      # l1_l2(l1=1e-5, l2=1e-4)
    # model.add(layers.Dropout(Dropout_val))
    # model.add(BatchNormalization())
    model.add(layers.Reshape((nx,nz,1)))
    return model
def auto_encoder2_deep5(input_shape):
    print(input_shape)
    model = tf.keras.models.Sequential()
    ##########  Convolution block   ############
    model.add(layers.Conv2D(128,(8,8),strides=(2,2),activation='elu', input_shape=(input_shape[0], input_shape[1], 1), padding='valid'))
    model.add(BatchNormalization())
    # model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(64,(7, 7),strides=(2,1),activation='elu', padding='valid'))
    model.add(BatchNormalization())
    # model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(32,(6,6) ,strides=(2,1),activation='elu', padding='valid'))
    model.add(BatchNormalization())
    # model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(2,(5,5),strides=(1,1),activation='elu', padding='valid'))
    model.add(BatchNormalization())
    # model.add(layers.Dropout(0.2))
    ##########  Dense block   ############
    a=(model.layers[-1])
    nx=a.output_shape[1];   nz=a.output_shape[2]
    # model.add(layers.Flatten())
    # l1_reg=1e-3
    # l2_reg=1e-3
    # Dropout_val=0.0
    # print('L1 reg value=',l1_reg)
    # print('L2 reg value=',l2_reg)
    # print('Dropout_val=',Dropout_val)
    # # model.add(layers.Dense(nx*nz))
    # # model.add(layers.Dense(nx*nz,kernel_regularizer=regularizers.l2(0.3)))      # l1_l2(l1=1e-5, l2=1e-4)
    # model.add(layers.Dense(nx*nz,kernel_regularizer=regularizers.l1_l2(l1=l1_reg,l2=l2_reg)))      # l1_l2(l1=1e-5, l2=1e-4)
    # # model.add(layers.Dropout(Dropout_val))
    # # model.add(BatchNormalization())
    # model.add(layers.Conv2D(1,(1,1),strides=(1,1),activation='sigmoid'))
    model.add(layers.Conv2D(1,(1,1),strides=(1,1),activation='elu'))
    # conv10 = Conv2D(1, 1, activation='sigmoid')
    # model.add(layers.Reshape((nx,nz,1)))
    return model
def load_true_data(NAME,test_status):
    with open(NAME, 'rb') as f:
        data=np.load(f,allow_pickle=True)
        if test_status==0:
            M2=data['output_data'][0,:,:,0]
        if test_status==1:
            M2=data['output_data'][0,:,:,0]
            output_data=Resize_data_deep2(data['input_data'])
            Nx_out=output_data.shape[1]
            Nz_out=output_data.shape[2]
            M2=imresize(M2,[Nx_out,Nz_out])
            M2.fill(1)
        if test_status==2:
            output_data=Resize_data_deep2(data['input_data'])
            Nx_out=output_data.shape[1]
            Nz_out=output_data.shape[2]
            x=data['input_data'][0,:,:,0]
            x=F_smooth(x, sigma_val=int(400/20))
            M2=imresize(x,[Nx_out,Nz_out])
        if test_status==3:
            inp=data['input_data'][0,:,:,0]
            M1=imresize( data['output_data'][0,:,:,0],inp.shape )
            
            M2=data['output_data'][0,:,:,0]
            output_data=Resize_data_deep2(data['input_data'])
            Nx_out=output_data.shape[1]
            Nz_out=output_data.shape[2]
            M2=imresize(M2,[Nx_out,Nz_out])
        data.close()

        true=np.expand_dims(M2,axis=0)
        true=np.expand_dims(true,axis=-1)
    return true
def load_numpy_file(name):
    with open(name,'rb') as f:
        data=np.load(f,allow_pickle=True)
        input_data =data['input_data']
        output_data=data['output_data']
        dx=data['dx']
        data.close()
    return input_data,output_data,dx

class DataGenerator(Sequence):
    def __init__(self, x, t, pars,shuffle=True,to_fit=True,to_predict=False):
        """Initialization
        :param list_IDs: list of all 'label' ids to use in the generator
        :param labels: list of image labels (file names)
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self.dim = np.asarray( [pars['patch_sz_x'],pars['patch_sz_z']],dtype=int)
        self.stride=np.array([pars['strides_x'],pars['strides_z']],dtype=int)
        Nm=x.shape[0];Nx=x.shape[1];Nz=x.shape[2]
        if to_predict==False:
            ind_x=mov_window(np.arange(Nx),self.dim[0],self.stride[0])
            ind_z =mov_window(np.arange(Nz), self.dim[1],self.stride[1])
        else:
            ind_x = mov_window(np.arange(Nx), self.dim[0], self.stride[0])
            ind_z = mov_window(np.arange(Nz), self.dim[1], self.stride[1])
            # ind_x = mov_window0(np.arange(Nx), self.dim[0], self.stride[0])
            # ind_z = mov_window0(np.arange(Nz), self.dim[1], self.stride[1])
        self.Npx=(ind_x).shape[0]
        self.Npz=(ind_z).shape[0]
        self.Nm=Nm
        self.Nch_in=x.shape[-1]
        self.Nch_out=t.shape[-1]
        self.Ns=Nm*self.Npx*self.Npz
        self.ind=np.empty((self.Ns,3,2),dtype=int);n=0
        for i in range(Nm):
            for j in range(self.Npx):
                for k in range(self.Npz):
                    self.ind[n,:,:]=np.array([[i,i],
                        [ind_x[j,0],ind_x[j,-1]],
                        [ind_z[k,0],ind_z[k,-1]]],dtype=int)
                    n+=1
                # print(self.ind[n, :, :])
        self.list_IDs =np.arange(self.Ns)
        if to_predict==False:
            self.batch_size = pars['batch_size']
        else:
            self.batch_size = pars['batch_size']
        self.t = t
        self.x = x
        self.shuffle = shuffle
        self.to_fit=to_fit
        self.on_epoch_end()
    def return_self(self):
        return self
    def __len__(self):
        'Denotes the number of batches per epoch'
        # if self.Ns%self.batch_size>0:
        #     print("Reminder of (Ns%batch_size):", self.Ns%self.batch_size)
        return int(np.floor(self.Ns / self.batch_size))
        # return int(np.ceil(self.Ns / self.batch_size))
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        if self.to_fit:
            X= self.__data_generation(list_IDs_temp,'x')
            y = self.__data_generation(list_IDs_temp,'y')
            return X, y
        else:
            X = self.__data_generation(list_IDs_temp,'x')
            return X
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    def __data_generation(self, list_IDs_temp,data_type):
        # Initialization
        if data_type=='x':
            out = np.empty((self.batch_size, self.dim[0], self.dim[1], self.Nch_in))
            for i, id in enumerate(list_IDs_temp):
                out[i, :, :, :] = self.x[self.ind[id, 0, 0], self.ind[id, 1, 0]:self.ind[id, 1, 1] + 1,
                                self.ind[id, 2, 0]:self.ind[id, 2, 1] + 1,:]
        if data_type == 'y':
            out = np.empty((self.batch_size, self.dim[0], self.dim[1], self.Nch_out))
            for i, id in enumerate(list_IDs_temp):
                out[i, :, :, :] = self.t[self.ind[id, 0, 0], self.ind[id, 1, 0]:self.ind[id, 1, 1] + 1,
                                self.ind[id, 2, 0]:self.ind[id, 2, 1] + 1,:]
        return out
    def return_reconsctructed_data(self,data):
        #   return predicted image, constructed from overlapping patches
        data_r = np.zeros(self.t.shape, dtype=self.x[0,0,0].dtype)
        ind=self.ind
        for n in range(self.Ns):
            data_r[
            ind[n,0,0],
            ind[n,1,0]: ind[n,1,1]+1,
            ind[n,2,0]: ind[n,2,1]+1,
            :] = data[n,::]
        return data_r
    def return_reconsctructed_data2(self,data):
        #   return predicted patches
        data_r = np.zeros((self.Nm,self.Npx,self.Npz,self.dim[0],self.dim[1],data.shape[-1]))
        n=0
        for i in range(self.Nm):
            for j in range(self.Npx):
                for k in range(self.Npz):
                    data_r[i,j,k,:,:,:] = data[n, :, :, :]
                    n += 1
        return data_r


class DataLoader(tf.keras.utils.Sequence):
    def __init__(self,files_list,batch_size,shuffle=True,to_fit=True,to_predict=False):
        """Initialization
        :param list_IDs: list of all 'label' ids to use in the generator
        :param labels: list of image labels (file names)
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param shuffle: True to shuffle label indexes after every epoch
        """

        input_data,output_data,dx=load_numpy_file(files_list[0])
        self.files=files_list

        # self.Nx_in=(input_data.shape[1]//8) * 8
        # self.Nz_in=(input_data.shape[2]//8) * 8
        self.Nx_in=(input_data.shape[1])
        self.Nz_in=(input_data.shape[2])
        self.Nx_out=output_data.shape[1]
        self.Nz_out=output_data.shape[2]
        self.Ns=len(files_list)
        self.Nch_in=1
        self.Nch_out=1
        self.list_IDs=np.arange(self.Ns)
        
        if to_predict==False:
            self.batch_size=batch_size
        else:
            self.batch_size=batch_size
        self.shuffle = shuffle
        self.to_fit=to_fit
        self.on_epoch_end()
    def return_self(self):
        return self
    def __len__(self):
        return int(np.floor(self.Ns / self.batch_size))
    def __getitem__(self, index):
        # 'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp=[self.list_IDs[k] for k in indexes]
        # Generate data
        if self.to_fit:
            X= self.__data_generation(list_IDs_temp,'x')
            y=self.__data_generation(list_IDs_temp,'y')
            return X, y
        else:
            X = self.__data_generation(list_IDs_temp,'x')
            return X
    def __getdataset__(self):
        # 'Generate one batch of data'
        # Find list of IDs
        list_IDs_temp = np.arange(self.Ns)
        # Generate data
        if self.to_fit:
            X= self.__data_generation(list_IDs_temp,'x')
            y=self.__data_generation(list_IDs_temp,'y')
            return X, y
        else:
            X = self.__data_generation(list_IDs_temp,'x')
            return X
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    def __data_generation(self, list_IDs_temp,data_type):
        # Initialization
        if data_type=='x':
            out=np.empty((len(list_IDs_temp),self.Nx_in,self.Nz_in,self.Nch_out))
        else:
            out=np.empty((len(list_IDs_temp),self.Nx_out,self.Nz_out,self.Nch_out))
        for i, id in enumerate(list_IDs_temp):
            NAME=self.files[id]
            with open(NAME, 'rb') as f:
                # data=np.load(f)
                data=np.load(f,allow_pickle=True)
                if data_type=='x':
                    x=data['input_data']
                else:
                    x=data['output_data']
                data.close()
            # x2 = StandardScaler().fit(x)
            # print('data type=', data_type)
            # print('data shape=',     out.shape)
            out[i,:,:,:]=x
        return out
def scaling_data(data):
    shape_orig=data.shape
    N=shape_orig[0]
    Nx=shape_orig[1]
    Nz=shape_orig[2]
    data=np.squeeze(data)
    data2=np.reshape(data,(N,Nx*Nz))
    scaler = StandardScaler()
    # scaler = MinMaxScaler(feature_range=(-1,1))
    data3 = scaler.fit_transform(data2)
    data4=np.reshape(data3,shape_orig)
    return data4,scaler
def transforming_data(data,scaler):
    shape_orig=data.shape
    N=shape_orig[0]
    Nx=shape_orig[1]
    Nz=shape_orig[2]
    data=np.squeeze(data)
    data2=np.reshape(data,(N,Nx*Nz))
    data3 = scaler.transform(data2)
    data4=np.reshape(data3,shape_orig)
    return data4
def transforming_data_inverse(data,scaler):
    shape_orig=data.shape
    N=shape_orig[0]
    Nx=shape_orig[1]
    Nz=shape_orig[2]
    data=np.squeeze(data)
    data2=np.reshape(data,(N,Nx*Nz))
    data3 = scaler.inverse_transform(data2)
    data4=np.reshape(data3,shape_orig)
    return data4


def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )
    # return (1 - SS_res / (SS_tot)  )
def coeff_determination_neg(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return -( 1 - SS_res/(SS_tot + K.epsilon()) )

def F_save_history_to_file(history,Keras_models_path,log_save_const):
    hist_df = pd.DataFrame(history)
    tmp= Keras_models_path + '/loss_history' + str(log_save_const) + '.csv'
    with open(tmp, mode='w') as f:
        hist_df.to_csv(f)
    print('Saving history to' + tmp)
    return None
def F_save_model_weights_to_file(model,Keras_models_path,log_save_const):
    fname = Keras_models_path + '/weights_' + str(log_save_const) + '.hdf5'
    print('Saving weights to' + fname)
    model.save_weights(fname)
    return None
def F_load_history_from_file(Keras_models_path,Model_to_load_const):
    tmp = Keras_models_path + '/loss_history' + str(Model_to_load_const) + '.csv'
    print('Loading history from file' + tmp)
    if os.path.isfile(tmp):
        history = pd.read_csv(tmp)
        return history
    else:
        return None

#   my parallelization attempts
def training(model,x_train,t_train,x_valid,t_valid,x,t,pars_h,
    pars_d,log_save_const,Callbacks='',flag_train_model=True):
    Save_pictures_path,local_dataset_path,extra_path,Keras_models_path,Dropbox_Access_token,dropbox_dataset_path=F_init_paths()
    Nm_to_invert=pars_h['Nm_to_invert']
    LOSS=pars_h['loss']
    Nm = x.shape[0];Nch = x.shape[-1]
    Nx = x.shape[1];Nz = x.shape[2]
    start = time.time()
    # model compile
    if pars_h['flag_parallel'] == False:
        a=1
        # model = unet_1(input_shape, pars_h['reg_value'])
        # # model = multi_gpu_model(model, gpus=16)
        # opt = Adam(lr=pars_h['learning_rate'])
        # model.compile(loss=LOSS,optimizer=opt,metrics=[coeff_determination])
        # model.summary()
    else:
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = unet_1(input_shape, pars_h['reg_value'])
            opt = Adam(learning_rate=pars_h['learning_rate'])
            model.compile(opt, loss=LOSS, metrics=['mse'])
    # model training
    N_workers=multiprocessing.cpu_count()-4
    #   multi-gpu model
    if pars_h['training_type'] == 4:
        # model = multi_gpu_model(model, gpus=1)
        pars_h['training_type']=1
    #   TF 1.12,Keras 2.2.training with generator is not working well
    if pars_h['training_type'] == 1:
        if flag_train_model==True:
            TT2=TicToc();TT2.tic()
            training_generator = DataGenerator(x_train,t_train,pars_h,shuffle=True,to_fit=True)
            validation_generator = DataGenerator(x_valid,t_valid,pars_h,shuffle=True,to_fit=True)
            # a1=training_generator.return_reconsctructed_data(x_train)
            # a2=training_generator.return_reconsctructed_data2(x_train)
            # history = model.fit(training_generator,
            #     epochs=pars_h['epochs'],
            #     verbose=2, shuffle=True, callbacks=Callbacks,
            #     validation_split=0.2,
            #     use_multiprocessing=True)
            # steps_per_epoch=ceil(n_points / pars_h['batch_size'])
            steps_per_epoch_train=training_generator.__len__();     print(steps_per_epoch_train)
            steps_per_epoch_valid=validation_generator.__len__();     print(steps_per_epoch_valid)
            # N_workers =1
            # checkpoint = ModelCheckpoint(Keras_models_path + '/weights_best' + str(log_save_const) + '.hdf5', verbose=1,
            #     save_best_only=True, save_weights_only=True)
            # K.tensorflow_backend.set_session(tf.Session(config=config))

            history=model.fit_generator((training_generator),
                validation_data=(validation_generator),workers=N_workers,
                epochs=pars_h['epochs'],validation_steps=steps_per_epoch_valid,
                verbose=2, shuffle=True, callbacks=Callbacks,
                use_multiprocessing=False, steps_per_epoch=steps_per_epoch_train)

            TT2.toc('training time')
            history = history.history
            F_save_history_to_file(history, Keras_models_path, log_save_const)
            # copyfile(Keras_models_path + '/weights_best' + str(log_save_const) + '.hdf5',
            #          Save_pictures_path + '/weights_best' + str(log_save_const) + '.hdf5')
            # K.tensorflow_backend.clear_session()
            TT3 = TicToc();
            TT3.tic()
            #######################
            # x_all = F_extract_patches(x, pars_h['patch_sz_x'], pars_h['patch_sz_z'], pars_h['strides_x'],pars_h['strides_z'])
            # Npx = x_all.shape[1];Npz = x_all.shape[2];
            # x_all = F_merge_123_dim(x_all)
            # t_predicted_patches = model.predict(x_all, batch_size=pars_h['batch_size'])
            # t_predicted = F_extract_patches_inverse3(t_predicted_patches, pars_h['patch_sz_x'],
            #     pars_h['patch_sz_z'], Nx, Nz, Nm, Npx, Npz,
            #     pars_h['strides_x'], pars_h['strides_z'], Nm_to_invert=Nm_to_invert)
            #######################
            pred_generator = DataGenerator(x, t, pars_h, shuffle=False, to_fit=False, to_predict=True)
            steps_per_epoch_pred = pred_generator.__len__();
            # print(steps_per_epoch_pred)
            pred_gen_data = pred_generator.return_self()
            steps_per_epoch_pred = int(np.floor(pred_gen_data.Ns / 1))

            # model = unet_1(input_shape, pars_h['reg_value'])
            # opt = Adam(lr=pars_h['learning_rate'])
            # model.compile(loss=LOSS, optimizer=opt, metrics=[coeff_determination])
            # model.load_weights(Keras_models_path + '/weights_best' + str(log_save_const) + '.hdf5')

            tmp = model.predict_generator(pred_generator,steps=steps_per_epoch_pred,workers=N_workers,
                max_queue_size=10,use_multiprocessing=True,verbose=1)
            # K.tensorflow_backend.clear_session()
            TT3.toc('predict time')
            #   Assemble patches back to pictures. Approach2
            t_predicted = pred_generator.return_reconsctructed_data(tmp)
            # t_predicted_patches = pred_generator.return_reconsctructed_data2(tmp)
            # print('t_predicted_patches shape', t_predicted_patches.shape)
            # t_predicted = F_extract_patches_inverse3(t_predicted_patches, pars_h['patch_sz_x'],
            #     pars_h['patch_sz_z'], Nx, Nz, Nm, pred_gen_data.Npx, pred_gen_data.Npz,
            #     pars_h['strides_x'], pars_h['strides_z'], Nm_to_invert=Nm_to_invert)
        else:
            Model_to_load_const = F_init_CNN_model()
            pred_generator = DataGenerator(x, t, pars_h, shuffle=False, to_fit=False, to_predict=True)
            steps_per_epoch_pred = pred_generator.__len__()
            pred_gen_data = pred_generator.return_self()
            steps_per_epoch_pred = int(np.floor(pred_gen_data.Ns / 1))
            model.load_weights(Keras_models_path + '/weights_best' + str(Model_to_load_const) + '.hdf5')
            tmp = model.predict_generator(pred_generator, steps=steps_per_epoch_pred, workers=N_workers,
                max_queue_size=10, use_multiprocessing=True, verbose=1)
            # K.tensorflow_backend.clear_session()
            #   Assemble patches back to pictures. Approach2
            t_predicted = pred_generator.return_reconsctructed_data(tmp)
            history = F_load_history_from_file(Keras_models_path, Model_to_load_const)
    #   training with model.fit
    if pars_h['training_type'] == 2:
        if flag_train_model == True:
            TT2=TicToc();TT2.tic()
            print('start of training')
            print(pars_h)
            # history = model.fit(train_dataset3,
            #     validation_data=val_dataset,
            print('x_trainx_trainx_trainx_trainx_trainx_trainx_trainx_trainx_train')
            print(x_train.shape)
            input_shape=(pars_h['patch_sz_x'], pars_h['patch_sz_z'], x.shape[-1])
            # x_train = F_extract_patches(x_train, pars_h['patch_sz_x'], pars_h['patch_sz_z'], pars_h['strides_x'],
            #     pars_h['strides_z'])
            # x_train = F_merge_123_dim(x_train)
            # print('x_train shape=', x_train.shape)
            # t_train = F_extract_patches(t_train, pars_h['patch_sz_x'], pars_h['patch_sz_z'], pars_h['strides_x'],
            #     pars_h['strides_z'])
            # t_train = F_merge_123_dim(t_train)
            # x_valid = F_extract_patches(x_valid, pars_h['patch_sz_x'], pars_h['patch_sz_z'], pars_h['strides_x'],
            #     pars_h['strides_z'])
            # x_valid = F_merge_123_dim(x_valid)
            # t_valid = F_extract_patches(t_valid, pars_h['patch_sz_x'], pars_h['patch_sz_z'], pars_h['strides_x'],
            #     pars_h['strides_z'])
            # t_valid = F_merge_123_dim(t_valid)

            MODEL=unet_orig_implementation(input_shape)
            # MODEL = unet_encoder(input_shape)
            model =MODEL
            model.summary()
            opt = Adam(lr=pars_h['learning_rate'])
            model.compile(loss=LOSS, optimizer=opt, metrics=[coeff_determination])
            # model.compile(loss=[coeff_determination_neg], optimizer=opt, metrics=[coeff_determination])
            # x_train=np.repeat(x_train,100,axis=0)
            # t_train = np.repeat(t_train, 100, axis=0)
            history = model.fit(x_train,t_train,
                validation_data=(x_valid,t_valid),
                # validation_split=0.3,
                epochs=pars_h['epochs'],
                batch_size=pars_h['batch_size'],
                verbose=2, shuffle=True, callbacks=Callbacks)
            del x_train,x_valid,t_train,t_valid
            TT2.toc('Train time');print(pars_h)
            history = history.history
            F_save_history_to_file(history, Keras_models_path, log_save_const)
            # F_save_model_weights_to_file(model, Save_pictures_path, log_save_const)
        #    PREDICTION!!!!!!!!!     PREDICTION!!!!!!!!!
        # stride_x_prediction=32;         stride_z_prediction = 32
        stride_x_prediction = pars_h['strides_x'];       stride_z_prediction = pars_h['strides_z']
        # Nm_to_invert=1
        print('x_allx_allx_allx_allx_allx_allx_allx_allx_allx_allx_allx_allx_allx_all')
        # x_all = F_extract_patches(x, pars_h['patch_sz_x'], pars_h['patch_sz_z'],stride_x_prediction,stride_z_prediction,Nm_to_invert=Nm_to_invert)
        # x_all = F_merge_123_dim(x_all)
        x_all=x
        Npx = x_all.shape[1];
        Npz = x_all.shape[2];

        input_shape = (pars_h['patch_sz_x'], pars_h['patch_sz_z'], x.shape[-1])
        if flag_train_model==False:
            Model_to_load_const = F_init_CNN_model()
            model = MODEL
            fname=Keras_models_path + '/weights_best' + str(Model_to_load_const)+'.hdf5';print(fname)
            model.load_weights(fname)
            history = F_load_history_from_file(Keras_models_path, Model_to_load_const)
        else:
            model = MODEL
            fname = Keras_models_path + '/weights_best' + str(log_save_const) + '.hdf5';
            print(fname)
            model.load_weights(fname)
        TT3=TicToc();TT3.tic()
        # %%
        t_predicted_patches=model.predict(x_all,batch_size=pars_h['batch_size'])
        del x_all

        TT3.toc('predict time')
        TT4 = TicToc();TT4.tic()
        t_predicted =t_predicted_patches
        # t_predicted = F_extract_patches_inverse3(t_predicted_patches,pars_h['patch_sz_x'],
        #     pars_h['patch_sz_z'], Nx, Nz, Nm, Npx, Npz,
        #     stride_x_prediction,stride_z_prediction,Nm_to_invert=Nm_to_invert)
        # aa=np.equal(t_predicted,t_predicted_patches)
        TT4.toc('Post processing time')
    #   pytorch attempt
    if pars_h['training_type'] == 5:
        print('x_allx_allx_allx_allx_allx_allx_allx_allx_allx_allx_allx_allx_allx_all')
        x_all = F_extract_patches(x, pars_h['patch_sz_x'], pars_h['patch_sz_z'], pars_h['strides_x'],pars_h['strides_z'])
        Npx = x_all.shape[1];Npz = x_all.shape[2];
        x_all = F_merge_123_dim(x_all)
        # %%
        print('x_trainx_trainx_trainx_trainx_trainx_trainx_trainx_trainx_train')
        x_train=F_extract_patches(x_train,pars_h['patch_sz_x'],pars_h['patch_sz_z'],pars_h['strides_x'],pars_h['strides_z'])
        x_train=F_merge_123_dim(x_train)
        t_train = F_extract_patches(t_train, pars_h['patch_sz_x'], pars_h['patch_sz_z'], pars_h['strides_x'],pars_h['strides_z'])
        t_train = F_merge_123_dim(t_train)
        x_valid = F_extract_patches(x_valid, pars_h['patch_sz_x'], pars_h['patch_sz_z'], pars_h['strides_x'],pars_h['strides_z'])
        x_valid = F_merge_123_dim(x_valid)
        t_valid = F_extract_patches(t_valid, pars_h['patch_sz_x'], pars_h['patch_sz_z'], pars_h['strides_x'],pars_h['strides_z'])
        t_valid = F_merge_123_dim(t_valid)

        ss = np.intersect1d(factors(x_train.shape[0]), factors(x_all.shape[0]))
        ss2 = np.intersect1d(ss, factors(x_valid.shape[0]))
        batch_size_val = find_nearest(ss, pars_h['batch_size'])
        pars_h['batch_size'] = batch_size_val

        if flag_train_model == True:
            TT2=TicToc();TT2.tic()
            print('start of training')
            print(pars_h)

            model = unet_1_mod(input_shape, pars_h['reg_value'])
            opt = Adam(lr=pars_h['learning_rate'])
            model.compile(loss=LOSS, optimizer=opt, metrics=[coeff_determination])
            model.summary()
            history = model.fit(x_train, t_train,
                validation_data=(x_valid, t_valid),
                epochs=pars_h['epochs'],
                batch_size=pars_h['batch_size'],
                verbose=2, shuffle=True, callbacks=Callbacks)
            TT2.toc('Train time');print(pars_h)
            history = history.history

        if flag_train_model==False:
            Model_to_load_const = F_init_CNN_model()
            # model = F_load_model_from_file(Keras_models_path, Model_to_load_const)
            model=unet_1_mod((pars_h['patch_sz_x'], pars_h['patch_sz_z'],x.shape[-1]),pars_h['reg_value'])
            fname=Keras_models_path + '/weights_best' + str(Model_to_load_const) + '.hdf5';print(fname)
            model.load_weights(fname)
            history=F_load_history_from_file(Keras_models_path,Model_to_load_const)

        TT3=TicToc();TT3.tic()
        t_predicted_patches = model.predict(x_all,batch_size=pars_h['batch_size'])

        TT3.toc('predict time')
        TT4 = TicToc();TT4.tic()
        t_predicted = F_extract_patches_inverse3(t_predicted_patches, pars_h['patch_sz_x'],
            pars_h['patch_sz_z'], Nx, Nz, Nm, Npx, Npz,
            pars_h['strides_x'], pars_h['strides_z'], Nm_to_invert=Nm_to_invert)
        TT4.toc('Post processing time')
    #   pytorch attempt
    if pars_h['training_type'] == 6:
        if flag_train_model==True:
            TT2=TicToc();TT2.tic()
            training_generator = DataGenerator(x_train,t_train,pars_h,shuffle=True,to_fit=True)
            validation_generator = DataGenerator(x_valid,t_valid,pars_h,shuffle=True,to_fit=True)
            steps_per_epoch_train=training_generator.__len__();     print(steps_per_epoch_train)
            steps_per_epoch_valid=validation_generator.__len__();     print(steps_per_epoch_valid)
            history = model.fit_generator((training_generator),
                validation_data=(validation_generator),workers=N_workers,
                epochs=pars_h['epochs'],validation_steps=steps_per_epoch_valid,
                verbose=2, shuffle=True, callbacks=Callbacks,
                use_multiprocessing=False, steps_per_epoch=steps_per_epoch_train)
            TT2.toc('training time')
            history = history.history
            F_save_history_to_file(history, Keras_models_path, log_save_const)
            # copyfile(Keras_models_path + '/weights_best' + str(log_save_const) + '.hdf5',
            #          Save_pictures_path + '/weights_best' + str(log_save_const) + '.hdf5')
            # K.tensorflow_backend.clear_session()
            TT3 = TicToc();
            TT3.tic()
            pred_generator = DataGenerator(x, t, pars_h, shuffle=False, to_fit=False, to_predict=True)
            steps_per_epoch_pred = pred_generator.__len__();
            # print(steps_per_epoch_pred)
            pred_gen_data = pred_generator.return_self()
            steps_per_epoch_pred = int(np.floor(pred_gen_data.Ns / 1))
            tmp = model.predict_generator(pred_generator,steps=steps_per_epoch_pred,workers=N_workers,
                max_queue_size=10,use_multiprocessing=True,verbose=1)
            # K.tensorflow_backend.clear_session()
            TT3.toc('predict time')
            #   Assemble patches back to pictures. Approach2
            t_predicted = pred_generator.return_reconsctructed_data(tmp)
            # t_predicted_patches = pred_generator.return_reconsctructed_data2(tmp)
            # print('t_predicted_patches shape', t_predicted_patches.shape)
            # t_predicted = F_extract_patches_inverse3(t_predicted_patches, pars_h['patch_sz_x'],
            #     pars_h['patch_sz_z'], Nx, Nz, Nm, pred_gen_data.Npx, pred_gen_data.Npz,
            #     pars_h['strides_x'], pars_h['strides_z'], Nm_to_invert=Nm_to_invert)
        else:
            Model_to_load_const = F_init_CNN_model()
            pred_generator = DataGenerator(x, t, pars_h, shuffle=False, to_fit=False, to_predict=True)
            steps_per_epoch_pred = pred_generator.__len__()
            pred_gen_data = pred_generator.return_self()
            steps_per_epoch_pred = int(np.floor(pred_gen_data.Ns / 1))
            model.load_weights(Keras_models_path + '/weights_best' + str(Model_to_load_const) + '.hdf5')
            tmp = model.predict_generator(pred_generator, steps=steps_per_epoch_pred, workers=N_workers,
                max_queue_size=10, use_multiprocessing=True, verbose=1)
            # K.tensorflow_backend.clear_session()
            #   Assemble patches back to pictures. Approach2
            t_predicted = pred_generator.return_reconsctructed_data(tmp)
            history = F_load_history_from_file(Keras_models_path, Model_to_load_const)
    print('Training TIME', time.time() - start)
    print("Optimization Finished!")
    return model, history,t_predicted
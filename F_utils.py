from F_models import *
from F_modules import *
from F_plotting import *
def F_r2(mat, mat_true):
    # r2 = 1 - (np.std(mat_true.flatten() - mat.flatten()) / np.std(mat_true.flatten())) ** 2
    v1 = mat.flatten()
    v2 = mat_true.flatten()
    r2_2 = r2_score(v1, v2)
    return r2_2
def RepresentsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False
class Tee(object):
    # Write terminal output to log
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()
def numstr(x):
    string = str('{0:.2f}'.format(x))
    return string

#  Initialization
def F_init_paths():
    Save_pictures_path = './Pictures'
    local_dataset_path = './datasets/'
    Keras_models_path = './keras_models'
    F_create_folder(Save_pictures_path)
    F_create_folder(local_dataset_path)
    F_create_folder(Keras_models_path)
    return Save_pictures_path,local_dataset_path,Keras_models_path

def F_init_dataset():
    dataset_to_load_name = 'dataset_7756'
    return dataset_to_load_name

def F_create_folder(folder):
    os.makedirs(folder,exist_ok=True)
    return None

def F_calculate_log_number(path, Word, type='.png'):
    Const = len(fnmatch.filter(os.listdir(path), Word + '*'))
    Name = path + '/' + Word + str(Const) + type
    while os.path.exists(Name):
        Const = Const + 1
        Name = path + '/' + Word + str(Const) + type
    return Const

def load_file(filename, variable_name):
    f = h5py.File(filename, 'r')
    dat = np.array(f.get(variable_name))
    return dat

def F_calculate_misfits_across_files(list_all,list_test,list_train,list_valid,
    dataset_predicted_train,dataset_predicted_valid,dataset_predicted_test,
    Train_on1_model_and_test_on_other, Train_models, Test_models,
    Valid_models, print_flag=0,test_status=0):
    # %%   Calculate model misfits (R2 score) on test, validation, training datasets separately
    misfit_stats = np.zeros((4, 2))
    # %%     Allocate    data
    if Train_on1_model_and_test_on_other == 1:
        dataset_predicted_all=np.concatenate((dataset_predicted_train, \
             dataset_predicted_valid,dataset_predicted_test),axis=0)
        Nl = dataset_predicted_all.shape[0]
        print('Train')
        tmp = np.zeros(len(Train_models))
        for i_x in range(len(Train_models)):
            NAME=list_train[i_x]
            true=load_true_data(NAME,test_status)
            A=true[0,:,:,0]
            B=dataset_predicted_train[i_x, :, :,0]
            tmp[i_x] = F_r2(B,A)
        
        train_misfits=tmp
        misfit_stats[0, 0] = tmp.min()
        misfit_stats[0, 1] = tmp.max()

        idx_train_best=np.where(tmp==tmp.max())
        idx_train_worst=np.where(tmp==tmp.min())
        idx_train_best=idx_train_best[0]
        idx_train_worst=idx_train_worst[0]
        train_best_val=tmp.max()
        train_worst_val=tmp.min()
        # if idx_train_best==[]:
        #     idx_train_best=[0]
        # if idx_train_worst==[]:
        #     idx_train_worst=[0]
        train_best_file= list_train[int(idx_train_best [0])]
        train_worst_file=list_train[int(idx_train_worst[0])]

        print('Test')
        tmp = np.zeros(len(Test_models))
        for i_x in range(tmp.shape[0]):
            NAME=list_test[i_x]
            true=load_true_data(NAME,test_status)
            A=true[0,:,:,0]
            B=dataset_predicted_test[i_x, :, :,0]
            tmp[i_x] = F_r2(B,A)
            # plt.figure()
            # plt.imshow(np.concatenate((A,B),axis=1).T)
            # plt.title('R2='+str(tmp[i_x]) )
            # plt.show()
            # print(os.getcwd())
            # plt.savefig(str(i_x)+'.png')
            # plt.close()
        # exit()
            
        misfit_stats[1, 0] = tmp.min()
        misfit_stats[1, 1] = tmp.max()
        test_misfits=tmp

        tmp = np.zeros(Nl)
        for i_x in range(Nl):
            NAME=list_all[i_x]
            true=load_true_data(NAME,test_status)
            A=true[0,:,:,0]
            B=dataset_predicted_all[i_x, :, :,0]
            tmp[i_x] = F_r2(B,A)
        misfit_stats[2, 0] = tmp.min()
        misfit_stats[2, 1] = tmp.max()

        tmp = np.zeros(len(Valid_models))
        for i_x in range(tmp.shape[0]):
            NAME=list_valid[i_x]
            true=load_true_data(NAME,test_status)
            A=true[0,:,:,0]
            B=dataset_predicted_valid[i_x, :, :,0]
            tmp[i_x] = F_r2(B,A)
        misfit_stats[3, 0] = tmp.min()
        misfit_stats[3, 1] = tmp.max()
        valid_misfits=tmp
    # %%    Print output
    if print_flag == 1:
        print('Min R2 score across all models/Max R2 score across all models:')
        print('In train data', numstr(misfit_stats[0, 0]) + '/' + numstr(misfit_stats[0, 1]))
        print('In validation data', numstr(misfit_stats[3, 0]) + '/' + numstr(misfit_stats[3, 1]))
        print('In test data', numstr(misfit_stats[1, 0]) + '/' + numstr(misfit_stats[1, 1]))

        print('idx_train_best ', idx_train_best)
        print('train_best_file ',train_best_file)
        print('train_best_val ',train_best_val)
        
        print('idx_train_worst ',idx_train_worst)
        print('train_worst_file ',train_worst_file)
        print('train_worst_val ',train_worst_val)

        # print('Average score for ALL models: Train R2', numstr(misfits[0]), ',Valid R2', numstr(misfits[3]), ',Test R2',
        #       numstr(misfits[1]),
        #       ',All R2', numstr(misfits[2]))
        # print('In (train+tested) data', numstr(misfit_stats[2, 0]) + '/' + numstr(misfit_stats[2, 1]))
        ss=str(train_misfits.tolist())
        # print('Train misfits='+str(train_misfits.tolist()) )
        # print('Validation misfits='+str(valid_misfits.tolist()))
        # print('Train misfits='+str(test_misfits.tolist()))
    return misfit_stats,(idx_train_best),(idx_train_worst),train_misfits,valid_misfits,test_misfits,  \
    train_best_val,train_worst_val,train_best_file,train_worst_file

def PLOT_ML_Result4(inp,R2val,history_flag=0, history=None,
                    Boundaries=0, save_file_path='', dx=1, dy=1, Plot_vertical_lines=0, Title='',
                    Title2='', Save_flag=0, Show_flag=0):
    Nm = len(inp);
    dim1 = inp[0].shape[0]
    dim2 = inp[0].shape[1]
    if dim1 >= dim2:
        Nx = dim1
        Nz = dim2
        for i in range(Nm):
            inp[i] = inp[i].swapaxes(0,1)
    else:
        Nx = dim2
        Nz = dim1
    #   flip z axis
    for i in range(Nm):
        inp[i] = np.flip(inp[i], axis=0)
    input = inp[0]
    output = inp[1]
    pred = inp[2]
    matrix = inp[3]
    true_model=inp[4]

    Title3 = Title
    x = np.arange(Nx) * dx / 1000
    y = np.arange(Nz) * dy / 1000
    extent = np.array([x.min(), x.max(), y.min(), y.max()])
    # matplotlib.rcParams.update({'font.size': 15})

    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10.4
    fig_size[1] = 8.0
    plt.rcParams["figure.figsize"] = fig_size
    plt.rcParams["font.size"] = 10
    fig, (ax1, ax2, ax3,ax5, ax6) = plt.subplots(nrows=5, ncols=1)
    #  allocate space for colorbar smartly
    divider1 = make_axes_locatable((ax1))
    divider2 = make_axes_locatable((ax2))
    divider3 = make_axes_locatable((ax3))
    divider5 = make_axes_locatable((ax5))
    divider6 = make_axes_locatable((ax6))

    cax1 = divider1.append_axes("right", size="2%", pad=0.05)
    cax2 = divider2.append_axes("right", size="2%", pad=0.05)
    cax3 = divider3.append_axes("right", size="2%", pad=0.05)
    cax5 = divider5.append_axes("right", size="2%", pad=0.05)
    cax6 = divider6.append_axes("right", size="2%", pad=0.05)
    plt.set_cmap('RdBu_r')
    MIN =np.min(output)
    MAX = np.max(output)
    orig_min=np.min(true_model)
    orig_max=np.max(true_model)

    im1 = ax1.imshow(input, extent=extent,aspect='auto')
    im2 = ax2.imshow(output, extent=extent,vmin=MIN,vmax=MAX, aspect='auto')
    im3 = ax3.imshow(pred,extent=extent,  vmin=MIN, vmax=MAX, aspect='auto')
    # im3 = ax3.imshow(pred,extent=extent,aspect='auto')

    tmp=matrix
    tmp_extent = np.array([0,tmp.shape[1]*dx/1000,0,tmp.shape[0]*dy / 1000])
    im5 = ax5.imshow(matrix, extent=tmp_extent,vmin=orig_min,vmax=orig_max,aspect='auto')

    tmp=true_model
    tmp_extent = np.array([0,tmp.shape[1]*dx/1000,0,tmp.shape[0]*dy / 1000])
    im6 = ax6.imshow(true_model, extent=tmp_extent,vmin=orig_min,vmax=orig_max,aspect='auto')

    x0 = 0.11;
    y0 = -0.25
    if Plot_vertical_lines == 1:
        Boundaries2 = Boundaries * dx / 1000
        for i in range(np.shape(Boundaries)[0]):
            ax1.axvline(Boundaries2[i], color='k', linestyle='--', linewidth=2.5)
            ax2.axvline(Boundaries2[i], color='k', linestyle='--', linewidth=2.5)
            ax3.axvline(Boundaries2[i], color='k', linestyle='--', linewidth=2.5)
    z_label_name = 'z, m'
    ax1.set_ylabel(z_label_name)
    ax1.set_xlabel('x (km)')
    ax2.set_ylabel(z_label_name)
    ax2.set_xlabel('x (km)')
    ax3.set_ylabel(z_label_name)
    ax3.set_xlabel('x (km)')
    ax5.set_ylabel(z_label_name)
    ax5.set_xlabel('x (km)')
    ax6.set_ylabel(z_label_name)
    ax6.set_xlabel('x (km)')

    ax1.invert_yaxis()
    ax2.invert_yaxis()
    ax3.invert_yaxis()
    ax5.invert_yaxis()
    ax6.invert_yaxis()

    ax1.xaxis.set_label_coords(x0, y0)
    ax2.xaxis.set_label_coords(x0, y0)
    ax3.xaxis.set_label_coords(x0, y0)
    ax5.xaxis.set_label_coords(x0, y0)
    ax6.xaxis.set_label_coords(x0, y0)

    ax1.set_title('Input'+Title2)
    # print(output.shape)
    # print(true_model.shape)
    ax2.set_title('Target')
    # ax2.set_title('True'+numstr(F_r2(output,true_model)))
    # ax2.set_title('True'+numstr(F_r2(true_model,output)))
    ax3.set_title(Title3)
    # ax5.set_title('difference (predicted,true)')
    ax5.set_title('Predicted initial model for fwi, R2(predicted initial,true)='+R2val)
    ax6.set_title('True model')

    cbar1 = plt.colorbar(im1, cax=cax1)
    cbar2 = plt.colorbar(im2, cax=cax2)
    cbar3 = plt.colorbar(im3, cax=cax3)
    cbar5 = plt.colorbar(im5, cax=cax5)
    cbar6 = plt.colorbar(im6, cax=cax6)
    x0 = -50;
    y0 = 1.16
    cbar1.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar2.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar3.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar5.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar6.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    plt.subplots_adjust(bottom=0.08, top=0.95, hspace=0.57, right=0.86, wspace=.1)
    if Save_flag == 1:
        plt.savefig(save_file_path,dpi=400)
        print('Saving ML_result to '+save_file_path)
    if Show_flag == 1:
        plt.show()
    else:
        plt.show(block=False)
    plt.close()
    return None

def cmd(command):
    """Run command and pipe what you would see in terminal into the output cell
    """
    process = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
    while True:
        output = process.stderr.readline().decode('utf-8')
        if output == '' and process.poll() is not None:
            # this prints the stdout in the end
            output2 = process.stdout.read().decode('utf-8')
            print(output2.strip())
            break
        if output:
            print(output.strip())
    rc = process.poll()
    return rc

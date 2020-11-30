from F_modules import *
#   Plotting
def numstr(x):
    string = str('{0:.2f}'.format(x))
    return string
def F_nrms(mat,mat_true):
    nrms = np.linalg.norm((mat-mat_true),ord=2)/np.linalg.norm(mat_true,ord=2)
    return nrms
def F_r2(mat,mat_true):
    r2=1- (np.std(mat_true.flatten()-mat.flatten()) / np.std(mat_true.flatten())  )
    v1=mat.flatten()
    v2=mat_true.flatten()
    r2_2=r2_score(v1,v2)
    return r2_2
def tight_figure(fig,**kwargs):
    canvas = fig.canvas._get_output_canvas("png")
    print_method = getattr(canvas, 'print_png')
    print_method(io.BytesIO(), dpi=fig.dpi,
                 facecolor=fig.get_facecolor(), dryrun=True)
    renderer = fig._cachedRenderer
    bbox_inches = fig.get_tightbbox(renderer)
    bbox_artists = fig.get_default_bbox_extra_artists()
    bbox_filtered = []
    for a in bbox_artists:
        bbox = a.get_window_extent(renderer)
        if a.get_clip_on():
            clip_box = a.get_clip_box()
            if clip_box is not None:
                bbox = Bbox.intersection(bbox, clip_box)
            clip_path = a.get_clip_path()
            if clip_path is not None and bbox is not None:
                clip_path = \
                    clip_path.get_fully_transformed_path()
                bbox = Bbox.intersection(
                    bbox, clip_path.get_extents())
        if bbox is not None and (
                bbox.width != 0 or bbox.height != 0):
            bbox_filtered.append(bbox)

    if bbox_filtered:
        _bbox = Bbox.union(bbox_filtered)
        trans = Affine2D().scale(1.0 / fig.dpi)
        bbox_extra = TransformedBbox(_bbox, trans)
        bbox_inches = Bbox.union([bbox_inches, bbox_extra])

    pad = kwargs.pop("pad_inches", None)
    if pad is None:
        pad = plt.rcParams['savefig.pad_inches']

    bbox_inches = bbox_inches.padded(pad)

    tight_bbox.adjust_bbox(fig, bbox_inches, canvas.fixed_dpi)

    w = bbox_inches.x1 - bbox_inches.x0
    h = bbox_inches.y1 - bbox_inches.y0
    fig.set_size_inches(w,h)
def Plot_image(Data, Title='Title', c_lim='',x='',x_label='',y='',y_label='',
               dx='',dy='',Save_flag=0,Save_pictures_path='./Pictures',
               Reverse_axis=1,Curve='',Show_flag=1,Aspect='equal'):
    # aspect - 'auto'
    if c_lim == '':  c_lim =[np.min(Data), np.max(Data)]
    if x == '':  x=(np.arange(np.shape(Data)[1]))
    if y == '':  y=(np.arange(np.shape(Data)[0]))
    if dx != '':  x=(np.arange(np.shape(Data)[1]))*dx
    if dy != '':  y=(np.arange(np.shape(Data)[0]))*dy
    extent = [x.min(), x.max(), y.min(), y.max()]
    
    #if Save_flag==1:
    #    plt.ion()
    
    fig=plt.figure()
    fig.dpi=330
    # fig_size = plt.rcParams["figure.figsize"]
    # fig_size[0] = 10.4
    # fig_size[1] = 8.0
    # plt.rcParams["figure.figsize"] = fig_size
    plt.set_cmap('RdBu_r')
    # plt.axis(extent, Aspect)
    # plt.axis(extent, 'auto')
    plt.title(Title)
    if Reverse_axis == 1:
        plt.imshow(np.flipud(Data), extent=extent, interpolation='nearest',aspect=Aspect)
        plt.gca().invert_yaxis()
    else:
        plt.imshow((Data), extent=extent, interpolation='nearest',aspect=Aspect)
    if Curve != '':
        # if len(np.shape(Curve)) == 2:
        #     Curve=Curve[0,:]
        plt.plot(x, Curve, color='white', linewidth=1.2, linestyle='--')
    
    ax = plt.gca()
    divider1 = make_axes_locatable((ax))
    cax1 = divider1.append_axes("right", size="2%", pad=0.05)
    cbar=plt.colorbar(cax=cax1)
    plt.clim(c_lim)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    # plt.axis('equal')
    # plt.axis('tight')
    tight_figure(fig)
    if Save_flag == 1:
        if not os.path.exists(Save_pictures_path):
            os.mkdir(Save_pictures_path)
        name=Save_pictures_path + '/' + Title + '.png'
        print(name)
        # plt.show()
        # plt.show(block=True)
        # plt.show(block=False)
        plt.savefig(name)
    if Show_flag==0:
        plt.show(block=False)
        # plt.show(block=True)
    else:
        if Show_flag == 2:
            a=1
        else:
            plt.show()
    plt.close()
    return None
def Plot_accuracy(history,Title='Title',Save_pictures_path='./Pictures',Save_flag=0):
    plt.figure()
    plt.plot(history['mean_absolute_error'])
    plt.plot(history['val_mean_absolute_error'])
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.axis('tight')
    plt.legend(['Training accuracy', 'Validation accuracy'], loc='lower right')
    if Save_flag == 1:
        name=Save_pictures_path + '/' + Title + '.png'
        print(name)
        plt.savefig(name)
    plt.show(block=False)
    plt.close()
    return None
def Plot_accuracy2(history,Title='Title',Save_pictures_path='./Pictures',Save_flag=0):
    plt.figure()
    plt.plot(history['coeff_determination'])
    plt.plot(history['val_coeff_determination'])
    plt.ylabel('R2')
    plt.xlabel('Epoch')
    plt.axis('tight')
    plt.ylim(-1,1)
    string=', R2 accuracy curve train/test='+numstr( history['coeff_determination'][len(history['coeff_determination'])-1] )+'/'+numstr(history['val_coeff_determination'][len(history['val_coeff_determination'])-1])
    plt.title(Title+string)
    plt.legend(['training R2','validation R2'], loc='lower right')
    if Save_flag == 1:
        name = Save_pictures_path + '/' + Title + '.png'
        print(name)
        plt.savefig(name)
    plt.show(block=False)
    plt.close()
    return None
def Plot_loss(history,Title='Title',Save_pictures_path='./Pictures',Save_flag=0):
    plt.figure()
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.yscale('log')
    plt.ylabel('Loss function')
    plt.xlabel('Epoch')
    plt.axis('tight')
    # len(history['coeff_determination'])
    # print(', R2 accuracy curve train/test='+numstr(history['coeff_determination'][-1])+'/'+numstr(history['val_coeff_determination'][-1]))
    string=', R2 accuracy curve train/test='+numstr( history['coeff_determination'][len(history['coeff_determination'])-1] )+'/'+numstr(history['val_coeff_determination'][len(history['val_coeff_determination'])-1])
    plt.title(Title)
    plt.legend(['Training', 'Validation'], loc='upper right')
    if Save_flag == 1:
        name=Save_pictures_path + '/' + Title + '.png'
        print(name)
        plt.savefig(name)
    plt.show(block=False)
    plt.close()
    return None
def Plot_loss_r2(history,Title='Title',Save_pictures_path='./Pictures',Save_flag=0):
    plt.figure()
    plt.plot(-np.array(history['loss']))
    plt.plot(-np.array(history['val_loss']))
    # plt.yscale('log')
    # ax.set_yscale('log')
    plt.ylabel('Loss function,R2')
    plt.xlabel('Epoch')
    plt.axis('tight')
    plt.legend(['Training', 'Validation'], loc='upper right')
    if Save_flag == 1:
        name=Save_pictures_path + '/' + Title + '.png'
        print(name)
        plt.savefig(name)
    plt.show(block=False)
    plt.close()
    return None
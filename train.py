import argparse
from  os import path
import os
from datetime import date, time,timedelta, datetime
def file_exists(path_to_file):
    return path.exists(path_to_file)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( '--training', type=bool, default = False, help= 'if model is used for training( True ) or testing( False )' )
    parser.add_argument( '--input_path', type=str, help='path to the input dataset for training / testing' ,required = True )
    parser.add_argument( '--model_path', type=str, help='path to the stored model for saving / loading',required=True )
    parser.add_argument( '--encoder_path', type = str,  help='path to the stored encoder-decoder model for loading ', required=True )
    parser.add_argument( '--model_dir', type = str, default = './stored_models/', help = 'dir in which model is to be stored' )
    parser.add_argument( '--epochs', type = int ,default = 10, help = 'number of epochs, applicable only for training ' )
    parser.add_argument( '--batch_size', type = int, default = 180, help='batch size '  )
    parser.add_argument( '--pred_dir', type= str, default ='', help = 'dir to store the output')
    args = parser.parse_args()
    print(args)   
    if not file_exists(args.input_path):
        print('[ERROR] Check if the input_path : "%s" exists'%args.input_path )
        exit(-1)
    if not file_exists(args.model_path ):
        print('[ERROR] Check if the model_path : "%s" exists'%args.model_path )
        exit(-2)
    if not file_exists( args.encoder_path ):
        print('[ERROR] Check if the encoder_path : "%s" exists'%args.encoder_path )
        exit(-3)
    if not path.isdir( args.model_dir ): 
        print('[ERROR] Check if the model directory : "%s" exists'%args.model_dir )
        exit(-4)
    if args.training :
        if not path.isdir( args.pred_dir ):
            print('[ERROR] Check the pred directory : "%s" exists'%args.pred_dir )
            exit(-4)
    #### LOAD THE MODLE HERE
    from models import *
    from prepare_dataset import *
    # from prepare_dataset import *
    # path_to_ds = '../val_dataset.csv'
    # df = read_and_clean( path_to_ds,'Time','Value' )
    # # plt.plot(df['Time'],df['Value'])
    # # plt.show()
    # date_ = date(2019,7,1)
    # df2 = df[ df['Time'].dt.date == date_+timedelta(days=1)  ]
    # # df2 = read_df('')
    # print(df2.head())
    # dp = DataParse()
    # (ymin, ymax) ,backup, X_encoder, X_mlp ,time_ = dp.prep_to_train_mlp(df= df, time_label = 'Time',val_label='Value',train=False,date_ = date_  )
    # # print('done')
    # vae = VAE()
    # vae.load_from(path_to_model='../model/vae_custom_loss_v2.hdf5')
    # mlp = MLP(vae =vae)
    # mlp.load_from(path_to_model='../model/mlp_custom_loss___.hdf5')
    # y_pred = mlp.predict(encoderIn=X_encoder,mlpIn = X_mlp)
    # y_pred = dp.get_original(backup,y_pred,ymin,ymax)
    # plt.plot(time_,y_pred,label ='pred')
    # plt.plot(df2['Time'],df2['Value'],label='true')
    # plt.legend()
    # plt.show()
    # print('sd')
    d_parser = DataParse()
    input_df =read_and_clean(path_to_csv = args.input_path,time_label = 'Time',val_label = 'Value',sep=';')
    # plt.plot(input_df['Time'], input_df['Value'])
    # plt.show()
    date_to_predict = date(2019,6,16)
    date_to_train = date_to_predict - timedelta(days =  1 )
    (ymin, ymax) ,backup, X_encoder, X_mlp ,time_ = d_parser.prep_to_train_mlp(df= input_df, time_label = 'Time',val_label='Value',train=True,date_ = date_to_train  )
    time_stamp = datetime.now().strftime("%d_%m_%Y__%H_%M")
    time_stamp = ''
    vae = VAE(path_to_save= args.encoder_path)
    vae.load_from(path_to_model=args.encoder_path)
    mlp = MLP(path_to_save= args.model_path, vae = vae)
    mlp.load_from( path_to_model = args.model_path )
    if args.training :
        X_encoder_in , Y_mlp = X_encoder
        # print(X_encoder_in.shape, Y_mlp.shape)
        vae_path = os.path.join( args.model_dir,  "vae.hdf5" )
        mlp_path = os.path.join( args.model_dir,  "mlp.hdf5" )
        # mlp.fit()
        # print(vae_path, mlp_path)
        # plt.show()
        # plt.figure()
        # for epoch in range(0,7):
            # plt.figure()            
        mlp.fit(encoderIn = X_encoder_in, mlpIn = X_mlp,target= Y_mlp, path_to_save = mlp_path,epochs = 20)
        (ymin_, ymax_) ,backup_, X_encoder_, X_mlp_ ,time_ = d_parser.prep_to_train_mlp(df= input_df, time_label = 'Time',val_label='Value',train=False,date_ = date_to_train  )
        y_hat = mlp.predict(encoderIn = X_encoder_, mlpIn = X_mlp_)
        yhat  = d_parser.get_original(backup_,y_hat, ymin_, ymax_ )
        print(yhat.shape)
        plt.plot( time_, yhat,label = 'Predicted ' )
        dt = input_df[input_df['Time'].dt.date == date_to_predict ]
        plt.plot( dt['Time'],dt['Value'],label= 'Original')
        plt.legend()
        plt.show()
        ystar = np.asarray(dt['Value'])
        yhat = yhat.reshape(len(ystar))
        mape = (np.absolute(ystar-yhat)/ystar).mean()
        # print(mape*100)
        print('Accuracy : %.2f %%'%((1-mape)*100))
        # Train the model

import argparse
from  os import path
import os

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
    if not path._isdir( args.model_dir ): 
        print('[ERROR] Check if the model directory : "%s" exists'%args.model_dir )
        exit(-4)
    if args.training :
        if not path._isdir( args.pred_dir ):
            print('[ERROR] Check the pred directory : "%s" exists'%args.pred_dir )
            exit(-4)
    #### LOAD THE MODLE HERE
    from models import *
    from prepare_dataset import DataParse
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
    vae = VAE(path_to_save= args.encoder_path)
    vae.load_from(path_to_model=args.encoder_path)
    mlp = MLP(path_to_save= args.model_path, vae = vae)
    mlp.load_from( path_to_model = args.model_path )
    input_df =read_and_clean(path_to_csv = args.input_path,time_label = 'Time',val_label = 'Value')
    if args.training :
        # Train the model

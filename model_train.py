import argparse
from  os import path
import os
from datetime import date, time,timedelta, datetime
def file_exists(path_to_file):
    return path.exists(path_to_file)
if __name__ == '__main__':
    '''
        Module ONLY for TRAINING MLP using pre-trained vae
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument( '--fresh', type=bool, default = False, help= ' True => Train the model from scratch !! ' )
    parser.add_argument( '--input_path', type=str, help='path to the input dataset for training / testing' ,required = True )
    parser.add_argument( '--mlp_path', type = str,  help='path to the stored mlp model for loading ' )
    parser.add_argument( '--freeze_encoder', type=bool, default= False , help ='True => the encoder layer will be frozen during training' )
    parser.add_argument( '--encoder_path', type = str,  help='path to the stored encoder-decoder model for loading ',required = True )
    parser.add_argument( '--store_at', type = str, default = './stored_models/', help = 'directory in which model is to be stored' )
    parser.add_argument( '--epochs', type = int ,default = 10, help = 'number of epochs, applicable only for training ' )
    parser.add_argument( '--batch_size', type = int, default = 180, help='batch size '  )
    args = parser.parse_args()
    # print(args)   
    if not file_exists(args.input_path):
        print('[ERROR] Check if the input_path : "%s" exists'%args.input_path )
        exit(-1)
   
    if  not file_exists( args.encoder_path ):
        print('[ERROR] Check if the encoder_path : "%s" exists'%args.encoder_path )
        exit(-3)
    if not args.fresh :
        if not file_exists( args.mlp_path ):
            print('[ERROR] Check if the mlp_path : "%s" exists'%args.mlp_path )
            exit(-3)
    if not path.isdir( args.store_at ): 
        print('[ERROR] Check if the store directory : "%s" exists'%args.store_at )
        exit(-4)
    #### LOAD THE MODLE HERE
    from prepare_dataset import *
    d_parser = DataParse()
    print('\n\nReading the data...\n\n')
    input_df = read_and_clean(path_to_csv = args.input_path,time_label = 'Time',val_label = 'Value',sep=';')
    print('Preparing dataset for training...\n')
    (ymin, ymax) ,backup, (X_encoder,Y_mlp), X_mlp = d_parser.prep_mlp_train( cdf = input_df )
    time_stamp = datetime.now().strftime("%d_%m_%Y__%H_%M")
    time_stamp = '' # COMMENT this line to include time stamp in the stored model name
    print('\n\nLoading the VAE from path:"%s"\n\n'%args.encoder_path)
    from models import VAE
    vae = VAE(path_to_save = '') 
    vae.load_from(path_to_model=args.encoder_path)
    if args.freeze_encoder:
        vae.freeze()
    from models import MLP
    mlp = MLP(vae = vae)
    if not args.fresh:
        print('\n\nLoading the MLP from path:"%s"\n\n'%args.mlp_path)
        mlp.load_from(args.mlp_path)
    else:
        mlp.compile()
    mlp_path = os.path.join( args.store_at,  "mlp%s.hdf5"%time_stamp )
    print('\n\nCommencing training procedure...\n\n')
    mlp.fit(encoderIn=X_encoder,mlpIn=X_mlp,target=Y_mlp,path_to_save=mlp_path,epochs=args.epochs,batch_size=args.batch_size )
    print('\n\n\n'+'=-='*50+'\nDone Training MLP\nModel Stored At: "%s"'%mlp_path+'\n'+'=-='*50+'\n\n\n')
    
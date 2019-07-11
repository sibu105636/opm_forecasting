import argparse
from  os import path
import os
from datetime import date, time,timedelta, datetime
def file_exists(path_to_file):
    return path.exists(path_to_file)
if __name__ == '__main__':
    '''
        Module ONLY for TRAINING encoder-decoder VAE
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument( '--fresh', type=bool, default = False, help= ' True => Train the model from scratch !! ' )
    parser.add_argument( '--input_path', type=str, help='path to the input dataset for training / testing' ,required = True )
    parser.add_argument( '--encoder_path', type = str,  help='path to the stored encoder-decoder model for loading ' )
    parser.add_argument( '--store_at', type = str, default = './stored_models/', help = 'dir in which model is to be stored' )
    parser.add_argument( '--epochs', type = int ,default = 10, help = 'number of epochs, applicable only for training ' )
    parser.add_argument( '--batch_size', type = int, default = 180, help='batch size '  )
    args = parser.parse_args()
    # print(args)   
    if not file_exists(args.input_path):
        print('[ERROR] Check if the input_path : "%s" exists'%args.input_path )
        exit(-1)
   
    if not args.fresh and not file_exists( args.encoder_path ):
        print('[ERROR] Check if the encoder_path : "%s" exists'%args.encoder_path )
        exit(-3)
    if not path.isdir( args.store_at ): 
        print('[ERROR] Check if the store directory : "%s" exists'%args.store_at )
        exit(-4)
    #### LOAD THE MODLE HERE
    from prepare_dataset import *
    d_parser = DataParse()
    print('Reading the data...\n')
    input_df = read_and_clean(path_to_csv = args.input_path,time_label = 'Time',val_label = 'Value',sep=';')
    print('Preparing dataset for training...\n')
    (ymin, ymax) ,backup, (X_encoder,X_decoder), Y_decoder = d_parser.prep_vae_train( cdf = input_df )
    time_stamp = datetime.now().strftime("%d_%m_%Y__%H_%M")
    time_stamp = '' # COMMENT this line to include time stamp in the stored model name
    from models import VAE
    vae = VAE(path_to_save = '') 
    if not args.fresh:
        vae.load_from(path_to_model=args.encoder_path)
    else:
        vae.compile()
    vae_path = os.path.join( args.store_at,  "vae%s.hdf5"%time_stamp )
    print('Commencing training procedure...\n')
    vae.fit(encoderIn=X_encoder,decoderIn=X_decoder,decoderOut=Y_decoder,path_to_save=vae_path,epochs=args.epochs,batch_size=args.batch_size )
    print('\n'+'=-='*50+'\nDone Training VAE\nModel Stored At: "%s"'%vae_path+'\n'+'=-='*50+'\n')
    
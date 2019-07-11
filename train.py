import argparse
from  os import path
import os
import random as rand
from datetime import date, time,timedelta, datetime
def file_exists(path_to_file):
    return path.exists(path_to_file)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( '--auto_dt', type=str,default='True', help='True -> pick the date automatically False -> Specifiy the date' )
    parser.add_argument( '--date_to_pred', type=str, default = None , help='date [YYYY-MM-DD] for which prediction is to be done '  )
    parser.add_argument( '--input_path', type=str, help='path to the input dataset for training / testing' ,required = True )
    parser.add_argument( '--model_path', type=str, help='path to the stored model for saving / loading',required=True )
    parser.add_argument( '--encoder_path', type = str,  help='path to the stored encoder-decoder model for loading ', required=True )
    parser.add_argument( '--model_dir', type = str, default = './stored_models/', help = 'dir in which model is to be stored' )
    parser.add_argument( '--epochs', type = int ,default = 10, help = 'number of epochs, applicable only for training ' )
    parser.add_argument( '--batch_size', type = int, default = 180, help='batch size '  )
    parser.add_argument( '--pred_dir', type= str, default ='', help = 'directory to store the output')
    parser.add_argument( '--to_UK_tz', type= bool, default = True, help = 'True => datetime in UK time zone else in INDIAN time zone')
    args = parser.parse_args()
    print(args)   
    args.auto_dt = args.auto_dt.strip().lower()=='true'
    pred_df_name = 'output.csv'
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
    if not path.isdir( args.pred_dir ):
            print('[ERROR] Check the pred directory : "%s" exists'%args.pred_dir )
            exit(-4)
    if not args.auto_dt and args.date_to_pred == None:
        print('[ERROR] Specify the date pls ' )
        exit(-5)
    #### LOAD THE MODLE HERE
    from models import *
    from prepare_dataset import *
    d_parser = DataParse()
    input_df =read_and_clean(path_to_csv = args.input_path,time_label = 'Time',val_label = 'Value',sep=';')
    # plt.plot(input_df['Time'], input_df['Value'])
    # plt.show()
    date_to_predict = date.today() if args.auto_dt else  datetime.strptime( args.date_to_pred , '%Y-%m-%d' ).date()
    print( date_to_predict )
    # print(  'Date to predict %d-%d-%d'%(date_to_predict.year(),date_to_predict.month(), date_to_predict.day()))
    # date_to_predict = date(2019,6,16)
    date_to_train = date_to_predict - timedelta(days =  1 )
    (ymin, ymax) ,backup, X_encoder, X_mlp ,time_ = d_parser.prep_to_train_mlp(df= input_df, time_label = 'Time',val_label='Value',train=True,date_ = date_to_train  )
    time_stamp = datetime.now().strftime("%d_%m_%Y__%H_%M")
    time_stamp = ''
    vae = VAE(path_to_save= args.encoder_path)
    vae.load_from(path_to_model=args.encoder_path)
    mlp = MLP(path_to_save= args.model_path, vae = vae)
    mlp.load_from( path_to_model = args.model_path )
    # if args.training :
    X_encoder_in , Y_mlp = X_encoder
    # print(X_encoder_in.shape, Y_mlp.shape)
    vae_path = os.path.join( args.model_dir,  "vae.hdf5" )
    mlp_path = os.path.join( args.model_dir,  "mlp.hdf5" )       
    mlp.fit(encoderIn = X_encoder_in, mlpIn = X_mlp,target= Y_mlp, path_to_save = mlp_path,epochs = args.epochs , batch_size = args.batch_size)
    (ymin_, ymax_) ,backup_, X_encoder_, X_mlp_ ,time_ = d_parser.prep_to_train_mlp(df= input_df, time_label = 'Time',val_label='Value',train=False,date_ = date_to_train  )
    y_hat = mlp.predict(encoderIn = X_encoder_, mlpIn = X_mlp_)
    yhat  = d_parser.get_original(backup_,y_hat, ymin_, ymax_ )

    len_yhat = len(yhat)

    from datetime import datetime
    start_ = datetime( date_to_predict.year, date_to_predict.month, date_to_predict.day, 0 , 0 )
    n_forecast = 1
    
    time_offset  = 0 if args.to_UK_tz else 330

    time__ = [ start_ + timedelta( minutes = i+time_offset ) for i in range(0, n_forecast*24*60) ]
    
    diff = len(time__) - len_yhat
    val = np.zeros(len(time__))
    for i in range(len(val)):
        if i >= diff:
            val[i] = yhat[i-diff]
        else:
            val[i] =   rand.uniform(2, 6) 
        val[i] = round(val[i],2)

    pred_df = pd.DataFrame({
        'Time':time__,
        'Value':val
    })
    pred_df.loc[:,'Series'] = 'predicted'
    pred_df_path = os.path.join(args.pred_dir , pred_df_name )
    if not args.to_UK_tz:
        pred_df.loc[:,'Time'] = pred_df.Time.astype(str).add('+05:30')
    pred_df.to_csv(pred_df_path,sep =';')
    print('\n'+'=-='*50+'\nDone \n Predicted Data Stored At: "%s"'%pred_df_path+'\n'+'=-='*50+'\n')
        # mape = (np.absolute(ystar-yhat)/ystar).mean()
        # # print(mape*100)
        # print('Accuracy : %.2f %%'%((1-mape)*100))
        # Train the model

## Import the necessary libraries
import keras
import tensorflow as tf
import keras.backend as K 
from keras.layers import Dropout, LSTM, Dense, TimeDistributed, Reshape, Concatenate, Input
from keras.models import load_model, Model, Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.vis_utils import model_to_dot
from keras.utils.vis_utils import plot_model
from IPython.display import SVG
from matplotlib import image as mpimg
from matplotlib import pyplot as plt
## Declare the encoder-decoder model
class VAE:
    def __init__(self,path_to_save='',embedding_dim = 128, input_feat = 2 ,look_back = 28, use_next = 7 ):
        self.path_to_save = path_to_save
        self.embed_dim = embedding_dim
        self.look_back = look_back
        self.use_next = use_next 
        self.features = input_feat
        self.vae_in = Input( shape = (self.look_back, self.features), name = 'encoderIn')
        self.vae_lstm = LSTM( self.embed_dim, return_state = True, return_sequences = False, recurrent_dropout = .3, name = 'encoder' )
        self.vae_lstm_out, self.vae_lstm_h, self.vae_lstm_c = self.vae_lstm( self.vae_in )
        self.decoder_state = [ self.vae_lstm_h, self.vae_lstm_c ]
        self.vae_decoder_in = Input( shape = (self.use_next , self.features), name = 'decoderIn')
        self.vae_decoder_lstm = LSTM( self.embed_dim, return_state = False, return_sequences = True, recurrent_dropout = .2, name = 'decoder' )
        self.vae_decoder_out  =  self.vae_decoder_lstm(self.vae_decoder_in, initial_state = self.decoder_state )
        self.vae_out = TimeDistributed( Dense( self.features ), name = 'output')(self.vae_decoder_out)
        self.vae  = Model( inputs = [ self.vae_in, self.vae_decoder_in ], outputs = self.vae_out )
        self.history = None
        self.callbacks = []
    def add_Callback(self,callback):
        self.callbacks.append(callback)
    def compile(self, loss = 'mse', optimizer = 'adam'):
        ### Don't use  custom loss will result in error during storing the model
        self.vae.compile(loss= loss,optimizer  = optimizer)
    def graph(self):
        plot_model(self.vae, to_file='vae.png', show_shapes=True, show_layer_names=True)
        img=mpimg.imread('vae.png')
        imgplot = plt.imshow(img)
        plt.show()
        # SVG(model_to_dot(self.vae,show_shapes = True).create(prog ='dot',format='SVG')) 
    def summary(self):
        print( self.vae.summary() )
    def load_from(self, path_to_model = '' ):
        if path_to_model == '':
            print( "Cann't load from the empty path ")
        else:
            print('loading from %s'%path_to_model)
            self.vae = load_model(path_to_model)
    def save_to( self, path_to_save ):
        if self.path_to_save == '' and path_to_save == None:
            print('Should specify a path to save model... ')
        else: 
            if path_to_save != None:
                self.path_to_save = path_to_save
            self.vae.save(self.path_to_save)
    def fit(self, encoderIn,decoderIn, decoderOut , path_to_save = None, batch_size = 180, epochs = 10 ):
        if self.path_to_save == '' and path_to_save == None:
            print('Should specify a path to save for training ... ')
        else: 
            if path_to_save != None:
                self.path_to_save = path_to_save
            print( 'VAE would be stored at %s'%self.path_to_save)
            checkpnt  =  self.callbacks+[ ModelCheckpoint( self.path_to_save, monitor='loss', verbose= 1,save_best_only= True, mode = 'min' )]
            self.history = self.vae.fit( {'encoderIn':encoderIn,'decoderIn':decoderIn},decoderOut,batch_size = batch_size , epochs = epochs, callbacks = checkpnt,shuffle = False)
            return self.history
    def predict(self, encoderIn,decoderIn , path_to_save = None ):
        return  self.vae.predict( {'encoderIn':encoderIn, 'decoderIn':decoderIn } )
    def freeze(self):
        for layer in self.vae.layers:
            layer.trainable = False
    def  get_encoder_in(self):
        # self.freeze()
        return self.vae.get_layer('encoderIn').input
    def get_encoder_out(self):
        out,_,_=  self.vae.get_layer('encoder').output
        return out

class MLP:
    def __init__( self, path_to_save = '', vae = None,look_back = 28, forecast = 1,extra_feat = 8 ):
        self.path_to_save = path_to_save
        self.vae  = vae
        self.is_compiled = False
        if vae == None:
            print('Could not proceed without VAE ')
        else:
            self.forecast = forecast
            self.extra_feat = extra_feat
            self.vae_to_mlp = Reshape( (1,self.vae.embed_dim))( self.vae.get_encoder_out() )
            self.mlp_in = Input( shape = (1,self.extra_feat), name='mlpIn')
            self.input = keras.layers.concatenate( name = 'Input',inputs = [ self.vae_to_mlp ,  self.mlp_in ] )
            self.mlp_h0 = Dense(128,name = 'H0'  )( self.input  )
            self.mlp_d0 = Dropout(0.2,name = 'D0')( self.mlp_h0 )
            self.mlp_h1 = Dense(64, name = 'H1'  )( self.mlp_d0 )
            self.mlp_d1 = Dropout(0.2,name = 'D1')( self.mlp_h1 )
            self.mlp_h2 = Dense(16, name = 'H2'  )( self.mlp_d1 )
            self.mlp_d2 = Dropout(0.2,name = 'D2')( self.mlp_h2 )
            self.mlp_out = Dense(1, name = 'yhat')( self.mlp_d2 )
            self.mlp = Model( inputs = [self.vae.get_encoder_in(), self.mlp_in], outputs = self.mlp_out)
            self.history = None
    def compile( self, loss = 'mse', optimizer = 'adam'):
        self.mlp.compile(loss = loss, optimizer = optimizer)
        self.is_compiled = True
        
    def summary(self):
        print( self.mlp.summary( ) )
    
    def graph(self):
        plot_model(self.mlp, to_file='mlp.png', show_shapes=True, show_layer_names=True)
        img=mpimg.imread('vae.png')
        imgplot = plt.imshow(img)
        plt.show()
    def load_from(self, path_to_model = '' ):
        if path_to_model == '':
            print( "Cann't load from the empty path ")
        else:
            print('loading from %s'%path_to_model)
            self.mlp = load_model(path_to_model)
            self.is_compiled = True
    def save_to( self, path_to_save ):
        if self.path_to_save == '' and path_to_save == None:
            print('Should specify a path to save model... ')
        else: 
            if path_to_save != None:
                self.path_to_save = path_to_save
            self.mlp.save(self.path_to_save)
    def fit(self, encoderIn,mlpIn, target , path_to_save = None, batch_size = 180, epochs = 1 ):
        if self.is_compiled:
            if self.path_to_save == '' and path_to_save == None:
                print('Should specify a path to save for training ... ')
            else: 
                if path_to_save != None:
                    self.path_to_save = path_to_save
                print( 'MLP would be stored at %s'%self.path_to_save)
                checkpnt  =  [ ModelCheckpoint( self.path_to_save, monitor='loss', verbose= 1,save_best_only= True, mode = 'min' )]
                self.history = self.mlp.fit( {'encoderIn':encoderIn,'mlpIn':mlpIn},target,batch_size = batch_size , epochs = epochs, callbacks = checkpnt,shuffle = False)
                return self.history
        else: 
            print('Compile the model first')
    def predict(self, encoderIn,mlpIn , path_to_save = None ):
        
        return  self.mlp.predict( {'encoderIn':encoderIn, 'mlpIn':mlpIn } )

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
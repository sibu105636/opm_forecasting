import pandas as pd
import numpy as np
from datetime import date, time, timedelta
import pickle
import holidays
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
# register_matplotlib_converters()
uk_holidays = holidays.CountryHoliday('UK')
class DataParse:
    def __init__(self):
        self.normalizer = 'scale norm'
        self.look_back = 28
        self.forecast_next = 1
        self.encoder_feat = 2
        self.decoder_feat = 1
        self.use_next  =  7
        self.offset_val = 0
        self.uk_holidays = holidays.CountryHoliday('UK')
        self.extra_feat = 8
        self.ymax = 1
        self.ymin = 0
    
    def get_min_max(self,dataframe,val_label):
        self.ymax = dataframe[val_label].max()
        self.ymin = dataframe[val_label].min()
        print( '\n'+'****'*30+'\n\n'+'Input Range <- [ %f, %f ]'%(self.ymin, self.ymax))
        print (  '\n\n' + '****'*30+'\n' )

    def symm_diff(self,curr,other):
        a  = min(curr,other)
        b  = max(curr,other )
        b  = b - a
        if b <= 3  :
            return b
        else:
            return 7 - b

    def extra_function(self, date_):
        '''
        Returns the feature extracted only from the date-time
        Parameters 
        ---------------------------
        date_  : date for which the feature is to be extracted\n
        Returns 
        ---------------------------
        X_mlp : features extracted, size ( len_, 1, 8 )
        '''
        x = date_
        x = .8 if x in self.uk_holidays else  .2 if timedelta(days = 1) + x in self.uk_holidays or  x - timedelta(days = 1)  in self.uk_holidays  else  -10000
        beta = 0.4
        h_rbf =  np.exp(-1*((x - .8)**2)/(2*beta))  
        x = date_.weekday()
        dow = np.zeros((8))
        alpha = 0.4
        dow[0] =  h_rbf
        xp = ( x+2 )  % 7
        w = 1 # (.1* (xp+5) )**2 if xp < 5 else .5 if xp == 5 else  0.35
        for i in range(0,7):
            dow[i+1] = w * np.exp(-1*((self.symm_diff(i,x))**2)/(2*alpha))
        return dow
    def get_exogenous(self, date_ , len_ ):
        '''
        Returns the feature extracted only from the date-time
        Parameters 
        ---------------------------
        date_  : date for which the feature is to be extracted\n
        len_   : length of the output to generated

        Returns 
        ---------------------------
        X_mlp : features extracted, size ( len_, 1, 8 )
        '''
        if len_ > 1080 :
            print('\n%s\nWARNING: designed to predict for one day only\nGot : %d \n%s\n'%('!!'*20,len_,'!!'*20))
        x = date_
        x = .8 if x in self.uk_holidays else  .2 if timedelta(days = 1) + x in self.uk_holidays or  x - timedelta(days = 1)  in self.uk_holidays  else  -10000
        beta = 0.4
        h_rbf =  np.exp(-1*((x - .8)**2)/(2*beta))  
        x = date_.weekday()
        dow = np.zeros((8))
        alpha = 0.4
        dow[0] =  h_rbf
        xp = ( x+2 )  % 7
        w = 1 # (.1* (xp+5) )**2 if xp < 5 else .5 if xp == 5 else  0.35
        for i in range(0,7):
            dow[i+1] = w * np.exp(-1*((self.symm_diff(i,x))**2)/(2*alpha))
        plt.show()
        X_mlp = np.zeros( ( len_ , 1, self.extra_feat ) )
        X_mlp[:,0,:] = dow[:]
        return X_mlp
    def prep_to_train_mlp(self,df,time_label , date_ , val_label , train ) :
        '''
        Returns the data in the format required by the VAE-MLP model for training and prediction\n
        ``df``          : dataframe with the time series data\n
        ``val_label``   : label for the opm in dataframe\n
        ``time_label``  : label for the time in dataframe\n
        ``train``       : False => data prepared for the `inference` procedure\n
        ``date_``       : date for which one needs to train\n
        @returns : (self.ymin, self.ymax) ,backup, (X_encoder, Y_mlp) if train else  X_encoder, X_mlp ,time_ )
        '''
        start_date = date_ - timedelta(days = (self.look_back ) )
        mask = df[time_label].dt.date >= start_date 
        mask = mask & ( df[time_label].dt.date <= date_ )
       
        if train :
            offset = 0
        else:
            offset = 1

        cdf = df.copy() [ mask ]
        # cdf = df.copy()
        # print(cdf.tail())
        cdf = cdf.reset_index()
        cdf.loc[:, time_label ] = pd.to_datetime( cdf[ time_label ] )
        # cdf = cdf [ mask ]
        cdf.loc[:,'dtx'] = cdf[ time_label ]
        cdf = cdf.set_index( 'dtx' )
        cdf.loc[:,'inh'] = cdf[ time_label ].dt.date.apply( lambda x : 1 if x+timedelta( days = 1 ) in self.uk_holidays else 0 )
        self.ymax = cdf[ val_label ].max()
        self.ymin = cdf[ val_label ].min()

        print( '\n\n'+'****'*20+('\nInput Range <- [ %.3f , %.3f ] '%(self.ymin, self.ymax)) + '\n'+'****'*20 +'\n\n' )
        
        cdf.loc[:,'ynorm'] = ( cdf[ val_label ] - self.ymin ) / ( self.ymax - self.ymin )  # Sets Input Range <- [ 0 , 1 ]
        # print(cdf.ynorm.describe())
        val_cols , nh_cols = self.data_shifter( cdf = cdf, val_label = 'ynorm', nh_label = 'inh' ,  train = train  )
        # print(cdf.head())
        # print(cdf.tail())
        mask2 = cdf[time_label].dt.date == date_
        cdf = cdf[ mask2 ]
        # if train : 
        # print(cdf.tail())
        cdf = cdf.dropna()
        len_train =  cdf[ 'ynorm' ].count()
        X_encoder = np.zeros( (len_train, self.look_back , self.encoder_feat) )
        for i in range(-self.look_back+offset,offset):
            col = '%dy'%i
            idx = 28+i-offset
            X_encoder[:,idx,0] = cdf[col]
            col = '%dinh'%i
            X_encoder[:,idx,1] = cdf[col]
        Y_mlp = np.zeros( (len_train,1,1))
        # if train
        Y_mlp [:,0,0] = cdf['0y']
        backup = X_encoder[:,0,0].copy()
        for i in range(self.look_back):
            X_encoder[:,i,0] -= backup
        Y_mlp[:,0,0] -= backup[:]
        # print(cdf['0y'].describe())
        time_ = list(cdf[ time_label ]+ timedelta( days = offset) )
        # print( 'done' )
        return ( (self.ymin, self.ymax) ,backup, (X_encoder, Y_mlp) if train else  X_encoder, self.get_exogenous(date_, len_train),time_)
    
    def data_shifter(self,cdf,val_label,nh_label,train):
        val_cols = []
        nh_cols = []
        offset = 0 if train else 1
        for i in range(-self.look_back+ offset, 1):
            sym =  ''
            col = '%s%dy'%(sym,i)
            val_cols.append(col)
            cdf.loc[:,col] = cdf[val_label].shift(periods = 1,freq = timedelta(days = -i)) 
            # cdf[col] = cdf[val_label].shift(periods = 1,freq = timedelta(days = -i))
            col = '%s%dinh'%(sym,i)
            nh_cols.append(col )
            cdf.loc[:,col] = cdf[nh_label].shift(periods = 1,freq = timedelta(days = -i)) 
            # cdf[col] = cdf[nh_label].shift(periods = 1,freq = timedelta(days = -i))
        # print('\n'+'--'*50+'\n',cdf.columns,'\n'+'--'*50)
        return ( val_cols, nh_cols )   

    def prep_mlp_train(self,cdf,val_label='Value',time_label='Time',series ='current'):
        '''
        This function prepares the data for the training from the given dataset
        '''
        cdf = cdf.copy().reset_index()
        cdf.loc[:,time_label] = pd.to_datetime( cdf[time_label] )
        cdf.loc[:,'inh'] = cdf[time_label].dt.date.apply( lambda x : 1 if x+timedelta( days = 1 ) in self.uk_holidays else 0 )
        ymax = cdf[val_label].max()
        ymin = cdf[val_label].min()
        print( '\n\n'+'****'*20+('\nInput Range <- [ %.3f , %.3f ] '%(ymin, ymax)) + '\n'+'****'*20 +'\n\n' )
        cdf.loc[:,'ynorm'] = ( cdf[ val_label ] - ymin ) / ( ymax - ymin )  # Sets Input Range <- [ 0 , 1 ]
        cdf.loc[:,'dt'] = cdf[time_label]
        cdf.loc[:,'ds'] = cdf['dt']
        cdf = cdf.set_index('dt')
        val_cols , nh_cols = self.data_shifter( cdf= cdf,val_label = 'ynorm',nh_label = 'inh',train = True)
        cdf = cdf.dropna()
        len_train = cdf['ynorm'].count()
        X_encoder = np.zeros((len_train, self.look_back, self.encoder_feat))
        for i in range( -self.look_back,0):
            col = '%dy'%i
            idx = 28+i
            X_encoder[:,idx,0] = cdf[col]
            col = '%dinh'%i
            X_encoder[:,idx,1] = cdf[col]
        Y_mlp = np.zeros( (len_train,1,1) )
        Y_mlp[:,0,0] = cdf['0y']
        # cdf = cdf.reset_index()
        beta = 0.4
        cdf['h_rbf'] = cdf[ 'ds' ].dt.date.apply( lambda x : np.exp(-1*(( (.8 if x in self.uk_holidays else  .2 if timedelta(days = 1) + x in uk_holidays or  x - timedelta(days = 1)  in uk_holidays  else  -10000) - .8)**2)/(2*beta))  )
        
        # cdf['h_rbf'] = cdf['h_rbf'].apply( lambda x : np.exp(-1*((x - .8)**2)/(2*beta))   )
        for i in range(0,7):
            cdf['%dw'%i] = cdf['ds'].dt.dayofweek.apply( lambda x : np.exp(-1*((self.symm_diff(i,x))**2)/(2*beta))  )
        X_mlp_in = cdf[['h_rbf']+['%dw'%i for i  in range(0,7)]].values
        X_mlp_in = X_mlp_in.reshape(-1,1,8)
        backup = X_encoder[:,0,0].copy()
        for i in range(self.look_back):
            X_encoder[:,i,0] -= backup
        Y_mlp[:,0,0] -= backup[:]
        return ( (ymin, ymax), backup, (X_encoder,Y_mlp), X_mlp_in )

    
    def prep_vae_train( self,cdf,val_label='Value',time_label='Time',series='current'):
        '''
         Function to prepare the data for training encoder-decoder(VAE) model only
        '''
        cdf = cdf.copy().reset_index()
        cdf.loc[:,time_label] = pd.to_datetime( cdf[time_label] )
        cdf.loc[:,'inh'] = cdf[time_label].dt.date.apply( lambda x : 1 if x+timedelta( days = 1 ) in self.uk_holidays else 0 )
        ymax = cdf[val_label].max()
        ymin = cdf[val_label].min()
        print( '\n\n'+'****'*20+('\nInput Range <- [ %.3f , %.3f ] '%(ymin, ymax)) + '\n'+'****'*20 +'\n\n' )
        cdf.loc[:,'ynorm'] = ( cdf[ val_label ] - ymin ) / ( ymax - ymin )  # Sets Input Range <- [ 0 , 1 ]
        cdf.loc[:,'dt'] = cdf[time_label]
        cdf.loc[:,'ds'] = cdf['dt']
        cdf = cdf.set_index('dt')
        val_cols , nh_cols = self.data_shifter( cdf= cdf,val_label = 'ynorm',nh_label = 'inh',train = True)
        # forward shifter
        for i in range(1,7):
            col = '+%dy'%i
            val_cols.append(col)
            cdf.loc[:,col] = cdf['ynorm'].shift(periods= 1, freq = timedelta(days=-i))
            col = '+%dinh'%i
            cdf.loc[:,col] = cdf['inh'].shift(periods= 1, freq = timedelta(days=-i))

        cdf = cdf.dropna()

        len_train = cdf['ynorm'].count()

        X_encoder = np.zeros((len_train, self.look_back, self.encoder_feat))
        X_decoder = np.zeros( (len_train,self.use_next,self.encoder_feat) )
        Y_decoder = np.zeros( (len_train,self.use_next,self.encoder_feat) )

        for i in range( -self.look_back,0):
            col = '%dy'%i
            idx = 28+i
            X_encoder[:,idx,0] = cdf[col]
            if i >= -6:
                X_decoder[:,6+i,0] = cdf[col]
            col = '%dinh'%i
            X_encoder[:,idx,1] = cdf[col]
            if i >= -6:
                X_decoder[:,6+i,1] = cdf[col]

        for i in range( 0,7):
            idx = i
            sym ='' if i == 0 else '+'
            col = '%s%dy'%(sym,i)
            # Y_decoder[:,idx,0] = np.log(cdf[col])
            Y_decoder[:,idx,0] = cdf[col]
            col = '%s%dinh'%(sym,i)
            Y_decoder[:,idx,1] = cdf[col]
        
        backup = X_encoder[:,0,0].copy()
        for i in range(self.look_back):
            X_encoder[:,i,0] -= backup
        for i in range(self.use_next):
            X_decoder[:,i,0] -= backup
            Y_decoder[:,i,0] -= backup
        return ( (ymin, ymax), backup, (X_encoder,X_decoder), Y_decoder )
        
    def get_original(self,backup,y_pred , ymin = 0, ymax = 1 ):
        '''
        Returns the data in the actual scale 
        Parameters:
        ------------------------------------------
        ``backup`` : generated by `` prep_to_train_mlp ``\n
        ``y_pred`` : data predicted by VAE-MLP \n
        ``ymin``   : generated by `` prep_to_train_mlp ``, the min value of the data\n
        `ymax`   : generated by `` prep_to_train_mlp ``, the max value of the data\n
        Returns:
        ------------------------------------------
        `y_original` : denormalized data( original OPM values )
        '''
        y_original  = y_pred.reshape(len(y_pred))
        y_original += backup 
        # y_original = np.exp( y_original ) - self.offset_val
        y_original = (y_original)*(ymax - ymin) + ymin
        # y_original = y_original* self.ymax  + self.ymin
        return y_original

def filter_order_downloads( dataframe , time_label ):
    '''
    Removes the opm values in between time 00:00(UK) to 06:00(UK)
    Parameters :
    -----------------------------------------------
    ``dataframe``  : time series dataframe\n
    ``time_label`` : label of time col
    Assumptions :
    -------
    the time column of the datframe is already in the datetime format \n
    if not so then  convert it by using pd.to_datetime(..)
    Returns :
    -----------------------------------------------
    ``df`` : dataframe with opm values in [ 00:00, 06:00 ] time removed
    '''
    df  = dataframe.copy()
    mask =  df[time_label].dt.time >= time(6,0,0)
    df = df[mask]
    return df

def read_df( path_to_csv, time_label='Time', val_label='Value', series = 'current', sep =';' ,  ):
    '''
    Reads the data from csv
    Params :
    ----------
    `path_to_csv` : path to the csv containing the time series data\n
    `time_label` : label of the time column\n
    `val_label` : label of the opm values\n
    `series` :  1w / 2w / current, default : ``'current'``\n
    `sep` : separator used for separating the column values, default : `';'`
    Returns:
    ----------
    `df` : read dataframe 
    '''
    to_UK_tz = True
    df = pd.read_csv(path_to_csv , sep = sep )
    mask =  df['Series'] == series 
    df = df[mask]
    if to_UK_tz:
        df.loc[:,time_label] =  pd.to_datetime( df[time_label].astype(str).str[:-6]  ).sub( timedelta( minutes = 330 ) )
    else:
        df.loc[:,time_label] = pd.to_datetime( df[time_label]  )
    return df
def read_and_clean(path_to_csv,time_label='Time',val_label='Value',series='current',sep=';'):
    df = read_df(path_to_csv,time_label,val_label,series,sep)
    return filter_order_downloads(df,time_label)

def join_df( path_csv1, path_csv2, time_label='Time', val_label = 'Value' , series = 'Series' ,sep=';', save_to = None ):
    '''
    Parameters:
    -----------
    `path_csv1` : Path to first csv\n
    `path_csv2` : Path to second csv\n
    `save_to`   : Path to save the dataframe  at, if not specified then df won't be saved\n
    Return :
    -----------
    `return_df` : Joined dataframe
    '''
    import os.path as path
    if not path.exists(path_csv1):
        print('[ERROR] Check if path_csv1 exists')
        exit(-6)
    if not path.exists(path_csv2):
        print('[ERROR] Check if path_csv2 exists')
        exit(-6)
    df1 = pd.read_csv(path_csv1,sep = sep).reset_index()
    df2 = pd.read_csv(path_csv2,sep = sep).reset_index()
    df1 = path_csv1
    df2 = path_csv2
    df1.loc[:,time_label] = pd.to_datetime( df1[ time_label ])
    df2.loc[:,time_label] = pd.to_datetime( df2[ time_label ])
    s1 = list(df1[time_label].head(1))[0]
    s2 = list(df2[time_label].head(1))[0]
    e1 = list(df1[time_label].tail(1))[0]
    e2 = list(df2[time_label].tail(1))[0]
    first_part = ''
    end_part = ''
    if s1 ==  s2 :
        if e1 == e2 :
            print(' Both this dataframe are same') 
            exit(1)
        else:
            if e1 > e2 :
                first_part = df1
                mask       = df2[time_label] > e1
                end_part   = df2[ mask ]
            else :
                first_part = df2
                mask = df1[time_label] > e2
                end_part = df1[mask]
    else : 
        if s1 > s2 :
            if e1 <= e2 :
                print('[WARNING] csv1 is already in csv2')
                exit(2)
            else :
                mask = df2[time_label] < s1 
                first_part = df2[mask]
                end_part = df1
        else:
            if e1 >= e2 :
                print('[WARNING] csv2 is already in csv1')
                exit(2)
            else :
                end_part = df2
                mask = df1[time_label] < s2
                first_part = df1[mask]
    return_df = pd.concat( [ first_part , end_part ] ) 
    return_df = return_df[ [ time_label, val_label, series  ] ]
    if save_to != None :
        return_df.to_csv( save_to , sep = sep, index = False )
    return return_df



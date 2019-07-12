
# Time Series Forecasting
A time series is a series of data points indexed in time order. Most commonly, a time series is a sequence taken at successive equally spaced points in time. For example the stocks data where the data is collected at the end of day. Orders Per Minute(OPM) is also an example of time series where the data is collected at the end of every minute.Time Series data more or less have the seasonality - 'trend' which repeats often at a certain trend like the weekly seasonality where the pattern of opm is repeated every week or even might repeat every day,like peak and low time of the opm might be consistent.
Time Series Forecasting is an interesting field where in the one needs to predict the future data points(opm in our case). Various Time Series specific statistical models like [ARIMA](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average),[Prophet by Facebook](https://facebook.github.io/prophet/) can be used to achieve time series forecasting.
Our approach is to leverage the power of [Neural Network](https://www.youtube.com/watch?v=aircAruvnKk). The paper [Deep and Confident Prediction for Time Series at Uber](https://arxiv.org/pdf/1709.01907.pdf) suggests that Bayesian Neural Network can be used for time series predictions with good uncertainity bounds. 
In our implementation we have deviated a bit from the paper where in we tried to use the model only for the predictions not for finding the uncertainity bound, which can also be done if required. The model is implemented using [Keras](https://keras.io/) in ``python`` language.
![Figure 1](https://gecgithub01.walmart.com/ukgr/opm_forecasting/blob/master/figure/image.png)<br/>
The above figure presents an overview of the model used for forecasting. Although the model might appear similar to be the one suggested in the [paper](https://arxiv.org/pdf/1709.01907.pdf)  but there exists a small difference in the internals of model.<br/>Our process can be broken into two steps:
  * Pre-Training
  * Inference/Prediction<br/>
  
 #### Pre-Training:
 This step involves using Encoder-Decoder model where in encoder extracts feature out of the data and decoder uses this extracted feature to generate the original data back.LSTMs for encoder as well as for decoder. The entire pre-training step is done in *supervised* manner( the target data is known, for example regression. )<br/>
[LSTMs](https://machinelearningmastery.com/gentle-introduction-long-short-term-memory-networks-experts/) are known to learn the order dependence( in our case the time order dependence) in the sequence predictions.
Here the encoders are fed in with the data since last 28 days back  for the same time and then the *thought-vector* is passed to the decoder which is in turn fed with the data since last seven days back at the same time and the decoder in turn tries to generate the data for next week. The idea that the *thought vector* would be learned in such a way that it might generalize the future data,i.e. the one week ahead data.The details of the structure used is presented below:
![VAE](https://gecgithub01.walmart.com/ukgr/opm_forecasting/blob/master/vae.png)

#### Inference/Prediction:
This step involves using a [Multilayer Perceptron(MLP)](https://en.wikipedia.org/wiki/Multilayer_perceptron) , is quite robust to outliers,
The feature extracted by the VAE( the model trained in the pre-training step) is fed to MLP along with the following features :
 * Holiday : if the date in concern is a holiday or not
 * Day of the Week : which day of the week is the current day and is represented as a distribution
Instead of using [one-hot encoding](https://en.wikipedia.org/wiki/One-hot), [Radial Basis Function](https://en.wikipedia.org/wiki/Radial_basis_function) is used which is believed to give better results. The number of hidden layers and the units in each hidden layer is similar to that suggested in this [paper](https://arxiv.org/pdf/1709.01907.pdf).<br/>
Below is the structure of the model used for  MLP <br/>
![MLP](https://gecgithub01.walmart.com/ukgr/opm_forecasting/blob/master/mlp.png)
<br/>
The layers with encoder is the layer from the VAE, which looks back into the past data and gives feature extracted out of it.
Dropouts are used instead of using traditional regularizers as which also acheives the goal of regularization and also it helps to generalize, a lot well, the distribution of the data.

## Data Cleaning and Normalization 
ASDA OPM time series has a system generated spikes,i.e. Order Download by system, in the time window `00:00 - 06:30` in UK time zone, 
so this time window has been removed from the time series and furthermore there are instances where due to failures( system crash, payment failures , etc.) where the OPM trends unusually. Those instances were handled using removing the trend and then doing a linear interpolation with adding noise to replace the unusual instances.
Further more the data is normalized to the range `[0-1]`.

#### Input Data
The input data to the encoder consists of last 28 days information for the given time. It has 2 features per 28 days. The features are normalized opm values and if the next day is holiday or not. Further more the opm values for each of last 28 days was subtracted from the normalized opm value at the last 28th day.This was done so as to avoid any exponential effects.
The input/output data to the decoder is similar to that of encoder but the data consists of last/future 7 days data with subtraction offset being last 28th days' normalized opm value.

####  References
Following references can be useful:
 * [Introduction to Forecasting in Machine Learning and Deep Learning](https://www.youtube.com/watch?v=bn8rVBuIcFg&t=598s)
 * [Intro to. Neural Network](https://www.youtube.com/watch?v=aircAruvnKk)
 * [Keras Functional API](https://keras.io/getting-started/functional-api-guide/)
 * [Two Effective Algorithms for Time Series Forecasting](https://www.youtube.com/watch?v=VYpAodcdFfA&t=3s)
 * [Linear Model For Time Series Forecasting](https://www.youtube.com/watch?v=68ABAU_V8qI)
 
### To Execute

#### Active Learning :
    `      python .\train.py    --input_path <path_to_csv> --model_path <path_to_mlp> --encoder_path <path_to_vae> --pred_dir <dir_to_csv_in> --model_dir <dir_to_store_model> --epochs <number_of_epochs>    
            usage: train.py [-h] [--auto_dt AUTO_DT] [--date_to_pred DATE_TO_PRED]
                            --input_path INPUT_PATH --model_path MODEL_PATH --encoder_path
                            ENCODER_PATH [--model_dir MODEL_DIR] [--epochs EPOCHS]
                            [--batch_size BATCH_SIZE] [--pred_dir PRED_DIR]
                            [--to_UK_tz TO_UK_TZ]

            optional arguments:
            -h, --help            show this help message and exit
            --auto_dt AUTO_DT     True -> pick the date automatically False -> Specifiy
                                    the date
            --date_to_pred DATE_TO_PRED
                                    date [YYYY-MM-DD] for which prediction is to be done
            --input_path INPUT_PATH
                                    path to the input dataset for training / testing
            --model_path MODEL_PATH
                                    path to the stored model for saving / loading
            --encoder_path ENCODER_PATH
                                    path to the stored encoder-decoder model for loading
            --model_dir MODEL_DIR
                                    dir in which model is to be stored
            --epochs EPOCHS       number of epochs, applicable only for training
            --batch_size BATCH_SIZE
                                    batch size
            --pred_dir PRED_DIR   directory to store the output
            --to_UK_tz TO_UK_TZ   True => datetime in UK time zone else in INDIAN time
                                    zone 
      `
    
#### Training MLP only :
    `usage: model_train.py [-h] [--fresh FRESH] --input_path INPUT_PATH
                                [--mlp_path MLP_PATH] [--freeze_encoder FREEZE_ENCODER]
                                --encoder_path ENCODER_PATH [--store_at STORE_AT]
                                [--epochs EPOCHS] [--batch_size BATCH_SIZE]

            arguments:
            -h, --help            show this help message and exit
            --fresh FRESH         True => Train the model from scratch !!
            --input_path INPUT_PATH
                                    path to the input dataset for training / testing
            --mlp_path MLP_PATH   path to the stored mlp model for loading
            --freeze_encoder FREEZE_ENCODER
                                    True => the encoder layer will be frozen during
                                    training
            --encoder_path ENCODER_PATH
                                    path to the stored encoder-decoder model for loading
            --store_at STORE_AT   directory in which model is to be stored
            --epochs EPOCHS       number of epochs, applicable only for training
            --batch_size BATCH_SIZE
                                    batch size
    `
    
#### Training Encoder Only:
          `usage: encoder_train.py [-h] [--fresh FRESH] --input_path INPUT_PATH
                                    [--encoder_path ENCODER_PATH] [--store_at STORE_AT]
                                    [--epochs EPOCHS] [--batch_size BATCH_SIZE]
            optional arguments:
            -h, --help            show this help message and exit
            --fresh FRESH         True => Train the model from scratch !!
            --input_path INPUT_PATH
                                    path to the input dataset for training / testing
            --encoder_path ENCODER_PATH
                                    path to the stored encoder-decoder model for loading
            --store_at STORE_AT   directory in which model is to be stored
            --epochs EPOCHS       number of epochs, applicable only for training
            --batch_size BATCH_SIZE
                                    batch size
           `
### Project By:

#### Ankit Kumar Singh 
Summer Intern 19 <br/>
CSE Dept. IIT Kanpur, India

#### Sibaprasad Tripathy
Mentor,<br/>
Senior Software Engineer,<br/>
Intl UK eCom - Leeds Support
Walmart Labs India

#### Sameesh Gupta
Mentor,Manager,<br/>
Senior Manager II- Quality Engineering,<br/>
Intl UK eCom - Leeds Support<br/>
Walmart Labs India


   

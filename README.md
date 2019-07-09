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
Dropouts are used instead of using traditional regularizers as which also acheives the goal of regularization and also it helps to generalize, a lot well, the distribution of the data. <br/>
## Refrences: 
 * [Intro to. Neural Network](https://www.youtube.com/watch?v=aircAruvnKk)
 * [Keras Functional API](https://keras.io/getting-started/functional-api-guide/)

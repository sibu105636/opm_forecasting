# Time Series Forecasting
A time series is a series of data points indexed in time order. Most commonly, a time series is a sequence taken at successive equally spaced points in time. For example the stocks data where the data is collected at the end of day. Orders Per Minute(OPM) is also an example of time series where the data is collected at the end of every minute.Time Series data more or less have the seasonality - 'trend' which repeats often at a certain trend like the weekly seasonality where the pattern of opm is repeated every week or even might repeat every day,like peak and low time of the opm might be consistent.
Time Series Forecasting is an interesting field where in the one needs to predict the future data points(opm in our case). Various Time Series specific statistical models like [ARIMA](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average),[Prophet by Facebook](https://facebook.github.io/prophet/) can be used to achieve time series forecasting.
Our approach is to leverage the power of [Neural Network](https://www.youtube.com/watch?v=aircAruvnKk). The paper [Deep and Confident Prediction for Time Series at Uber](https://arxiv.org/pdf/1709.01907.pdf) suggests that Bayesian Neural Network can be used for time series predictions with good uncertainity bounds. 
In our implementation we have deviated a bit from the paper where in we tried to use the model only for the predictions not for finding the uncertainity bound, which can also be done if required.
![Figure 1](https://gecgithub01.walmart.com/a0s051m/time_series/blob/master/figure/image.png)<br/>
The above figure presents an overview of the model used for forecasting. Although the model might appear similar to be the one suggested in the [paper](https://arxiv.org/pdf/1709.01907.pdf)  but there exists a small difference in the internals of model.<br/>Our process can be broken into two steps:
  * Pre-Training
  * Inference/Prediction<br/>
 #### Pre-Training:
 This step involves using LSTMs for encoder as well as for decoder.<br/>
[LSTMs](https://machinelearningmastery.com/gentle-introduction-long-short-term-memory-networks-experts/) are known to learn the order dependence( in our case the time order dependence) in the sequence predictions.
## Refrences: 
[Intro to. Neural Network](https://www.youtube.com/watch?v=aircAruvnKk)

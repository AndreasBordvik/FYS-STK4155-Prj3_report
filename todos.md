# General goals

* Implement Sliding Window to use own NN
* Implement Andreas Dataset for RNN and LSTM
* Implement Layer Normalization for better result
* Implement 1D convolution of data to move further backwards in time
* Remove trend and seasonality

# Theory

* RNNs -> LSTMs
* Sliding Window
* Time Series Data
* ENTSO data

# Fyllingsgrad data
Dataen fra NVE er forskjøvet, starter 03. jan
Dette må skiftes på, f.eks ved å starte de tre første dagene fra ENTSO tidsseriene passer

# Forecast - Actual
Undersøke om forholdet mellom forecast og actual for både load og generation er en interessant feature, 
Spearman correlation

# What we want to show
## Our best models

Both are Seq-to-Seq

Univariate multilayered SimpleRNN 128 and 128 neurons

Multivariate forecast 1d convolution 64 filters kernel size 3 stride 3 multilayer GRU 128 and 64
## results

Grid search both models in terms of learning rate and batch size

Forecast med en antatt veldig god modell

Create a iterative MSE that shows the MSE at each predicted hour
# Other results

Study learning rate dependance for all cells



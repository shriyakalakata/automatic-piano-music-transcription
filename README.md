# Automatic Piano Music Transcription

This project aims to develop a machine learning model capable of predicting the activations of MIDI notes from an audio file containing a musical piece.

## Code Structure
```
├── datasets/ 
├── data_analysis.ipynb
├── data_preprocessing.ipynb
├── amt_baseline.ipynb
├── amt_dnn.ipynb
├── amt_lstm.ipynb
```

## Dataset

The dataset used is the [OMAPS2](https://github.com/itec-hust/OMAPS2) dataset, which consists of audio recordings from a piano in .wav format and corresponding manually annotated music transcription sheets in .txt format. The dataset is already split into train, validation, and test sets.

## Preprocessing

The audio data was converted into a constant-Q transform (CQT) form, which decomposes the audio signal into frequency components over time. The CQT vectors are then aligned with the MIDI annotations using one-hot encoding.

## Model Architecture

**Baseline Model**: The One-vs-Rest Classifier with a Logistic Regression estimator was used as a baseline for comparison.

Two main architectures were explored:

1. **Deep Neural Network (DNN)**: The DNN takes CQT vectors as input and outputs the one-hot encoded MIDI activations. It consists of several hidden layers with ReLU activation functions and employs techniques like dropout and early stopping to prevent overfitting.

2. **Long Short-Term Memory (LSTM)**: The LSTM architecture was implemented to capture short-term and long-term dependencies in the audio data. Transfer learning was also explored by initializing the LSTM weights with pre-trained weights from the DNN models.

## Evaluation

The models' performance was evaluated using metrics like accuracy (a modified version that doesn't take True Negatives (TN) into consideration), precision, recall, and F1-score. The predictions were compared against the time-aligned MIDI annotations.

## Results

The best DNN model achieved an accuracy of 37.32%, outperforming the baseline logistic regression model, which achieved an accuracy of 10.75%. However, the LSTM models struggled to achieve satisfactory accuracy despite various optimization strategies.

## Conclusion

While the results were not ideal, this project provides valuable insights into the challenges of automatic music transcription and paves the way for future research in this area, potentially exploring attention-based models, transformers, etc.

## Contributors
* Argy Sakti
* Sean Wiryadi
* Shriya Kalakata
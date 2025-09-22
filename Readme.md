## 🧠 Breast Cancer Classification – Experiment Report

This project explores how different neural network architectures and hyperparameters affect performance on the Breast Cancer Wisconsin dataset.
The dataset contains 30 features for each sample, with a binary target:
```bash
0 = malignant

1 = benign
```
The goal: achieve high classification accuracy while understanding the effect of design choices in neural networks.

## 🔬 Experiments
## 1. Epochs

Trained models for up to 25 epochs.

Observed how accuracy steadily improved over time.

Too few epochs → underfitting.

Beyond ~25 epochs → diminishing returns.

## 2. Hidden Layers & Neurons

[32] neurons: strong baseline, simple and effective.

[64] or [128, 64]: slightly higher accuracy, but marginal improvements.

Very large hidden layers → no significant gains and risk of overfitting.

Key Insight: Small networks (just 32 neurons) were enough for this dataset.

## 3. Activation Functions

We compared:

ReLU – stable, strong default.

Tanh – consistently gave the best results.

Sigmoid – slower learning, weaker.

Softplus – surprisingly competitive in some trials.

LeakyReLU – decent, but not top.

## 4. Optimizers

Adam → consistently strong and stable.

AdamW → close to Adam, slight regularization benefits.

RMSprop → decent but less reliable.

SGD → unstable unless tuned carefully.

## 5. Learning Rate

Too high (0.01) → unstable, poor results.

Too low (0.0005) → underfitting, accuracy dropped (~89%).

Sweet spot (0.005) → best accuracy (~95–98%).

## 📊 Results
Random Search Trials (Examples)
```bash
Trial 2: Tanh + Adam, hidden=[32], lr=0.005 → val_acc=98.82%

Trial 6: Softplus + AdamW, hidden=[32], lr=0.01 → val_acc=98.82%

Trial 7: Softplus + RMSprop, hidden=[128,64], lr=0.001 → val_acc=98.82%

Trial 11: Sigmoid + RMSprop, hidden=[32,32], lr=0.01 → val_acc=98.82%
```
Despite several configs reaching high validation accuracy, Tanh + Adam with [32] neurons was the most consistent and generalizable.

## 🏆 Best Parameters

Activation: Tanh

Optimizer: Adam

Hidden Layers: [32]

Learning Rate: 0.005

Accuracy: ~95–98%

## ✅ Conclusion

Small networks are enough – even 32 neurons achieved top accuracy.

Activation matters – Tanh consistently outperformed ReLU and others.

Learning rate is critical – too high/low hurts performance; 0.005 worked best.

Adam optimizer is the safest choice for this dataset.

👉 With the best configuration, the model reached ~97% accuracy on test data, showing that a well-tuned MLP can effectively handle real-world medical classification tasks.
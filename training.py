import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_utils import load_and_process_data, create_and_fit_encoder, split_dataset
from data_utils import scale_numeric_features, convert_to_tensor, apply_one_hot_encoding


class RaceOutcomePredictor(nn.Module):
    def __init__(self, input_dim):
        super(RaceOutcomePredictor, self).__init__()

        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 64)

        # New layer
        self.layer3 = nn.Linear(64, 32)

        self.output_layer = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout(x)

        x = torch.relu(self.layer2(x))
        x = self.dropout(x)

        # Using the new layer
        x = torch.relu(self.layer3(x))
        x = self.dropout(x)

        return self.output_layer(x)


def prepare_data():
    race_data = load_and_process_data()
    encoder = create_and_fit_encoder(
        race_data, ['circuit', 'name', 'constructor'])
    race_data = apply_one_hot_encoding(
        encoder, race_data, ['circuit', 'name', 'constructor'])
    scalers = scale_numeric_features(
        race_data, ['result', 'start_position', 'year', 'month', 'day'])

    input_data = race_data.drop("result", axis=1)
    input_data = input_data.astype(
        {col: 'int' for col in input_data.select_dtypes(['bool']).columns})
    output_data = race_data["result"]
    return input_data, output_data, scalers, encoder


def initialize_hyperparameters():
    return 100, 32, 0.00001


def initialize_model(input_dim, learning_rate):
    model = RaceOutcomePredictor(input_dim)
    loss_function = nn.MSELoss()
    model_optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(
        model_optimizer, 'min', patience=10, factor=0.1, verbose=True)

    return model, loss_function, model_optimizer, scheduler


def execute_training(model, scheduler, loss_function, optimizer, train_loader, val_loader, epochs):
    training_losses = []
    validation_losses = []
    training_loss = None

    for epoch in range(epochs):
        model.train()
        for data_batch in train_loader:
            inputs, targets = data_batch
            optimizer.zero_grad()
            predictions = model(inputs)
            training_loss = loss_function(predictions, targets)
            training_loss.backward()
            optimizer.step()

        model.eval()
        validation_loss = 0
        with torch.no_grad():
            for data_batch in val_loader:
                inputs, targets = data_batch
                predictions = model(inputs)
                validation_loss += loss_function(predictions, targets).item()

        scheduler.step(validation_loss)

        if training_loss is not None:
            current_val_loss = validation_loss / len(val_loader)
            training_losses.append(training_loss.item())
            validation_losses.append(current_val_loss)

            st.write(f"Epoch {epoch + 1} of {epochs} | "
                     f"Training Loss: {training_loss.item():.4f} | "
                     f"Validation Loss: {current_val_loss:.4f}")

    return training_losses, validation_losses


def predict_upcoming_races(predictor, upcoming_race_data, scalers, encoder):
    processed_upcoming_data = apply_one_hot_encoding(
        encoder, upcoming_race_data, ['circuit', 'name', 'constructor'])

    for col in ['start_position', 'year', 'month', 'day']:
        processed_upcoming_data[col] = scalers[col].transform(
            upcoming_race_data[col].values.reshape(-1, 1))

    processed_upcoming_data = processed_upcoming_data.astype(
        {col: 'int' for col in processed_upcoming_data.select_dtypes(['bool']).columns})

    upcoming_race_tensor = convert_to_tensor(processed_upcoming_data)

    upcoming_race_loader = DataLoader(
        upcoming_race_tensor, batch_size=64, shuffle=False)

    predictor.eval()
    predictions = []
    with torch.no_grad():
        for batch in upcoming_race_loader:
            outputs = predictor(batch)
            predictions.extend(outputs)

    return predictions


def denormalize_prediction(prediction, scaler):
    return scaler.inverse_transform(np.array([prediction]).reshape(-1, 1))[0][0]


def show_training_and_validation_loss_graph(training_losses, validation_losses):
    plt.figure(figsize=(12, 6))
    plt.plot(training_losses, label="Training Loss")
    plt.plot(validation_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss Over Epochs")
    st.pyplot(plt)


def number_to_ordinal(n):
    if 10 <= n % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suffix}"


def create_upcoming_race_data_for_positions():
    data = {
        'start_position': list(range(1, 23)),
        'year': [2023] * 22,
        'month': [10] * 22,
        'day': [8] * 22,
        'circuit': ['Losail International Circuit'] * 22,
        'name': ['Carlos Sainz'] * 22,
        'constructor': ['Ferrari'] * 22
    }

    return pd.DataFrame(data)


def create_and_show_predictions(predictor, scalers, encoder):
    upcoming_race_data = create_upcoming_race_data_for_positions()
    race_predictions = predict_upcoming_races(
        predictor, upcoming_race_data, scalers, encoder)

    st.write("### Race Outcome Predictions")
    for i, prediction_tensor in enumerate(race_predictions):
        starting_position_ordinal = number_to_ordinal(i + 1)
        predicted_position_ordinal = number_to_ordinal(
            round(denormalize_prediction(prediction_tensor.item(), scalers['result'])))

        st.write(f"If {upcoming_race_data['name'].iloc[0]} starts {starting_position_ordinal}, "
                 f"he would finish {predicted_position_ordinal}")


def calculate_test_loss(predictor, test_set, batch_size):
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    criterion = nn.MSELoss()
    predictor.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            data, target = batch
            outputs = predictor(data)
            test_loss += criterion(outputs, target).item()

    st.write(f"Test Loss: {test_loss / len(test_loader):.4f}")


def main():
    st.title("Race Outcome Predictor")

    input_data, output_data, scalers, encoder = prepare_data()

    feature_tensor = convert_to_tensor(input_data)
    label_tensor = convert_to_tensor(output_data).unsqueeze(1)

    # Set hyperparameters
    EPOCHS, BATCH_SIZE, LEARNING_RATE = initialize_hyperparameters()

    predictor, loss_function, model_optimizer, scheduler = initialize_model(
        feature_tensor.shape[1], LEARNING_RATE)

    # Split dataset
    full_dataset = TensorDataset(feature_tensor, label_tensor)
    training_set, validation_set, test_set = split_dataset(full_dataset)
    train_loader = DataLoader(
        training_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(
        validation_set, batch_size=BATCH_SIZE, shuffle=False)

    # Train the model
    st.write("Training the model...")
    training_losses, validation_losses = execute_training(predictor, scheduler, loss_function,
                                                          model_optimizer, train_loader, val_loader, EPOCHS)

    show_training_and_validation_loss_graph(training_losses, validation_losses)

    # Test loss
    st.write("Calculating test loss...")
    calculate_test_loss(predictor, test_set, BATCH_SIZE)

    # Generate and display predictions for upcoming race
    st.write("### Predictions for Upcoming Race")
    create_and_show_predictions(predictor, scalers, encoder)


if __name__ == "__main__":
    main()

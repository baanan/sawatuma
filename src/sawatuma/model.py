from typing import Any
import torch
from torch import nn
from torch.utils.data import DataLoader
import math

from sawatuma.datasets import ListeningCountsDataset, Parameters

from math import sqrt


class Model(nn.Module):
    def __init__(
        self, parameters: Parameters, latent_factor_count: int, learning_rate: float
    ):
        super().__init__()

        drop_percentage = 0.25

        # Embedding layers
        self.user_factor_layer = nn.Sequential(
            nn.Linear(parameters.user_count(), latent_factor_count),
            # nn.Dropout(p=drop_percentage),
        )
        nn.init.xavier_uniform_(self.user_factor_layer[0].weight)
        self.track_factor_layer = nn.Sequential(
            nn.Linear(parameters.track_count(), latent_factor_count),
            # nn.Dropout(p=drop_percentage),
        )
        nn.init.xavier_uniform_(self.track_factor_layer[0].weight)

        # Hidden layers
        self.neural_net = nn.Sequential(
            nn.Linear(2 * latent_factor_count, latent_factor_count),
            nn.ReLU(),
            # nn.Dropout(p=drop_percentage),
            nn.Linear(latent_factor_count, latent_factor_count // 2),
            nn.ReLU(),
            # nn.Dropout(p=drop_percentage),
            nn.Linear(latent_factor_count // 2, parameters.rating_size),
            nn.Softmax(dim=1),
        )

        # use mean squared error for the loss function
        self.loss_function = nn.CrossEntropyLoss()

        # technically, a gradient descent solver isn't the best for matrix factorization
        # (it's usually better to use alternating least squares, especially for parallelization)
        # but pytorch doesn't support ALS, so use Adam instead
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, user_one_hots: torch.Tensor, track_one_hots: torch.Tensor):
        user_factors = self.user_factor_layer(user_one_hots)
        track_factors = self.track_factor_layer(track_one_hots)
        # concatenate both the user and track factors together to serve as an input
        concatenated_factors = torch.cat((user_factors, track_factors), dim=1)
        # use a non-linear neural network instead of matrix multiplication
        # because neural networks are smart or something who knows
        return self.neural_net(concatenated_factors)

    def calculate_loss(
        self, found: torch.Tensor, expected: torch.Tensor
    ) -> torch.Tensor:
        return self.loss_function(found, expected)

    def optimize(self, loss: torch.Tensor):
        # zero the gradients of the optimizer
        self.optimizer.zero_grad()
        # backpropagate the loss into the optimizer, inputting new gradients
        loss.backward()
        # optimize the neural network using the gradients from the loss
        self.optimizer.step()

    def train_once(self, dataset: ListeningCountsDataset, *, batch_size: int = 512):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.train()
        for batch, (user_one_hots, track_one_hots, expected_ratings) in enumerate(
            dataloader
        ):
            print(f"batch {batch + 1}/{len(dataloader)}...")
            found_ratings = self(user_one_hots, track_one_hots)
            expected_ratings = expected_ratings.to(torch.float32)
            loss = self.calculate_loss(found_ratings, expected_ratings)
            print(f"  mean loss: {loss.mean().item()}")
            print(f"  first {example(found_ratings[0], expected_ratings[0])}")
            self.optimize(loss)

    def evaluate(self, dataset: ListeningCountsDataset) -> Any:
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

        print("evaluating...")

        self.eval()
        total_loss = 0
        total_points = 0

        with torch.no_grad():
            for user_one_hots, track_one_hots, expected_ratings in dataloader:
                found_ratings = self(user_one_hots, track_one_hots)
                expected_ratings = expected_ratings.to(torch.float32)
                loss = self.calculate_loss(found_ratings, expected_ratings)
                total_loss += loss.sum().item()
                total_points += math.prod(loss.size())
                print(f"  {example(found_ratings[0], expected_ratings[0])}")

        average_loss = total_loss / total_points

        return average_loss


def example(found: torch.Tensor, expected: torch.Tensor) -> str:
    found_rating = torch.argmax(found)
    expected_rating = torch.argmax(expected)

    found_components = [f"{item:.2f}" for item in found.tolist()]
    found_components = f"[{", ".join(found_components)}]"

    return f"found: {found_rating} ({found_components}), expected: {expected_rating}"

from typing import Any
import torch
from torch import nn
from torch.utils.data import DataLoader

from sawatuma.datasets import ListeningCountsDataset, Parameters


class Model(nn.Module):
    def __init__(
        self, parameters: Parameters, latent_factor_count: int, learning_rate: float
    ):
        super().__init__()

        # Embedding layers
        self.user_factor_layer = nn.Linear(parameters.user_count, latent_factor_count)
        nn.init.xavier_uniform_(self.user_factor_layer.weight)
        self.track_factor_layer = nn.Linear(
            parameters.track_count(), latent_factor_count
        )
        nn.init.xavier_uniform_(self.track_factor_layer.weight)

        # Hidden layers
        self.neural_net = nn.Sequential(
            nn.Linear(2 * latent_factor_count, latent_factor_count),
            nn.Sigmoid(),
            nn.Linear(latent_factor_count, 1),
        )

        # use mean squared error for the loss function
        self.loss_function = nn.MSELoss()

        # technically, simple gradient descent isn't the best optimizer
        # (it's usually better to use alternating least squares, especially for parallelization)
        # but pytorch doesn't support ALS, so use SGD instead
        self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)

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

    def train_once(self, dataset: ListeningCountsDataset):
        dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

        self.train()
        for batch, (user_one_hots, track_one_hots, expected_ratings) in enumerate(
            dataloader
        ):
            print(f"batch {batch}/{len(dataloader)}...")
            found_ratings = self(user_one_hots, track_one_hots)
            expected_ratings = expected_ratings.unsqueeze(1).to(torch.float32)
            loss = self.calculate_loss(found_ratings, expected_ratings)
            print(f"  mean loss: {loss.mean().item()}")
            self.optimize(loss)

    def evaluate(self, dataset: ListeningCountsDataset) -> Any:
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

        total_loss = 0

        with torch.no_grad():
            for user_one_hots, track_one_hots, expected_ratings in dataloader:
                self.eval()
                found_ratings = self(user_one_hots, track_one_hots)
                expected_ratings = expected_ratings.unsqueeze(1).to(torch.float32)
                loss = self.calculate_loss(found_ratings, expected_ratings)
                total_loss += loss.sum().item()

        average_loss = total_loss / len(dataset)

        return average_loss

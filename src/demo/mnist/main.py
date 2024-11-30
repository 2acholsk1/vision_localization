import neptune
import torch
from model import Net
from pytorch_metric_learning import distances, losses, miners, reducers
from pytorch_metric_learning.utils.accuracy_calculator import \
    AccuracyCalculator
from testing import test
from train import train

from logger import get_logger
from utils import get_mnist_loaders


def main():
    # Polaczenie z neptune
    run = neptune.init_run(
        monitoring_namespace='monitoring'
        )

    # Ustawienia
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 256
    num_epochs = 10
    learning_rate = 0.01
    loss_margin = 0.2
    logger = get_logger()

    # Dane
    train_loader, test_loader = get_mnist_loaders(batch_size)

    # Model
    model = Net().to(device)

    # Straty i funkcje metryczne
    distance = distances.CosineSimilarity()
    reducer = reducers.ThresholdReducer(low=0)
    loss_func = losses.TripletMarginLoss(margin=loss_margin, distance=distance, reducer=reducer)
    mining_func = miners.TripletMarginMiner(
        margin=0.2, distance=distance, type_of_triplets="semihard"
    )
    accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)

    # Optymalizator

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    run["parameters"] = {
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "loss_margin": loss_margin,
        "distance": "cosine_similarity",
        "reducer": "treshold_reducer",
        "loss_func": "triplet_margin_loss",
        "mining_func": "triplet_margin_miner",
        "optimizer": "Adam"
    }

    # Trening
    logger.info("Starting training...")
    train(
        model, train_loader, loss_func, mining_func, optimizer, device, num_epochs, logger, run
    )

    # Testowanie
    logger.info("Starting testing...")
    test(model, train_loader.dataset, test_loader.dataset, accuracy_calculator, logger, run)

    torch.save(model.state_dict(), "model.pth")
    run["model/weights"].upload("model.pth")

    run.stop()


if __name__ == "__main__":
    main()

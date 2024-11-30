from pytorch_metric_learning import testers


def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)


def test(model, train_set, test_set, accuracy_calculator, logger, neptune_run):
    model.eval()
    train_embeddings, train_labels = get_all_embeddings(train_set, model)
    test_embeddings, test_labels = get_all_embeddings(test_set, model)

    train_labels = train_labels.squeeze(1)
    test_labels = test_labels.squeeze(1)

    logger.info("Computing accuracy...")
    accuracies = accuracy_calculator.get_accuracy(
        test_embeddings, test_labels, train_embeddings, train_labels, False
    )
    logger.info(f"Test set accuracy (Precision@1): {accuracies['precision_at_1']:.4f}")
    neptune_run["test/accuracy"].log(accuracies["precision_at_1"])

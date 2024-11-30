def train_one_epoch(model, loss_func, mining_func, device, train_loader, optimizer, logger, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(data)
        indices_tuple = mining_func(embeddings, labels)
        loss = loss_func(embeddings, labels, indices_tuple)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % 20 == 0:
            logger.info(
                f"Epoch {epoch}, Batch {batch_idx}: Loss = {loss.item()}, "
                f"Mined triplets = {mining_func.num_triplets}"
            )
    return total_loss / len(train_loader)


def train(model, train_loader, loss_func, mining_func, optimizer, device, num_epochs, logger):
    for epoch in range(1, num_epochs + 1):
        avg_loss = train_one_epoch(
            model, loss_func, mining_func, device, train_loader, optimizer, logger, epoch
        )
        logger.info(f"Epoch {epoch} completed. Average Loss: {avg_loss:.4f}")

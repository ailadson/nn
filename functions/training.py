import config
from functions.activations import cross_entropy
from functions.data_prep import batch_data
import time

def evaluate(observations, net):
    loss_sum = 0
    misclassification = 0
    num_observations = len(observations)

    for (ipt, expected_output) in observations:
        true_class_index = list(expected_output).index(1)
        output = net.forward_propagate(ipt)
        loss_sum += cross_entropy(output, true_class_index)
        estimated_true_class_idx = list(output).index(max(output))
        misclassification += (
            1 if true_class_index != estimated_true_class_idx else 0
        )

    avg_loss = loss_sum / num_observations
    misclassification_rate = misclassification / num_observations
    return (avg_loss, misclassification_rate)

def train_epoch(
        nn,
        trainer,
        epoch_num,
        training_set,
        validation_set,
        batch_size):
    print(f"Epoch number {epoch_num}")
    batches = batch_data(training_set, batch_size)

    train_loss, train_misclass_rate = 0.0, 0.0
    start_time = time.time()
    for batch_idx, batch in enumerate(batches):
        batch_train_loss, batch_train_misclass_rate = (
            trainer.train_with_examples(batch)
        )
        train_loss += batch_train_loss
        train_misclass_rate += batch_train_misclass_rate

        if (batch_idx + 1) % config.BATCHES_PER_EVALUATION == 0:
            valid_loss, valid_misclass_rate = evaluate(
                validation_set, nn
            )
            print(f">>Epoch number {epoch_num} | "
                  f"Batch {batch_idx + 1}/{len(batches)} | "
                  f"Valid Loss: {valid_loss:.3f} | "
                  f"Valid Misclass Rate: {valid_misclass_rate:.3f}")
        if (batch_idx + 1) % config.BATCHES_PER_LOG == 0:
            train_loss /= config.BATCHES_PER_LOG
            train_misclass_rate /= config.BATCHES_PER_LOG
            total_time = time.time() - start_time
            examples_per_second = (
                config.BATCHES_PER_LOG * batch_size / total_time
            )
            print(f"Epoch number {epoch_num} | "
                  f"Batch {batch_idx + 1}/{len(batches)} | "
                  f"Train Loss: {train_loss:.3f} | "
                  f"Train Misclass Rate: {train_misclass_rate:.3f} | "
                  f"Examples per second: {examples_per_second:.3f}")
            train_loss, train_misclass_rate = 0.0, 0.0
            start_time = time.time()

    loss, misclassification = evaluate(validation_set, nn)
    print(f"Epoch number {epoch_num} | Loss: {loss} | "
          f"Misclassification: {misclassification}")

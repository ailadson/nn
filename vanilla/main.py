from net import Net
from file_reader import FileReader
from vocabulary import Vocabulary
from trainer import Trainer
from data_set_functions import *
import time


label_d, review_d = FileReader.run()
v = Vocabulary(review_d, label_d)
reviews = v.featurize_reviews()

nn = Net(v.num_of_items())
nn.add_layer(20)
nn.add_output_layer(1)

t = Trainer(nn, 0.1)

train_data, validation_data, test_data = segment_data(list(zip(reviews, label_d)))

def train_epoch(epoch_num, training_set, validation_set):
    start_time = time.time()
    print(f"Epoch number {epoch_num}")
    batches = batch_data(training_set, 1)

    for i, batch in enumerate(batches):
        # print(f"Batch {i + 1} of {len(batches)}")
        if i % 100 == 0:
            time_diff = time.time() - start_time
            per_sec = i/time_diff
            print(f"Examples per second: {per_sec:.3f}")
        batch_loss, batch_misclas_rate = t.train_with_examples(batch)
        # print(f"Batch Loss: {batch_loss} | Batch Misclassification: {batch_misclas_rate}")

    loss, misclassification = evaluate(validation_set, nn)
    print(f"Epoch number {epoch_num} Loss: {loss} | Epoch number {epoch_num} Misclassification: {misclassification}")
    print("")


for i in range(1):
    train_epoch(i, train_data, validation_data)

            # strings = [f"\r\x1b[KEpoch {self.epoch_num}",
            #            f"Batch #{b_num}.",
            #            f"Train Error rate: {error_rate:.3f}",
            #            f"Speed: {int(eps):5}ex/sec"]
            # sys.stdout.write("\t".join(strings))



print(t.train_with_examples( [(reviews[0], label_d[0])] ))

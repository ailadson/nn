from net import Net
from file_reader import FileReader
from vocabulary import Vocabulary


label_d, review_d = FileReader.run()
v = Vocabulary(review_d, label_d)
reviews = v.featurize_reviews()

nn = Net(v.num_of_items())
nn.add_layer(20)
nn.add_layer(1)
print(nn.forward_propagate(reviews[0]))

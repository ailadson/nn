#
# 1
#
# print("MODEL 1")
# print("~"*20)
# # Epoch number 9 Loss: 1.4870173504333104 | Epoch number 9 Misclassification: 0.0258
# nn1 = Net(mnist.test.images.shape[1])
# nn1.add_layer(397)
# nn1.add_output_layer(10)
#
# t1 = Trainer(nn1, 0.1)
#
# for i in range(10):
#     train_epoch(nn1, t1, i, train_observations, validation_observations)
# prompt_save(nn1)
#
# print("MODEL 2")
# print("~"*20)
# #Epoch number 9 Loss: 1.4840154230518507 | Epoch number 9 Misclassification: 0.0242
# nn2 = Net(mnist.test.images.shape[1])
# nn2.add_layer(510)
# nn2.add_layer(203)
# nn2.add_output_layer(10)
#
# t2 = Trainer(nn2, 0.1)
#
# for i in range(10):
#     train_epoch(nn2, t2, i, train_observations, validation_observations)
# prompt_save(nn2)
#
#
# for i in range(10):
#     train_epoch(nn3, t3, i, train_observations, validation_observations)
# prompt_save(nn3)

from utils.datareader import Datareader
from recommenders.cdae import CDAE

import numpy as np
np.random.seed(0)

# read data
dr = Datareader(mode='offline', only_load=True, test_num='1', verbose=False)

# create instance of the Recommender
cdae = CDAE()

# initialize URM
cdae.get_data(datareader=dr, from_file=True)

# setup NN architecture
cdae.create(I=cdae.urm.shape[1], U=cdae.urm.shape[0], K=50)
cdae.compile()
cdae.summary()

# fit model
# Not working with sparse matrix yet
cdae.fit_generator(x=[cdae.urm, cdae.train_x_users], y=cdae.urm, batch_size=128, epochs=10, verbose=1)

'''
import movie_lens
import metrics

# data
train_users, train_x, test_users, test_x = movie_lens.load_data()
train_x_users = numpy.array(train_users, dtype=numpy.int32).reshape(len(train_users), 1)
test_x_users = numpy.array(test_users, dtype=numpy.int32).reshape(len(test_users), 1)

# model
model = cdae.create(I=train_x.shape[1], U=len(train_users)+1, K=50,
                    hidden_activation='relu', output_activation='sigmoid', q=0.50, l=0.01)
model.compile(loss='mean_absolute_error', optimizer='adam')
model.summary()

# train
history = model.fit(x=[train_x, train_x_users], y=train_x,
                    batch_size=128, epochs=15, verbose=1,
                    validation_data=[[test_x, test_x_users], test_x])

# predict
pred = model.predict(x=[train_x, numpy.array(train_users, dtype=numpy.int32).reshape(len(train_users), 1)])
pred = pred * (train_x == 0) # remove watched items from predictions
pred = numpy.argsort(pred)

for n in range(1, 11):
    sr = metrics.success_rate(pred[:, -n:], test_x)
    print("Success Rate at {:d}: {:f}".format(n, sr))
'''
'''
Success Rate at 1: 27.783669
Success Rate at 2: 39.236479
Success Rate at 3: 45.281018
Success Rate at 4: 49.310710
Success Rate at 5: 51.219512
Success Rate at 6: 53.234358
Success Rate at 7: 54.188759
Success Rate at 8: 55.673383
Success Rate at 9: 56.733828
Success Rate at 10: 57.688229
'''

'''import csv
import graphlab as gl

# Use collaborative filtering to make recommendations

train_file = 'train.csv'
test_file  = 'test.csv'
soln_file  = 'collab_filter.csv'

data = gl.SFrame.read_csv(train_file, column_type_hints={"plays":int})
model = gl.recommender.create(data, user_id="user", item_id="artist", target="plays")
testdata = gl.SFrame(test_file)

user = '306e19cce2522fa2d39ff5dfc870992100ec22d2'
artist = '4ac4e32b-bd18-402e-adad-ae00e72f8d85'

a = model.predict(gl.SFrame(user), gl.SFrame(artist))
print "a is ", a

# Write out test solutions.

with open(test_file, 'r') as test_fh:
    test_csv = csv.reader(test_fh, delimiter=',', quotechar='"')
    next(test_csv, None)

    with open(soln_file, 'w') as soln_fh:
        soln_csv = csv.writer(soln_fh,
                              delimiter=',',
                              quotechar='"',
                              quoting=csv.QUOTE_MINIMAL)
        soln_csv.writerow(['Id', 'plays'])

        for row in test_csv:
            id     = row[0]
            user   = row[1]
            artist = row[2]
            users = []
            items = []
            users.append(user)
            items.append(artist)
            frame = gl.SFrame(row)
            
            ls = [user, artist]
            #soln_csv.writerow([id, model.predict(gl.SFrame(users=user), gl.SFrame(items=artist))])
            #soln_csv.writerow([id, model.predict(gl.SFrame(users), gl.SFrame(items))])
            #soln_csv.writerow([id, model.predict(frame)])
            #soln_csv.writerow([id, model.predict(gl.SFrame(user), gl.SFrame(artist))])
            #soln_csv.writerow([id, model.predict(gl.SFrame(ls))])'''
            
'''
#import numpy as np
import csv
#import scipy.sparse as spr
import graphlab as gl


train_file = 'train.csv'
test_file  = 'test.csv'
soln_file  = 'collab_filter_saroj.csv'


data = gl.SFrame.read_csv(train_file, column_type_hints={"plays":int})
model = gl.recommender.item_similarity_recommender.create(data, user_id="user", item_id="artist", target="plays")


result = model.predict(gl.SFrame(test_file))

normalized_result = (result - result.min())/(result.max() - result.min()) * data['plays'].max()


with open(soln_file, 'w') as soln_fh:
    soln_csv = csv.writer(soln_fh,
                          delimiter=',',
                          quotechar='"',
                          quoting=csv.QUOTE_MINIMAL)
    soln_csv.writerow(['Id', 'plays'])

    for i in range(4154804):
        soln_csv.writerow([i + 1, normalized_result[i]])'''
        

# coding

import numpy as np
import csv
from sklearn import cross_validation
from sklearn import decomposition

# Predict via the median number of plays.

train_file = 'train.csv'
test_file  = 'test.csv'
soln_file  = 'global_median.csv'


# Load the training data.
train_data = {}
xc_train_data={}
xc_test_data={}
plays_list=[]
with open(train_file, 'r') as train_fh:
    train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
    next(train_csv, None)
    counter=0
    for row in train_csv:
        user   = row[0]
        artist = row[1]
        plays  = int(row[2])
    
        if not user in train_data:
            train_data[user] = {}
        
        if not user in xc_train_data:
            xc_train_data[user] ={}
        if not user in xc_test_data:
            xc_test_data[user] ={}
        
        if counter < 116643:
            xc_train_data[user][artist] = plays
        else:
            xc_test_data[user][artist] = plays
        plays_list+=[plays]
        counter+=1
        
        #train_data[user][artist] = plays



users_list = xc_train_data.keys()
users_np = np.array(users_list)
artists_dict={}
for user in users_list:
    for artist in xc_train_data[user].keys():
        if artist not in artists_dict:
            artists_dict[artist]=1
        else:
            artists_dict[artist]+=1
artists_list = np.array(artists_dict.keys())
#print users_np.shape
num_users = users_np.shape[0]
#print artists_list.shape
num_artists = artists_list.shape[0]



data = np.zeros([num_users, num_artists])
for i in xrange(num_users):
    for j in xrange(num_artists):
        try:
            data[i][j]= xc_train_data[users_np[i]][artists_list[j]]
        except KeyError:
            data[i][j]= 118.0


test_components=1000
model = decomposition.NMF(n_components=test_components, random_state=1)
part_one = model.fit_transform(data)
part_two=model.components_
mult_result = np.dot(part_one, part_two)

print "mult_result is ", mult_result

plays_array = []
for user, user_data in train_data.iteritems():
    for artist, plays in user_data.iteritems():
        plays_array.append(plays)
global_median = np.median(np.array(plays_array))
#print "global median:", global_median


with open(test_file, 'r') as test_fh:
    test_csv = csv.reader(test_fh, delimiter=',', quotechar='"')
    next(test_csv, None)

    with open(soln_file, 'w') as soln_fh:
        soln_csv = csv.writer(soln_fh,
                              delimiter=',',
                              quotechar='"',
                              quoting=csv.QUOTE_MINIMAL)
        soln_csv.writerow(['Id', 'plays'])

        for row in test_csv:
            id     = row[0]
            user   = row[1]
            artist = row[2]

            soln_csv.writerow([id, global_median])






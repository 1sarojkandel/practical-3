import numpy as np
import csv
from sklearn.cluster import AgglomerativeClustering

# Predict via the user-specific median.
# If the user has no data, use the global median.

train_file = 'train.csv'
test_file  = 'test.csv'
soln_file  = 'agglom_solution.csv'
shortFile = 'short.csv'
userFile = 'profiles.csv'
shortUserFile = 'shortProfiles.csv'

# Load the training data.
train_data = {}
artistData = {}
userList = []
artistList = []
playsArray = []

count = 0
limit = 20
artistCount = 0
userCount = 0

# with open(train_file, 'r') as train_fh:
with open(train_file, 'r') as train_fh:
    train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
    next(train_csv, None)
    for row in train_csv:
        user   = row[0]
        artist = row[1]
        plays  = int(row[2])
        playsArray.append(plays)
    
        if not user in userList:
            train_data[user] = {}
            train_data[user]["index"] = userCount
            userCount += 1
            train_data[user]["playsArray"] = []
            userList.append(user)
        
        train_data[user][artist] = int(plays)
        train_data[user]["playsArray"].append(plays)
        train_data[user]["age"] = 0

        if not artist in artistList:
            artistData[artist] = {}
            artistData[artist]["playsArray"] = []
            artistData[artist]["index"] = artistCount
            artistCount += 1
            artistList.append(artist)

        artistData[artist]["playsArray"].append(int(plays))
        
        count += 1
        # if (count > limit):
        #     break
        if (count % 50000 == 0):
            print count

print "read train data"
globalMedian = np.median(np.array(playsArray))
globalSD = np.std(np.array(playsArray))

tempAgeArray = []
count = 0
#read in user data
with open(userFile, 'r') as user_fh:
    profile_csv = csv.reader(user_fh, delimiter=',', quotechar='"')
    next(profile_csv, None)
    for row in profile_csv:
        user = row[0]
        sex = row[1]
        age = 0
        country = row[3]
        count += 1
        # if (count > limit):
        #     break
        if not user in train_data:
            train_data[user] = {}
        if (not (row[2] == '')):
            age = int(row[2])
            tempAgeArray.append(age)
            train_data[user]["age"] = int(age)
        else:
            train_data[user]["age"] = 0
        train_data[user]["sex"] = sex
        train_data[user]["country"] = country

print "finished reading user data"
userAges = np.asarray(tempAgeArray)
medianAge = np.median(userAges)
meanAge = np.mean(userAges)
sdAge = np.std(userAges)

# Compute the global median and per-user median.
# plays_array  = []
# user_medians = {}
# userRatios = []
# for user, user_data in train_data.iteritems():
#     # print user_data
#     user_plays = []
#     for artist, plays in user_data.iteritems():
#         plays_array.append(plays)
#         user_plays.append(plays)

#     user_medians[user] = np.median(np.array(user_plays))
# global_median = np.median(np.array(plays_array))

# print "computed median"

featureMatrix = []
for i in range(len(userList)):
    userFeatures = []
    user = userList[i]
    userData = train_data[user]
    userPlays = np.array(userData["playsArray"])
    # print userPlays

    userMean = np.mean(userPlays)
    userSD = np.std(userPlays)
    userMedian = np.median(userPlays)
    train_data[user]["mean"] = userMean
    train_data[user]["SD"] = userSD
    train_data[user]["median"] = userMedian

    for j in range(len(artistList)):
        artist = artistList[j]
        # artistData = artistData[artist]

        if (artist in userData):
            numPlays = userData[artist]
            normalized = 0
            if (userSD == 0):
                normalized = 0
            else:
                normalized = ((float(numPlays) - float(userMean)) / float(userSD))
            userFeatures.append(normalized)
        else:
            userFeatures.append(0)

    # print userData
    userAge = userData["age"]
    if (userAge == 0):
        userFeatures.append(0)
    else:
        normalizedAge = ((float(userAge) - float(meanAge)) / float(sdAge))
        userFeatures.append(normalizedAge)

    featureMatrix.append(userFeatures)

print "computed Features"

n_digits = 200
#kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
kmeans = AgglomerativeClustering(n_clusters = n_digits, linkage = "average")
kmeans.fit(featureMatrix)
means = kmeans.cluster_centers_
labels = kmeans.labels_

print "computed clusters"

# print means
# print labels
# print userRatios

testLimit = 11

with open(test_file, 'r') as test_fh:
    test_csv = csv.reader(test_fh, delimiter=',', quotechar='"')
    next(test_csv, None)

    with open(soln_file, 'w') as soln_fh:
        soln_csv = csv.writer(soln_fh,
                              delimiter=',',
                              quotechar='"',
                              quoting=csv.QUOTE_MINIMAL)
                              
                              
        soln_csv.writerow(['Id', 'plays'])

        count = 0
        for row in test_csv:
            id     = row[0]
            user   = row[1]
            artist = row[2]
            count += 1
            if (count % 50000 == 0):
                print count
            # if (count > testLimit):
            #     break
            if (user not in userList):
                print "User", id, "not in training data."
                soln_csv.writerow([id, globalMedian])
            else:
                userIndex = train_data[user]["index"]
                cluster = labels[userIndex]
                if (artist not in artistList):
                    print "Artist", id, "not in training data."
                    soln_csv.writerow([id, train_data[user]["median"]])
                else:
                    artistIndex = artistData[artist]["index"]
                    artistScore = means[cluster][artistIndex]
                    userSD = train_data[user]["SD"]
                    if (userSD == 0):
                        solution = float(artistScore) * globalSD + train_data[user]["mean"]
                    else:
                        solution = float(artistScore) * train_data[user]["SD"] + train_data[user]["mean"]

                    if (user in userList):
                        soln_csv.writerow([id, solution])
                    else:
                        print "User", id, "not in training data."
                        soln_csv.writerow([id, globalMedian])


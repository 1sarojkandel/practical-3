import numpy as np
import csv

# Predict via the user-specific median.
# If the user has no data, use the global median.

train_file = 'train.csv'
test_file  = 'test.csv'
soln_file  = 'country.csv'
userFile = 'profiles.csv'

count = 0
limit = 20

# Load the training data.
train_data = {}
with open(train_file, 'r') as train_fh:
    train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
    next(train_csv, None)
    for row in train_csv:
        user   = row[0]
        artist = row[1]
        plays  = row[2]
    
        if not user in train_data:
            train_data[user] = {}
            train_data[user]["plays"] = {}
        
        train_data[user]["plays"][artist] = int(plays)
        count += 1
        # if (count > limit):
        #     break
        if (count % 50000 == 0):
            print count

print "done reading training"

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
            # tempAgeArray.append(age)
            train_data[user]["age"] = int(age)
        else:
            train_data[user]["age"] = 0
        train_data[user]["sex"] = sex
        train_data[user]["country"] = country

print "read user data"

# Compute the global median and per-user median.
plays_array  = []
user_medians = {}
countryPlays = {}
countryMedians = {}
resultMedians = {}
artistDict = {}
for user, user_data in train_data.iteritems():
    user_plays = []

    if ("plays" in user_data):
        for artist, plays in user_data["plays"].iteritems():
            artistDict[artist] = 1
            plays_array.append(plays)
            user_plays.append(plays)
            if ("country") in train_data[user]:
                country = train_data[user]["country"]
                if (country):
                    if (country in countryMedians):
                        countryMedians[country].append(plays)
                    else:
                        countryMedians[country] = [plays]
                    if (country in countryPlays):
                        if (artist in countryPlays[country]):
                            countryPlays[country][artist].append(plays)
                        else:
                            # print plays
                            countryPlays[country][artist] = [plays]
                    else:
                        countryPlays[country] = {}
                        countryPlays[country][artist] = [plays]
        # print user_plays
        user_medians[user] = np.median(np.array(user_plays))
global_median = np.median(np.array(plays_array))

for country in countryPlays:
    resultMedians[country] = np.median(np.array(countryMedians[country]))

print "about to get lit"

countryArtistDict = {}
countryInfo = {}
for country in countryPlays:
    countryArtistVector = []
    curDict = countryPlays[country]
    countryInfo[country] = {}
    countryArtistDict[country] = {}
    for artist in artistDict:
        if (artist in curDict):
            countryArtistVector.append(np.median(np.array(curDict[artist])))
        else:
            countryArtistVector.append(resultMedians[country])
    npArray = np.array(countryArtistVector)
    countryMean = np.mean(npArray)
    countrySD = np.std(npArray)
    countryInfo[country]["mean"] = countryMean
    countryInfo[country]["SD"] = countrySD
    i = 0
    for artist in artistDict:
        value = countryArtistVector[i]
        zscore = (float(value - countryMean) / float(countrySD))
        countryArtistDict[country][artist] = zscore
        i += 1
    


print "done computing"

testLimit = 11

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

            count += 1
            if (count % 50000 == 0):
                print count
            # if (count > testLimit):
            #     break

            if user in user_medians:
                userMedian = user_medians[user]
                if ("country") not in train_data[user]:
                    soln_csv.writerow([id, userMedian])
                else:
                    country = train_data[user]["country"]
                    zscore = countryArtistDict[country][artist]
                    result = (zscore * countryInfo[country]["SD"]) + userMedian
                    soln_csv.writerow([id, result])
                
            else:
                print "User", id, "not in training data."
                if ("country") in train_data[user]:
                    soln_csv.writerow([id, countryMedians[train_data[user]["country"]]])
                else:
                    soln_csv.writerow([id, global_median])
                

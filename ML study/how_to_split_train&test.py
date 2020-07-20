def get_one_fold(data, turn, fold=10):
    tot_length = len(data)
    each = int(tot_length / fold)
    mask = np.array([True if each * turn <= i < each * (turn + 1) else False for i in list(range(tot_length))])
    return data[~mask], data[mask]

def runCV(clf, shuffled_data, shuffled_labels, fold=10, isAcc = True):
    from sklearn.metrics import precision_recall_fscore_support
    results = []
    for i in range(fold):
        train_data, test_data = get_one_fold(shuffled_data, i, fold=fold)
        train_labels, test_labels = get_one_fold(shuffled_labels, i, fold=fold)
        clf = clf.fit(train_data, train_labels)
        pred = clf.predict(test_data)
        correct = pred == test_labels
        if isAcc:
            acc = sum([1 if x == True else 0 for x in correct]) / len(correct)
            results.append(acc)
        else:
            results.append(precision_recall_fscore_support(pred, test_labels))
        return results

def random_shuffle(data):
    from random import shuffle, seed
    advertising = pd.read_csv('./Advertising.csv', usecols=[1, 2, 3, 4])

    #seed 값을 설정함으로써 매번 같은 shuffle 결과값을 얻을 수 있음
    seed(0)
    #numbers(index값)을 shuffle
    numbers = list(range(len(data)))
    shuffle(numbers)
    #shuffle 된 number를 이용해 advertising도 shuffle, 그 결과값을 shuffled_data에 저장
    shuffled_data = advertising[numbers]

    #split_train_test

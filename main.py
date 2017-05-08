from input_file import getData, getLabel
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import time
def draw_by_pixel(ndarray):
    im = ndarray.reshape(28, 28)
    plt.figure()
    plt.imshow(im, cmap='gray')
    plt.show()
    return 'draw success!'


def just_test(s, n1, n2):
    data = getData(s)
    label = getLabel(s)
    for x in range(n1, n2):
        print(label[x])
        draw_by_pixel(data[x])

# print "train group:"
# just_test('train', 5, 10)
# print "test group:"
# just_test('test', 105, 110)
train_data ,train_lables = getData('train'),getLabel('train').squeeze()
test_data ,test_lables = getData('test'),getLabel('test').squeeze()

t0 =time.time()
nb_clf = GaussianNB()
nb_clf.fit(train_data,train_lables)
nb_pred = nb_clf.predict(test_data)
print("nb_clf has fitted,time cost :%.3fs"%(time.time() -t0))
print("the accuracy of Gaussian navie bayes classifier:\n",accuracy_score(test_lables, nb_pred))  


from sklearn.decomposition import PCA
n_components = 100

pca = PCA(svd_solver='randomized',n_components = n_components,whiten = True).fit(train_data)

pca_train_data = pca.transform(train_data)
pca_test_data = pca.transform(test_data)


t0 = time.time()
svc_clf = SVC()
#svc_clf.fit(train_data[:len(train_data)//50],train_lables[:len(train_data)//50])
svc_clf.fit(pca_train_data,train_lables)
svc_pred = svc_clf.predict(pca_test_data)
print("svc_clf has fitted,time cost :%2f"%(time.time() -t0))
print("the accuracy of svm classifier:\n",accuracy_score(test_lables, svc_pred))  
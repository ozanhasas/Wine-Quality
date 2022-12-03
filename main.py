import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

accuracy_table = {}
sc = StandardScaler()
wine = pd.read_csv('dataset.csv',sep=';')
wine = wine[wine['quality']<9]

attributes = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide',
              'total sulfur dioxide','density','pH','sulphates','alcohol']
x = wine[attributes]
y = wine['quality']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0,stratify=y)
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

rfc = RandomForestClassifier(n_estimators=200)
dtc = DecisionTreeClassifier()

rfc.fit(x_train,y_train)
dtc.fit(x_train,y_train)

rfc_predict = rfc.predict(x_test)
dtc_predict = dtc.predict(x_test)

rfc_conmax = confusion_matrix(y_test,rfc_predict)
dtc_conmax = confusion_matrix(y_test,dtc_predict)

scores = cross_val_score(rfc, x_train, y_train, cv=10)
mean = scores.mean()
accuracy_table["Random Forest"] = mean
scores = cross_val_score(dtc,x_train, y_train, cv=10)
mean = scores.mean()
accuracy_table["Decision Tree"] = mean

print("Classifier\t\t\tAccuracy Score")

for i in accuracy_table.keys():
    print(i +"\t\t"+ str("%0.3f" % accuracy_table[i]))

test = ([7,0.27,0.36,20.7,0.045,45,170,1.001,3,0.45,8.8],[5,0.25,0.45,15,0.05,0,159,1.008,5,0.5,9])

result_rfc = rfc.predict(test)
result_dtc = dtc.predict(test)
print()
print("Result by Random Forest>>>")
print(result_rfc)
print("Result by Decision Tree>>>")
print(result_dtc)
print()

print(dtc_conmax)
print()
print(rfc_conmax)

plt.matshow(rfc_conmax)
plt.title("Confusion Matrix")
plt.colorbar()
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()

plt.matshow(dtc_conmax)
plt.title("Confusion Matrix")
plt.colorbar()
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()

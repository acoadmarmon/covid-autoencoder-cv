from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split
import shutil

covid_path = '../archive/COVID'
non_covid_path = '../archive/non-COVID'
covid_files = [join(covid_path, f) for f in listdir(covid_path) if isfile(join(covid_path, f))]
non_covid_files = [join(non_covid_path,f) for f in listdir(non_covid_path) if isfile(join(non_covid_path, f))]


covid_train, covid_test = train_test_split(covid_files, test_size=0.2, random_state=42)
non_covid_train, non_covid_test = train_test_split(non_covid_files, test_size=0.2, random_state=42)



for i in covid_train:
    shutil.copy(i, '../images/train/covid/')
for i in covid_test:
    shutil.copy(i, '../images/test/covid/')
for i in non_covid_train:
    shutil.copy(i, '../images/train/non_covid/')
for i in non_covid_test:
    shutil.copy(i, '../images/test/non_covid/')
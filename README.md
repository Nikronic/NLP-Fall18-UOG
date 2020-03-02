
# Final NLP project (University of Guilan)
In this repository, we implemented a Statistical NLP model to predict news agency, news tags, etc as final project of NLP course in university of Guilan.

# Contents
* Libraries and Constants
* Importing Data
* Preprocessing
* Creating Model for the First Task
* Creating Model for the Second Task
* Creating Model for the Third Task

## Libraries and Constants


```python
from __future__ import unicode_literals

import json
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from functools import reduce
from operator import add
from hazm import *
from utils.preprocessing import *
from utils.models import *
from copy import deepcopy

```


```python
# Data root path
data_root = 'data'
fars_file = 'farsnews.jsonl'
asriran_file = 'asriran.jsonl'
# Dataset dataframe column names
keys = None

# News headline tags
not_valid_labels = None

# News agencies
news_agencies = None
```

## Importing Data


```python
with open(os.path.join(data_root, asriran_file), encoding='utf-8') as jd:
    asriran = [json.loads(line) for line in jd]
    asriran = pd.DataFrame(asriran)
print('Number of Datapoints: {}'.format(len(asriran)))
```

    Number of Datapoints: 15000
    


```python
with open(os.path.join(data_root, fars_file), encoding='utf-8') as jd:
    fars = [json.loads(line) for line in jd]
    fars = pd.DataFrame(fars)
print('Number of Datapoints: {}'.format(len(fars)))
```

    Number of Datapoints: 15000
    

## Preprocessing

Finding Valid Labels:


```python
asr_labels = list(set(reduce(np.append, asriran.newsPathLinks.apply(lambda x: tuple(x.keys())))))
fars_labels = list(set(reduce(np.append, fars.newsPathLinks.apply(lambda x: list(x.keys())))))
```


```python
set((list(asr_labels) + list(fars_labels)))
```

Some labels are not valid so:


```python
not_valid_labels = [
     'دانلود',
     'ساير حوزه ها',
     'سایر حوزه ها',
     'دیگر رسانه ها',
     'نامشخص',
     'پیامک',
     'صفحه نخست',
     'عصرايران دو',
]
valid_labels = list(filter(lambda x: x not in not_valid_labels, list(set((list(asr_labels) + list(fars_labels))))))
```

Creating Documents & Labels:


```python
asriran_tags = asriran['tags'].apply(lambda x: ' '.join(list(x.keys())))
fars_tags = fars['tags'].apply(lambda x: ' '.join(list(x.keys())))
```


```python
title_count = 2
tag_count = 10
documents = np.append(asriran['body'] + ' ' + asriran['title'] * title_count + asriran_tags*tag_count,
                        fars['body'] + ' ' + fars['title'] * title_count + fars_tags*tag_count)
raw_labels = np.append(asriran.newsPathLinks.apply(lambda x: tuple(x.keys())),
                        fars.newsPathLinks.apply(lambda x: tuple(x.keys())))
org_labels = np.append( ['AsrIran'] * len(asriran), ['Fars'] * len(fars)) # For the third task
```

Removing Documents which are emtpy:


```python
none_zero_docs = list(map(lambda x: len(x) > 1, documents))
documents = documents[none_zero_docs]
raw_labels = cleans_labels(raw_labels[none_zero_docs], valid_labels)
org_labels = org_labels[none_zero_docs]
```

Duplicating documents for each of their labels:


```python
proc_documents, proc_labels = extend_labels(documents, raw_labels, valid_labels)
```

Normalizing & Tokenizing & Removing Stopwords Documents:


```python
normalizer = Normalizer()
word_filter = WordFilter()
documents = list(pd.Series(documents).apply(normalizer.normalize).apply(tokenize).apply(word_filter.filter_words))
proc_documents = list(proc_documents.apply(normalizer.normalize).apply(tokenize).apply(word_filter.filter_words))
```

Replacing words with less than 2 occurances with unknown word


```python
documents = make_unknown(documents)
proc_documents = make_unknown(proc_documents)
```

    created
    created
    

Making documents one hot encoded


```python
label_set, proc_labels = one_hot_encoder(proc_labels)
label_set_th, org_labels = one_hot_encoder(org_labels)
```

Deviding document to train and test datasets:


```python
x_train, y_train, x_test, y_test = train_test_split(proc_documents , proc_labels, train_size = 0.80, random_state=85)
x_train_th, y_train_th, x_test_th, y_test_th = train_test_split(documents , org_labels, train_size = 0.80, random_state=85)
```

## Creating Model for the First Task

Training:


```python
nb = NaiveBayes()
nb.fit(x_train, y_train)
```

    Vocab created
    P(c) calculated
    93
    %0.0 continue...
    %0.010752688172043012 continue...
    %0.021505376344086023 continue...
    %0.03225806451612903 continue...
    %0.043010752688172046 continue...
    %0.053763440860215055 continue...
    %0.06451612903225806 continue...
    %0.07526881720430108 continue...
    %0.08602150537634409 continue...
    %0.0967741935483871 continue...
    %0.10752688172043011 continue...
    %0.11827956989247312 continue...
    %0.12903225806451613 continue...
    %0.13978494623655913 continue...
    %0.15053763440860216 continue...
    %0.16129032258064516 continue...
    %0.17204301075268819 continue...
    %0.1827956989247312 continue...
    %0.1935483870967742 continue...
    %0.20430107526881722 continue...
    %0.21505376344086022 continue...
    %0.22580645161290322 continue...
    %0.23655913978494625 continue...
    %0.24731182795698925 continue...
    %0.25806451612903225 continue...
    %0.26881720430107525 continue...
    %0.27956989247311825 continue...
    %0.2903225806451613 continue...
    %0.3010752688172043 continue...
    %0.3118279569892473 continue...
    %0.3225806451612903 continue...
    %0.3333333333333333 continue...
    %0.34408602150537637 continue...
    %0.3548387096774194 continue...
    %0.3655913978494624 continue...
    %0.3763440860215054 continue...
    %0.3870967741935484 continue...
    %0.3978494623655914 continue...
    %0.40860215053763443 continue...
    %0.41935483870967744 continue...
    %0.43010752688172044 continue...
    %0.44086021505376344 continue...
    %0.45161290322580644 continue...
    %0.46236559139784944 continue...
    %0.4731182795698925 continue...
    %0.4838709677419355 continue...
    %0.4946236559139785 continue...
    %0.5053763440860215 continue...
    %0.5161290322580645 continue...
    %0.5268817204301075 continue...
    %0.5376344086021505 continue...
    %0.5483870967741935 continue...
    %0.5591397849462365 continue...
    %0.5698924731182796 continue...
    %0.5806451612903226 continue...
    %0.5913978494623656 continue...
    %0.6021505376344086 continue...
    %0.6129032258064516 continue...
    %0.6236559139784946 continue...
    %0.6344086021505376 continue...
    %0.6451612903225806 continue...
    %0.6559139784946236 continue...
    %0.6666666666666666 continue...
    %0.6774193548387096 continue...
    %0.6881720430107527 continue...
    %0.6989247311827957 continue...
    %0.7096774193548387 continue...
    %0.7204301075268817 continue...
    %0.7311827956989247 continue...
    %0.7419354838709677 continue...
    %0.7526881720430108 continue...
    %0.7634408602150538 continue...
    %0.7741935483870968 continue...
    %0.7849462365591398 continue...
    %0.7956989247311828 continue...
    %0.8064516129032258 continue...
    %0.8172043010752689 continue...
    %0.8279569892473119 continue...
    %0.8387096774193549 continue...
    %0.8494623655913979 continue...
    %0.8602150537634409 continue...
    %0.8709677419354839 continue...
    %0.8817204301075269 continue...
    %0.8924731182795699 continue...
    %0.9032258064516129 continue...
    %0.9139784946236559 continue...
    %0.9247311827956989 continue...
    %0.9354838709677419 continue...
    %0.946236559139785 continue...
    %0.956989247311828 continue...
    %0.967741935483871 continue...
    %0.978494623655914 continue...
    %0.989247311827957 continue...
    P(w|c) calculated
    

Train Evaluation:


```python
nb.evaluate(x_train, y_train, label_set=label_set)
```

    %0 continue...
    %1000 continue...
    %2000 continue...
    %3000 continue...
    %4000 continue...
    %5000 continue...
    %6000 continue...
    %7000 continue...
    %8000 continue...
    %9000 continue...
    %10000 continue...
    %11000 continue...
    %12000 continue...
    %13000 continue...
    %14000 continue...
    %15000 continue...
    %16000 continue...
    %17000 continue...
    %18000 continue...
    %19000 continue...
    %20000 continue...
    %21000 continue...
    %22000 continue...
    %23000 continue...
    %24000 continue...
    %25000 continue...
    %26000 continue...
    %27000 continue...
    %28000 continue...
    %29000 continue...
    %30000 continue...
    %31000 continue...
    Label مسئولیت های اجتماعی: 
         Precision: 0.4745762711864407
         Recall: 1.0
         F1-Measure: 0.6436781609195402
    Label صنعت ، تجارت ، بازرگانی: 
         Precision: 0.5139664804469274
         Recall: 0.9787234042553191
         F1-Measure: 0.673992673992674
    Label ایران در جهان: 
         Precision: 0.5211267605633803
         Recall: 0.9024390243902439
         F1-Measure: 0.6607142857142856
    Label شهری: 
         Precision: 0.472
         Recall: 0.9874476987447699
         F1-Measure: 0.638700947225981
    Label غرب از نگاه غرب: 
         Precision: 1.0
         Recall: 1.0
         F1-Measure: 1.0
    Label خانواده: 
         Precision: 1.0
         Recall: 1.0
         F1-Measure: 1.0
    Label تور و توپ: 
         Precision: 0.49710982658959535
         Recall: 0.9347826086956522
         F1-Measure: 0.6490566037735849
    Label فوتبال ایران: 
         Precision: 0.5058721183123097
         Recall: 0.917192429022082
         F1-Measure: 0.6520885898514158
    Label علمی: 
         Precision: 0.8245614035087719
         Recall: 0.7704918032786885
         F1-Measure: 0.7966101694915254
    Label اجتماعی: 
         Precision: 0.9508361204013378
         Recall: 0.5859439406430338
         F1-Measure: 0.7250701351695996
    Label سرگرمی: 
         Precision: 0.07942238267148015
         Recall: 1.0
         F1-Measure: 0.14715719063545152
    Label مسجد و هیئت: 
         Precision: 0.5
         Recall: 1.0
         F1-Measure: 0.6666666666666666
    Label فرهنگ و هنر: 
         Precision: 0.6198347107438017
         Recall: 0.08269018743109151
         F1-Measure: 0.14591439688715954
    Label احزاب و تشکل ها: 
         Precision: 0.5092592592592593
         Recall: 0.990990990990991
         F1-Measure: 0.672782874617737
    Label پاکستان: 
         Precision: 0.5179856115107914
         Recall: 0.9113924050632911
         F1-Measure: 0.6605504587155964
    Label بورس: 
         Precision: 0.47468354430379744
         Recall: 1.0
         F1-Measure: 0.6437768240343348
    Label گروههای توان خواه: 
         Precision: 0.41975308641975306
         Recall: 1.0
         F1-Measure: 0.5913043478260869
    Label بازار: 
         Precision: 0.5483870967741935
         Recall: 0.6891891891891891
         F1-Measure: 0.6107784431137725
    Label حماسه و مقاومت: 
         Precision: 0.6666666666666666
         Recall: 0.6666666666666666
         F1-Measure: 0.6666666666666666
    Label خبر خوب: 
         Precision: 1.0
         Recall: 1.0
         F1-Measure: 1.0
    Label آفریقا: 
         Precision: 0.48878923766816146
         Recall: 1.0
         F1-Measure: 0.6566265060240963
    Label زنان و جوانان: 
         Precision: 0.5066666666666667
         Recall: 0.95
         F1-Measure: 0.6608695652173914
    Label مجلس: 
         Precision: 0.4743169398907104
         Recall: 0.9455337690631809
         F1-Measure: 0.6317321688500728
    Label تاریخ: 
         Precision: 0.6153846153846154
         Recall: 0.8888888888888888
         F1-Measure: 0.7272727272727274
    Label جنگ اقتصادی: 
         Precision: 0.5
         Recall: 1.0
         F1-Measure: 0.6666666666666666
    Label سینما و تئاتر: 
         Precision: 0.48604651162790696
         Recall: 0.9766355140186916
         F1-Measure: 0.6490683229813664
    Label داستان کوتاه: 
         Precision: 0.9
         Recall: 1.0
         F1-Measure: 0.9473684210526316
    Label استانها: 
         Precision: 0.9779673063255153
         Recall: 0.8380024360535931
         F1-Measure: 0.9025910134470319
    Label انقلاب اسلامی: 
         Precision: nan
         Recall: 0.0
         F1-Measure: nan
    Label علم و فن آوری جهان: 
         Precision: 0.4959016393442623
         Recall: 0.9758064516129032
         F1-Measure: 0.6576086956521738
    Label اندیشه: 
         Precision: 0.4583333333333333
         Recall: 0.9428571428571428
         F1-Measure: 0.616822429906542
    Label امام و رهبری: 
         Precision: 0.53125
         Recall: 1.0
         F1-Measure: 0.6938775510204082
    Label شرق آسیا و اقیانوسیه: 
         Precision: 0.44285714285714284
         Recall: 0.9393939393939394
         F1-Measure: 0.6019417475728155
    Label تحلیل بین الملل: 
         Precision: nan
         Recall: 0.0
         F1-Measure: nan
    Label آسياي مرکزی و روسيه: 
         Precision: 0.5449591280653951
         Recall: 0.8547008547008547
         F1-Measure: 0.6655574043261231
    Label ورزش بانوان: 
         Precision: 0.5094339622641509
         Recall: 0.7941176470588235
         F1-Measure: 0.6206896551724137
    Label فرهنگی/هنری: 
         Precision: 0.8955512572533849
         Recall: 0.954639175257732
         F1-Measure: 0.9241516966067864
    Label فناوری و IT: 
         Precision: 0.86
         Recall: 0.9662921348314607
         F1-Measure: 0.9100529100529101
    Label حوادث: 
         Precision: 0.4929078014184397
         Recall: 0.9686411149825784
         F1-Measure: 0.6533490011750881
    Label آمریکا، اروپا: 
         Precision: 0.4913232104121475
         Recall: 0.9476987447698745
         F1-Measure: 0.6471428571428571
    Label ویژه نامه ها: 
         Precision: 0.6111111111111112
         Recall: 0.4520547945205479
         F1-Measure: 0.5196850393700787
    Label ورزش بین الملل: 
         Precision: 0.47424511545293074
         Recall: 0.9501779359430605
         F1-Measure: 0.6327014218009479
    Label آموزش و پرورش: 
         Precision: 0.4610951008645533
         Recall: 1.0
         F1-Measure: 0.6311637080867849
    Label محور مقاومت: 
         Precision: nan
         Recall: 0.0
         F1-Measure: nan
    Label حج و زیارت و وقف: 
         Precision: 0.48314606741573035
         Recall: 1.0
         F1-Measure: 0.6515151515151515
    Label اقتصادی: 
         Precision: 0.853990914990266
         Recall: 0.548790658882402
         F1-Measure: 0.6681898959126681
    Label قرآن و فعالیت های دینی: 
         Precision: 0.4769874476987448
         Recall: 1.0
         F1-Measure: 0.6458923512747875
    Label تشکل های دانشگاهی: 
         Precision: 0.48520710059171596
         Recall: 0.9761904761904762
         F1-Measure: 0.6482213438735178
    Label کتاب و ادبیات: 
         Precision: 0.5018450184501845
         Recall: 0.9645390070921985
         F1-Measure: 0.6601941747572815
    Label رسانه: 
         Precision: 0.5571428571428572
         Recall: 0.8863636363636364
         F1-Measure: 0.6842105263157894
    Label محیط زیست و گردشگری: 
         Precision: 0.5348837209302325
         Recall: 1.0
         F1-Measure: 0.6969696969696969
    Label عمومی: 
         Precision: 0.8790322580645161
         Recall: 0.9316239316239316
         F1-Measure: 0.904564315352697
    Label سلامت: 
         Precision: 0.6308186195826645
         Recall: 0.8308668076109936
         F1-Measure: 0.7171532846715328
    Label دفاتر منطقه ای: 
         Precision: 0.7094594594594594
         Recall: 0.30973451327433627
         F1-Measure: 0.4312114989733059
    Label تعاون و اشتغال: 
         Precision: 0.46511627906976744
         Recall: 0.9302325581395349
         F1-Measure: 0.6201550387596899
    Label کشاورزی و امور دام: 
         Precision: 0.4423076923076923
         Recall: 1.0
         F1-Measure: 0.6133333333333333
    Label ورزشی: 
         Precision: 0.877164056059357
         Recall: 0.30201532784558616
         F1-Measure: 0.4493243243243243
    Label رفاه و آسیب های اجتماعی: 
         Precision: 0.49056603773584906
         Recall: 0.9512195121951219
         F1-Measure: 0.6473029045643154
    Label راه و مسکن: 
         Precision: 0.4273972602739726
         Recall: 0.968944099378882
         F1-Measure: 0.5931558935361217
    Label غرب آسیا: 
         Precision: 0.4952015355086372
         Recall: 0.9608938547486033
         F1-Measure: 0.6535782140595312
    Label صنفی فرهنگی: 
         Precision: 0.46534653465346537
         Recall: 0.9215686274509803
         F1-Measure: 0.618421052631579
    Label انتظامی و حوادث: 
         Precision: 0.4698275862068966
         Recall: 0.9478260869565217
         F1-Measure: 0.6282420749279539
    Label آموزش: 
         Precision: 0.42786069651741293
         Recall: 0.819047619047619
         F1-Measure: 0.5620915032679739
    Label خواندنی ها و دیدنی ها: 
         Precision: 0.94140625
         Recall: 0.9836734693877551
         F1-Measure: 0.9620758483033932
    Label روانشناسی: 
         Precision: 0.7391304347826086
         Recall: 1.0
         F1-Measure: 0.85
    Label موسیقی و هنرهای تجسمی: 
         Precision: 0.49514563106796117
         Recall: 0.9902912621359223
         F1-Measure: 0.6601941747572816
    Label مجله فارس پلاس: 
         Precision: 0.9558823529411765
         Recall: 0.8783783783783784
         F1-Measure: 0.9154929577464789
    Label دیدگاه: 
         Precision: 1.0
         Recall: 1.0
         F1-Measure: 1.0
    Label بین الملل: 
         Precision: 0.9417870036101083
         Recall: 0.5981656635139009
         F1-Measure: 0.7316389132340052
    Label ارتباطات و فن آوری اطلاعات: 
         Precision: 0.4858757062146893
         Recall: 0.9885057471264368
         F1-Measure: 0.6515151515151515
    Label رزمی: 
         Precision: 0.49693251533742333
         Recall: 0.9642857142857143
         F1-Measure: 0.6558704453441296
    Label فرهنگ عمومی: 
         Precision: 0.6666666666666666
         Recall: 1.0
         F1-Measure: 0.8
    Label علمی و دانشگاهی: 
         Precision: 0.5789473684210527
         Recall: 0.2515592515592516
         F1-Measure: 0.35072463768115947
    Label سیاست خارجی: 
         Precision: 0.5814332247557004
         Recall: 0.9248704663212435
         F1-Measure: 0.714
    Label بیمه و بانک: 
         Precision: 0.5
         Recall: 0.9692307692307692
         F1-Measure: 0.6596858638743455
    Label حقوقی و قضایی: 
         Precision: 0.4411764705882353
         Recall: 0.967741935483871
         F1-Measure: 0.6060606060606061
    Label رادیو و تلویزیون: 
         Precision: 0.48562300319488816
         Recall: 0.9806451612903225
         F1-Measure: 0.6495726495726495
    Label امنیتی و دفاعی: 
         Precision: 0.4251968503937008
         Recall: 1.0
         F1-Measure: 0.5966850828729281
    Label پژوهش: 
         Precision: 0.5
         Recall: 0.2
         F1-Measure: 0.28571428571428575
    Label اقتصاد بین الملل: 
         Precision: 0.430622009569378
         Recall: 0.989010989010989
         F1-Measure: 0.6000000000000001
    Label سفر و تفریح: 
         Precision: 1.0
         Recall: 1.0
         F1-Measure: 1.0
    Label سیاسی: 
         Precision: 0.8949720670391061
         Recall: 0.5491943777853959
         F1-Measure: 0.6806883365200764
    Label اطلاعات عمومی و دانستنی ها: 
         Precision: 0.5192307692307693
         Recall: 0.8852459016393442
         F1-Measure: 0.6545454545454545
    Label فارس من: 
         Precision: 1.0
         Recall: 1.0
         F1-Measure: 1.0
    Label آشپزی: 
         Precision: 1.0
         Recall: 1.0
         F1-Measure: 1.0
    Label نفت و انرژی: 
         Precision: 0.4876847290640394
         Recall: 0.9801980198019802
         F1-Measure: 0.6513157894736842
    Label دولت: 
         Precision: 0.45161290322580644
         Recall: 0.8828828828828829
         F1-Measure: 0.5975609756097561
    Label شبهه و شایعه: 
         Precision: 1.0
         Recall: 0.9375
         F1-Measure: 0.967741935483871
    Label فوتبال جهان: 
         Precision: 0.4669631512071156
         Recall: 0.98
         F1-Measure: 0.6325301204819277
    Label کشتی و وزنه برداری: 
         Precision: 0.484375
         Recall: 1.0
         F1-Measure: 0.6526315789473685
    Label علم و فن آوری ایران: 
         Precision: 0.5808823529411765
         Recall: 0.79
         F1-Measure: 0.6694915254237288
    Label اقتصاد کلان و بودجه: 
         Precision: 0.44029850746268656
         Recall: 0.9833333333333333
         F1-Measure: 0.6082474226804124
    Label تاریخ معاصر: 
         Precision: 0.6666666666666666
         Recall: 0.2857142857142857
         F1-Measure: 0.4
    Total Accuracy: 0.6632662759385395
    

    C:\Users\Erfan\AnacondaProjects\github\NLP-Fall18-UOG\utils\models.py:89: RuntimeWarning: invalid value encountered in true_divide
      percision = confusion_matrix.diagonal()/np.sum(confusion_matrix, axis=1)
    

Test Evaluation:


```python
nb.evaluate(x_test, y_test, label_set=label_set)
```

    %0 continue...
    %1000 continue...
    %2000 continue...
    %3000 continue...
    %4000 continue...
    %5000 continue...
    %6000 continue...
    %7000 continue...
    Label مسئولیت های اجتماعی: 
         Precision: 0.5384615384615384
         Recall: 1.0
         F1-Measure: 0.7000000000000001
    Label صنعت ، تجارت ، بازرگانی: 
         Precision: 0.358974358974359
         Recall: 0.7777777777777778
         F1-Measure: 0.49122807017543857
    Label ایران در جهان: 
         Precision: 0.38461538461538464
         Recall: 0.5555555555555556
         F1-Measure: 0.4545454545454546
    Label شهری: 
         Precision: 0.46551724137931033
         Recall: 0.8852459016393442
         F1-Measure: 0.6101694915254237
    Label غرب از نگاه غرب: 
         Precision: 1.0
         Recall: 1.0
         F1-Measure: 1.0
    Label خانواده: 
         Precision: 1.0
         Recall: 0.3333333333333333
         F1-Measure: 0.5
    Label تور و توپ: 
         Precision: 0.4186046511627907
         Recall: 0.6
         F1-Measure: 0.49315068493150693
    Label فوتبال ایران: 
         Precision: 0.4357429718875502
         Recall: 0.7331081081081081
         F1-Measure: 0.5465994962216625
    Label علمی: 
         Precision: 0.16666666666666666
         Recall: 0.09090909090909091
         F1-Measure: 0.11764705882352942
    Label اجتماعی: 
         Precision: 0.7337748344370861
         Recall: 0.4855390008764242
         F1-Measure: 0.5843881856540084
    Label سرگرمی: 
         Precision: 0.15384615384615385
         Recall: 0.9090909090909091
         F1-Measure: 0.26315789473684215
    Label مسجد و هیئت: 
         Precision: 0.5
         Recall: 1.0
         F1-Measure: 0.6666666666666666
    Label فرهنگ و هنر: 
         Precision: 0.13114754098360656
         Recall: 0.03755868544600939
         F1-Measure: 0.058394160583941604
    Label احزاب و تشکل ها: 
         Precision: 0.4262295081967213
         Recall: 0.9629629629629629
         F1-Measure: 0.5909090909090909
    Label پاکستان: 
         Precision: 0.34782608695652173
         Recall: 0.47058823529411764
         F1-Measure: 0.39999999999999997
    Label بورس: 
         Precision: 0.5348837209302325
         Recall: 0.92
         F1-Measure: 0.6764705882352942
    Label گروههای توان خواه: 
         Precision: 0.5555555555555556
         Recall: 0.8333333333333334
         F1-Measure: 0.6666666666666667
    Label بازار: 
         Precision: 0.15
         Recall: 0.1875
         F1-Measure: 0.16666666666666663
    Label حماسه و مقاومت: 
         Precision: 0.0
         Recall: nan
         F1-Measure: nan
    Label خبر خوب: 
         Precision: 1.0
         Recall: 0.6666666666666666
         F1-Measure: 0.8
    Label آفریقا: 
         Precision: 0.42592592592592593
         Recall: 0.7419354838709677
         F1-Measure: 0.5411764705882354
    Label زنان و جوانان: 
         Precision: 0.46153846153846156
         Recall: 0.75
         F1-Measure: 0.5714285714285714
    Label مجلس: 
         Precision: 0.43243243243243246
         Recall: 0.8495575221238938
         F1-Measure: 0.573134328358209
    Label تاریخ: 
         Precision: 0.6666666666666666
         Recall: 0.4
         F1-Measure: 0.5
    Label جنگ اقتصادی: 
         Precision: 0.5
         Recall: 1.0
         F1-Measure: 0.6666666666666666
    Label سینما و تئاتر: 
         Precision: 0.532608695652174
         Recall: 0.8448275862068966
         F1-Measure: 0.6533333333333333
    Label داستان کوتاه: 
         Precision: 1.0
         Recall: 0.3333333333333333
         F1-Measure: 0.5
    Label استانها: 
         Precision: 0.9695121951219512
         Recall: 0.7378190255220418
         F1-Measure: 0.8379446640316206
    Label انقلاب اسلامی: 
         Precision: nan
         Recall: 0.0
         F1-Measure: nan
    Label علم و فن آوری جهان: 
         Precision: 0.5208333333333334
         Recall: 0.7352941176470589
         F1-Measure: 0.6097560975609756
    Label اندیشه: 
         Precision: 0.65
         Recall: 0.7647058823529411
         F1-Measure: 0.7027027027027027
    Label امام و رهبری: 
         Precision: 0.2
         Recall: 1.0
         F1-Measure: 0.33333333333333337
    Label شرق آسیا و اقیانوسیه: 
         Precision: 0.5833333333333334
         Recall: 0.6363636363636364
         F1-Measure: 0.6086956521739131
    Label تحلیل بین الملل: 
         Precision: nan
         Recall: nan
         F1-Measure: nan
    Label آسياي مرکزی و روسيه: 
         Precision: 0.273972602739726
         Recall: 0.4166666666666667
         F1-Measure: 0.33057851239669417
    Label ورزش بانوان: 
         Precision: 0.42857142857142855
         Recall: 0.5
         F1-Measure: 0.4615384615384615
    Label فرهنگی/هنری: 
         Precision: 0.605
         Recall: 0.8461538461538461
         F1-Measure: 0.705539358600583
    Label فناوری و IT: 
         Precision: 0.4878048780487805
         Recall: 0.6451612903225806
         F1-Measure: 0.5555555555555556
    Label حوادث: 
         Precision: 0.3974358974358974
         Recall: 0.7948717948717948
         F1-Measure: 0.5299145299145299
    Label آمریکا، اروپا: 
         Precision: 0.47767857142857145
         Recall: 0.8492063492063492
         F1-Measure: 0.6114285714285714
    Label ویژه نامه ها: 
         Precision: 0.06666666666666667
         Recall: 0.05263157894736842
         F1-Measure: 0.058823529411764705
    Label ورزش بین الملل: 
         Precision: 0.4925373134328358
         Recall: 0.7586206896551724
         F1-Measure: 0.597285067873303
    Label آموزش و پرورش: 
         Precision: 0.5102040816326531
         Recall: 0.9615384615384616
         F1-Measure: 0.6666666666666667
    Label محور مقاومت: 
         Precision: nan
         Recall: 0.0
         F1-Measure: nan
    Label حج و زیارت و وقف: 
         Precision: 0.42105263157894735
         Recall: 1.0
         F1-Measure: 0.5925925925925926
    Label اقتصادی: 
         Precision: 0.6202247191011236
         Recall: 0.46779661016949153
         F1-Measure: 0.5333333333333333
    Label قرآن و فعالیت های دینی: 
         Precision: 0.4067796610169492
         Recall: 1.0
         F1-Measure: 0.5783132530120482
    Label تشکل های دانشگاهی: 
         Precision: 0.46153846153846156
         Recall: 0.75
         F1-Measure: 0.5714285714285714
    Label کتاب و ادبیات: 
         Precision: 0.42857142857142855
         Recall: 0.6976744186046512
         F1-Measure: 0.5309734513274337
    Label رسانه: 
         Precision: 0.2631578947368421
         Recall: 0.5
         F1-Measure: 0.3448275862068966
    Label محیط زیست و گردشگری: 
         Precision: 0.3548387096774194
         Recall: 0.6470588235294118
         F1-Measure: 0.4583333333333333
    Label عمومی: 
         Precision: 0.6153846153846154
         Recall: 0.6153846153846154
         F1-Measure: 0.6153846153846154
    Label سلامت: 
         Precision: 0.5657142857142857
         Recall: 0.7333333333333333
         F1-Measure: 0.6387096774193548
    Label دفاتر منطقه ای: 
         Precision: 0.28846153846153844
         Recall: 0.18072289156626506
         F1-Measure: 0.2222222222222222
    Label تعاون و اشتغال: 
         Precision: 0.4444444444444444
         Recall: 0.7272727272727273
         F1-Measure: 0.5517241379310345
    Label کشاورزی و امور دام: 
         Precision: 0.5
         Recall: 0.7142857142857143
         F1-Measure: 0.588235294117647
    Label ورزشی: 
         Precision: 0.6521739130434783
         Recall: 0.2927669345579793
         F1-Measure: 0.40412044374009504
    Label رفاه و آسیب های اجتماعی: 
         Precision: 0.4878048780487805
         Recall: 0.7142857142857143
         F1-Measure: 0.5797101449275363
    Label راه و مسکن: 
         Precision: 0.35714285714285715
         Recall: 0.7692307692307693
         F1-Measure: 0.48780487804878053
    Label غرب آسیا: 
         Precision: 0.3735408560311284
         Recall: 0.8205128205128205
         F1-Measure: 0.5133689839572193
    Label صنفی فرهنگی: 
         Precision: 0.36
         Recall: 1.0
         F1-Measure: 0.5294117647058824
    Label انتظامی و حوادث: 
         Precision: 0.4375
         Recall: 1.0
         F1-Measure: 0.6086956521739131
    Label آموزش: 
         Precision: 0.2553191489361702
         Recall: 0.5217391304347826
         F1-Measure: 0.3428571428571428
    Label خواندنی ها و دیدنی ها: 
         Precision: 0.6460176991150443
         Recall: 0.6293103448275862
         F1-Measure: 0.6375545851528385
    Label روانشناسی: 
         Precision: 0.3333333333333333
         Recall: 0.2857142857142857
         F1-Measure: 0.30769230769230765
    Label موسیقی و هنرهای تجسمی: 
         Precision: 0.48484848484848486
         Recall: 0.8421052631578947
         F1-Measure: 0.6153846153846154
    Label مجله فارس پلاس: 
         Precision: 1.0
         Recall: 0.5
         F1-Measure: 0.6666666666666666
    Label دیدگاه: 
         Precision: 1.0
         Recall: 1.0
         F1-Measure: 1.0
    Label بین الملل: 
         Precision: 0.8356164383561644
         Recall: 0.5501691093573844
         F1-Measure: 0.6634942216179469
    Label ارتباطات و فن آوری اطلاعات: 
         Precision: 0.40625
         Recall: 0.8125
         F1-Measure: 0.5416666666666666
    Label رزمی: 
         Precision: 0.4318181818181818
         Recall: 0.7916666666666666
         F1-Measure: 0.5588235294117647
    Label فرهنگ عمومی: 
         Precision: 0.0
         Recall: nan
         F1-Measure: nan
    Label علمی و دانشگاهی: 
         Precision: 0.2727272727272727
         Recall: 0.168
         F1-Measure: 0.2079207920792079
    Label سیاست خارجی: 
         Precision: 0.4064516129032258
         Recall: 0.7777777777777778
         F1-Measure: 0.5338983050847458
    Label بیمه و بانک: 
         Precision: 0.43243243243243246
         Recall: 0.7272727272727273
         F1-Measure: 0.5423728813559323
    Label حقوقی و قضایی: 
         Precision: 0.55
         Recall: 0.9166666666666666
         F1-Measure: 0.6874999999999999
    Label رادیو و تلویزیون: 
         Precision: 0.4
         Recall: 0.7222222222222222
         F1-Measure: 0.5148514851485149
    Label امنیتی و دفاعی: 
         Precision: 0.3137254901960784
         Recall: 0.8888888888888888
         F1-Measure: 0.46376811594202894
    Label پژوهش: 
         Precision: nan
         Recall: 0.0
         F1-Measure: nan
    Label اقتصاد بین الملل: 
         Precision: 0.5544554455445545
         Recall: 0.9032258064516129
         F1-Measure: 0.6871165644171778
    Label سفر و تفریح: 
         Precision: nan
         Recall: nan
         F1-Measure: nan
    Label سیاسی: 
         Precision: 0.7044967880085653
         Recall: 0.4380825565912117
         F1-Measure: 0.5402298850574713
    Label اطلاعات عمومی و دانستنی ها: 
         Precision: 0.4
         Recall: 0.5263157894736842
         F1-Measure: 0.45454545454545453
    Label فارس من: 
         Precision: 1.0
         Recall: 1.0
         F1-Measure: 1.0
    Label آشپزی: 
         Precision: 0.0
         Recall: 0.0
         F1-Measure: nan
    Label نفت و انرژی: 
         Precision: 0.38461538461538464
         Recall: 0.7894736842105263
         F1-Measure: 0.5172413793103449
    Label دولت: 
         Precision: 0.5357142857142857
         Recall: 0.8571428571428571
         F1-Measure: 0.6593406593406593
    Label شبهه و شایعه: 
         Precision: 1.0
         Recall: 0.4
         F1-Measure: 0.5714285714285715
    Label فوتبال جهان: 
         Precision: 0.44086021505376344
         Recall: 0.8864864864864865
         F1-Measure: 0.5888689407540395
    Label کشتی و وزنه برداری: 
         Precision: 0.5294117647058824
         Recall: 0.8181818181818182
         F1-Measure: 0.6428571428571428
    Label علم و فن آوری ایران: 
         Precision: 0.10714285714285714
         Recall: 0.16666666666666666
         F1-Measure: 0.13043478260869565
    Label اقتصاد کلان و بودجه: 
         Precision: 0.35714285714285715
         Recall: 0.7142857142857143
         F1-Measure: 0.4761904761904762
    Label تاریخ معاصر: 
         Precision: 0.0
         Recall: 0.0
         F1-Measure: nan
    Total Accuracy: 0.5556259503294475
    

    C:\Users\Erfan\AnacondaProjects\github\NLP-Fall18-UOG\utils\models.py:89: RuntimeWarning: invalid value encountered in true_divide
      percision = confusion_matrix.diagonal()/np.sum(confusion_matrix, axis=1)
    C:\Users\Erfan\AnacondaProjects\github\NLP-Fall18-UOG\utils\models.py:90: RuntimeWarning: invalid value encountered in true_divide
      recall = confusion_matrix.diagonal()/np.sum(confusion_matrix, axis=0)
    C:\Users\Erfan\AnacondaProjects\github\NLP-Fall18-UOG\utils\models.py:91: RuntimeWarning: invalid value encountered in true_divide
      f1_measure = 2*percision*recall/(percision+recall)
    

## Creating Model for the Second Task


```python
t = []
for i, raw_label in enumerate(raw_labels):
    l = []
    for j, label in enumerate(raw_label):
        l.append(np.argmax(label == label_set))
    t.append(l)
```


```python
nb.evaluate(documents, t, label_set, eval_type='multiple')
```

    %0 continue...
    %1000 continue...
    %2000 continue...
    %3000 continue...
    %4000 continue...
    %5000 continue...
    %6000 continue...
    %7000 continue...
    %8000 continue...
    %9000 continue...
    %10000 continue...
    %11000 continue...
    %12000 continue...
    %13000 continue...
    %14000 continue...
    %15000 continue...
    %16000 continue...
    %17000 continue...
    %18000 continue...
    %19000 continue...
    %20000 continue...
    %21000 continue...
    %22000 continue...
    %23000 continue...
    %24000 continue...
    %25000 continue...
    %26000 continue...
    %27000 continue...
    %28000 continue...
    %29000 continue...
    Total Score: -88182
    

## Creating Model for the Third Task

Training:


```python
nb_th = NaiveBayes()
nb_th.fit(x_train_th, y_train_th)
```

    Vocab created
    P(c) calculated
    2
    %0.0 continue...
    %0.5 continue...
    P(w|c) calculated
    

Train Evaluation:


```python
nb_th.evaluate(x_train_th, y_train_th, label_set_th)
```

    %0 continue...
    %1000 continue...
    %2000 continue...
    %3000 continue...
    %4000 continue...
    %5000 continue...
    %6000 continue...
    %7000 continue...
    %8000 continue...
    %9000 continue...
    %10000 continue...
    %11000 continue...
    %12000 continue...
    %13000 continue...
    %14000 continue...
    %15000 continue...
    %16000 continue...
    %17000 continue...
    %18000 continue...
    %19000 continue...
    %20000 continue...
    %21000 continue...
    %22000 continue...
    %23000 continue...
    Label AsrIran: 
         Precision: 0.965974765974766
         Recall: 0.9862865691489362
         F1-Measure: 0.9760250030842621
    Label Fars: 
         Precision: 0.9853072128227961
         Recall: 0.9635983627971785
         F1-Measure: 0.9743318804209042
    Total Accuracy: 0.9752073144801191
    

Test Evaluation:


```python
nb_th.evaluate(x_test_th, y_test_th, label_set_th)
```

    %0 continue...
    %1000 continue...
    %2000 continue...
    %3000 continue...
    %4000 continue...
    %5000 continue...
    Label AsrIran: 
         Precision: 0.9382040553588671
         Recall: 0.9821428571428571
         F1-Measure: 0.959670781893004
    Label Fars: 
         Precision: 0.9808802308802309
         Recall: 0.9340432840948127
         F1-Measure: 0.9568889670948443
    Total Accuracy: 0.9583262459601973
    

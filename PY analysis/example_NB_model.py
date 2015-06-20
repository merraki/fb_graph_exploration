import collections
import nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB


stopset = stopwords.words('english')

negids = movie_reviews.fileids('neg')
posids = movie_reviews.fileids('pos')

negdocs = [movie_reviews.raw(f) for f in negids]
posdocs = [movie_reviews.raw(f) for f in posids]
negtags=[0]*len(negdocs)
postags=[1]*len(posdocs)
                              
negcutoff = int(len(negdocs)*0.8)
poscutoff = int(len(posdocs)*0.8)

traindocs = negdocs[:negcutoff] + posdocs[:poscutoff]
traintags = negtags[:negcutoff] + postags[:poscutoff]
testdocs = negdocs[negcutoff:] + posdocs[poscutoff:]
testtags = negtags[negcutoff:] + postags[negcutoff:]
print 'train on %d instances, test on %d instances' % (len(traindocs), len(testdocs))


vectorizer = CountVectorizer(min_df=1, binary=True, stop_words=stopset)
trainfeats = vectorizer.fit_transform(traindocs)
clf = MultinomialNB()
clf.fit(trainfeats, traintags)

testfeats=vectorizer.transform(testdocs)
predicted = clf.predict(testfeats)
print len(predicted)
print testtags
print str(np.mean(predicted == testtags))

from sklearn import metrics
print(metrics.classification_report(testtags, predicted,target_names=['neg','pos']))
#Y = vectorizer.fit_transform(testfeats)
#Y_array= Y.toarray()
#clf.score(np.array(testfeats),np.array(testtags))
#print (nltk.classify.accuracy(classifier, testfeats))
#print classifier.show_most_informative_features(5)
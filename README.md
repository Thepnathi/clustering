# clustering

## Dataset
animals contains 50 features 
countries contains 161 features
fruits contains 58 features
veggies contains 60 features
In total, the whole dataset contains 329 objects/features.

## Refactor
* Cluster objects -> instead of list that represent the objects, that points to ith representative. Create a 2-d list with objects belonging to each cluster. Easier to merge later
* Rename convention -> category to label or class
* Rename convention -> representative to cluster or cluster centroid

## Bugs

{'animals': 0, 'countries': 80, 'fruits': 0, 'veggies': 0}
{'animals': 0, 'countries': 54, 'fruits': 0, 'veggies': 0}
{'animals': 50, 'countries': 0, 'fruits': 59, 'veggies': 61}
{'animals': 0, 'countries': 27, 'fruits': 0, 'veggies': 0}
Some reason there is one extra fruits and veggies
List index out of range during the computation of median representative.

Some improvement is that we can use a smarter way to pick cluster representatives
For example select an object from each label or object type.

## Ploting with multidimension data
http://www.apnorton.com/blog/2016/12/19/Visualizing-Multidimensional-Data-in-Python/

https://stackoverflow.com/questions/32276391/feature-normalization-advantage-of-l2-normalization

https://www.kaggle.com/residentmario/l1-norms-versus-l2-norms

https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.5-Norms/

## References 

A Course in Machine Learning by Hal Daumé III, 2007
http://ciml.info/

Data Mining The Textbook by Charu C. Aggarwal

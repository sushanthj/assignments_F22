# Hyperparameter Tuning

The hyperparams to be tuned included the following:
- filter scales: a list of filter scales used in extracting filter response
- K: the number of visual words and also the size of the dictionary
- alpha: the number of sampled pixels in each image when creating the dictionary
- L: the number of spatial pyramid layers used in feature extraction

## Tuning Alpha Values

### Using Random Crops Instead of Pixels

- Each train image was cropped randomly (with crop size = alpha x alpha)
- While a small training set may have reduced the randomization of pixels which would feed to KMeans, having over 1000 training images seemed to create a good enough KMeans model
- The accuracy observed with K = 10, L = 2, alpha = 25*25 was 56.25%

Confusion matrix given below:

[[32.  1.  1.  1.  3.  2.  5.  5.]
 [ 0. 25.  8.  6.  3.  1.  2.  5.]
 [ 1.  9. 29.  0.  1.  2.  2.  6.]
 [ 1.  3.  1. 33.  9.  0.  0.  3.]
 [ 2.  3.  2. 10. 25.  3.  2.  3.]
 [ 3.  0.  9.  3.  2. 29.  4.  0.]
 [ 5.  1.  2.  2.  6.  7. 25.  2.]
 [ 2.  6.  5.  0.  3.  6.  1. 27.]]

### Using alpha random pixels in each image

- The hyper--parameters mentioned in the above sectioned were maintained to compare the two methods of getting randomized pixels
- An accuracy observed with K = 10, L = 2, alpha = 625 was 54.75%

Confusion matrix given below:
confusion matrix is 
 [[29.  1.  1.  1.  2.  1. 10.  5.]
 [ 0. 32.  5.  3.  5.  0.  2.  3.]
 [ 1.  7. 29.  0.  0.  1.  3.  9.]
 [ 4.  4.  1. 30.  8.  2.  0.  1.]
 [ 2.  5.  1. 13. 20.  3.  4.  2.]
 [ 3.  1.  4.  0.  3. 31.  4.  4.]
 [ 4.  0.  3.  0.  7.  9. 25.  2.]
 [ 2.  8. 10.  0.  1.  3.  3. 23.]]

- An accuracy observed with K = 10, L = 2, alpha = 10 was 57.5%

 confusion matrix is 
 [[32.  1.  4.  1.  2.  1.  3.  6.]
 [ 0. 22.  5.  8.  4.  2.  2.  7.]
 [ 1.  5. 30.  0.  0.  4.  1.  9.]
 [ 1.  4.  2. 35.  7.  0.  1.  0.]
 [ 2.  2.  2. 10. 29.  1.  1.  3.]
 [ 4.  0.  8.  2.  1. 28.  5.  2.]
 [ 3.  1.  1.  3.  5.  8. 27.  2.]
 [ 2.  5.  7.  0.  1.  7.  1. 27.]]
accuracy is 0.575

- An accuracy observed with K = 10, L = 2, alpha = 30 was 59.25%

confusion matrix is 
 [[30.  0.  3.  0.  4.  2.  5.  6.]
 [ 1. 32.  6.  3.  2.  0.  2.  4.]
 [ 2.  6. 33.  0.  0.  3.  1.  5.]
 [ 1.  1.  2. 30. 13.  0.  2.  1.]
 [ 1.  1.  2. 17. 21.  4.  2.  2.]
 [ 4.  0.  4.  2.  2. 32.  5.  1.]
 [ 4.  0.  2.  0.  6.  5. 30.  3.]
 [ 1.  5.  6.  1.  1.  5.  2. 29.]]
accuracy is 0.5925



## Tuning L values

- The higher the L value, the more layers that will be formed in the spatial pyramid. However, since the finer layers (with more cells) get weighted more, the model becomes more sensitive to spatial features.

- L = 1 gave the following results

confusion matrix is 
 [[31.  0.  1.  1.  6.  3.  2.  6.]
 [ 0. 29.  3.  6.  2.  2.  3.  5.]
 [ 1.  5. 35.  1.  0.  1.  2.  5.]
 [ 1.  3.  2. 34.  9.  1.  0.  0.]
 [ 4.  2.  2. 17. 16.  5.  4.  0.]
 [ 1.  0.  5.  2.  4. 29.  6.  3.]
 [ 8.  1.  1.  0.  4.  6. 28.  2.]
 [ 1.  4.  7.  2.  1.  5.  6. 24.]]
accuracy is 0.565

- L = 3 gave the following results:

confusion matrix is 
 [[31.  0.  5.  0.  2.  1.  5.  6.]
 [ 1. 35.  9.  0.  1.  0.  2.  2.]
 [ 2.  6. 31.  0.  0.  2.  1.  8.]
 [ 2.  2.  2. 34.  9.  0.  1.  0.]
 [ 0.  2.  6. 12. 19.  3.  2.  6.]
 [ 4.  0.  8.  1.  2. 30.  3.  2.]
 [ 3.  1.  1.  0.  5.  9. 29.  2.]
 [ 1.  5.  7.  1.  1.  5.  0. 30.]]
accuracy is 0.5975

- L = 4 gave the following results

confusion matrix is 
 [[32.  0.  4.  0.  2.  3.  2.  7.]
 [ 1. 39.  6.  0.  0.  0.  2.  2.]
 [ 2. 10. 29.  0.  0.  0.  1.  8.]
 [ 2. 10.  2. 25.  9.  0.  0.  2.]
 [ 0.  6.  6.  7. 19.  4.  2.  6.]
 [ 4.  3.  7.  1.  1. 31.  2.  1.]
 [ 4.  2.  2.  1.  5. 11. 24.  1.]
 [ 1.  7.  7.  1.  1.  5.  0. 28.]]
accuracy is 0.5675

-------------------------------
- An accuracy observed with K = 30, L = 3, alpha = 30 was 63.5%
confusion matrix is 
 [[39.  0.  0.  1.  5.  1.  1.  3.]
 [ 1. 33.  4.  4.  1.  0.  1.  6.]
 [ 1.  4. 28.  0.  2.  2.  1. 12.]
 [ 1.  1.  0. 34. 11.  1.  0.  2.]
 [ 0.  2.  1.  9. 29.  5.  4.  0.]
 [ 3.  0.  5.  1.  2. 35.  0.  4.]
 [ 4.  0.  2.  0.  7.  7. 26.  4.]
 [ 0.  2. 10.  0.  0.  6.  2. 30.]]
accuracy is 0.635

- An accuracy observed with K = 30, L = 2, alpha = 30 was 64.5%
confusion matrix is 
 [[38.  0.  2.  2.  2.  1.  1.  4.]
 [ 0. 33.  3.  7.  3.  1.  2.  1.]
 [ 0.  5. 33.  0.  0.  4.  3.  5.]
 [ 1.  1.  0. 35. 10.  0.  3.  0.]
 [ 1.  1.  1. 14. 26.  4.  3.  0.]
 [ 2.  0.  6.  1.  1. 36.  1.  3.]
 [ 5.  0.  1.  1.  4.  8. 28.  3.]
 [ 3.  3. 10.  0.  1.  3.  1. 29.]]
accuracy is 0.645



- **Therefore L = 3 was the best compromise between being too specific on spatial features**

## Tuning K values

K = 30 with randomized pixel crops of alpha = 25 (25*25 crops) and L = 3 yields:
confusion matrix is 
 [[40.  0.  1.  1.  3.  1.  1.  3.]
 [ 1. 35.  3.  5.  0.  0.  2.  4.]
 [ 2.  4. 31.  0.  0.  2.  2.  9.]
 [ 1.  0.  0. 33. 10.  1.  2.  3.]
 [ 0.  1.  3.  9. 31.  2.  4.  0.]
 [ 2.  0.  6.  1.  1. 34.  3.  3.]
 [ 6.  0.  3.  1.  4.  7. 26.  3.]
 [ 1.  3.  8.  0.  0.  4.  1. 33.]]
accuracy is 0.6575

K = 30 with randomized pixels of 30 and L = 3 yields:
- An accuracy observed with K = 30, L = 3, alpha = 30 was 63.5%
confusion matrix is 
 [[39.  0.  0.  1.  5.  1.  1.  3.]
 [ 1. 33.  4.  4.  1.  0.  1.  6.]
 [ 1.  4. 28.  0.  2.  2.  1. 12.]
 [ 1.  1.  0. 34. 11.  1.  0.  2.]
 [ 0.  2.  1.  9. 29.  5.  4.  0.]
 [ 3.  0.  5.  1.  2. 35.  0.  4.]
 [ 4.  0.  2.  0.  7.  7. 26.  4.]
 [ 0.  2. 10.  0.  0.  6.  2. 30.]]
accuracy is 0.635
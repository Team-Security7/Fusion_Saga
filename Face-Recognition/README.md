# Face-Recognition

<h2>Instructions for making face Dataset</h2>
<ol>
  <li>Train Unmasked : 400</li>
  <li>Val Unmasked : 200</li>
  <li>Masked : 100</li>
</ol>


### Data preparation

- There is no dataset available in the public domain with subjects wearing masks and its corresponding unmasked faces since such unique challenge was not posed earlier thanks to the absence of a pandemic. 
- Our research group has made a conscious effort and built a database of subjects with and without face masks. We refer to this face database as Masked Face database. 
- Few representative faces are below figure where the top row corresponds to unmasked faces which constitutes the gallery set while its corresponding masked faces for the same subjects are shown below which constitutes the probe set. 
![MaskedVUnmasked](https://user-images.githubusercontent.com/56304060/167352262-18e46e40-688b-4e10-bc42-22d13bc0db33.png)

- The faces of the subjects has been captured using a ` Caffe Based single shot face-detector model with ResNet-10` as its base architecture keeping in consideration some minor variation in light and facial expressions. Below figure shows the working of the face-detector.

![Untitled drawing](https://user-images.githubusercontent.com/56304060/167352399-a8a28e02-10df-4541-8798-391f38a6198b.png)


- The train and validation distributions comprises of the subjects unmasked faces while the test distributions comprise of masked faces. 

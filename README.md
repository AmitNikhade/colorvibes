
# ColorVibes

A deep learning application that converts the Black & white images to colored ones, which brings life to the image.




## Authors

- [@amitnikhade](https://amitnikhade.com/)

  
## Badges

Add badges from somewhere like: [shields.io](https://shields.io/)

[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)

## Usage/Installation

Training the model with custom data
```bash
  #cloning the repository
  >> git clone https://github.com/AmitNikhade/colorvibes.git

  #installing dependencies
  >> pip install requirements.txt

  #training
  >> python train.py -data /path-to-train-data/

  
```

  To use the web app you just need to browse the image from your local machine and press the predict button, you'll get teh coloured image within some seconds.

## Deployment
[Link](https://colorvibes.herokuapp.com/)

Basically the project is deployed on Heroku. It was pretty much easier for me to deploy my model on Heroku. Even I have to face certain problem related to versioning of deep learning frameworks. 

![Logo](https://coursework.vschool.io/content/images/size/w2000/2017/12/Image-result-for-heroku-photo-banner.png)


## AutoEncoders

Autoencoders played a dominant role in building the system. I wasn't having a good GPU so I tried training it on kaggle notebooks. It took me a lot of time to train the model with lots of variations of data. The data needs to me picked properly for the model to train, otherwise it won't give you the desired results.  

Here is the Autoencoder Architecture.


![Image](https://github.com/AmitNikhade/colorvibes/blob/master/Autoencoder.png?raw=true)



## Working

Autoencoders (AE) are type of artificial neural network. They task is to compress the input image into a latent-space representation also known as bottleneck, and then reconstructing the output image from this representation. Autoencoder is an unsupervised machine learning algorithm.It can also be called as a feature extraction algorithm.
Applications of Autoencoders includes Data denoising and Dimensionality reduction for data visualization.

Encoder : This part of the network that compresses the input image data into a latent-space representation.
Decoder : This part of network that reconstructs the latent space representation back to original dimension. Which is not exactly the same but a lossy reconstruction of the original Image.

I have used tensorflow and Keras python libraries to build the code.

![Image](https://miro.medium.com/max/1000/1*LkKz4wtZNBo5i-Vc8DWhTA.png)

## Screenshots

![App Screenshot](https://github.com/AmitNikhade/colorvibes/blob/master/Screenshot-20210813132428-1444x976.png?raw=true)



  
## Support

For support, email amitnikhade@outlook.com also visit https://amitnikhade.com.

  

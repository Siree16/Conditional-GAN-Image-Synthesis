# Conditional GAN Image Synthesis

## Project Overview
This project aims to develop a Conditional Generative Adversarial Network (cGAN) model capable of generating realistic images from input sketches conditioned on class labels. The model has been trained and tested using the ISIC HAM10000 dataset, which includes images from seven classes: MEL, NV, BCC, AKIEC, BKL, DF, and VASC.



## Problem Statement
The project aims to create a machine learning model that transforms grayscale sketches, representing basic shapes and forms, into colorful, detailed images that reflect intended characteristics defined by class labels.

## Dataset
The dataset used is the ISIC HAM10000, which contains 10015 images and corresponding paired sketches. It is divided into:
- Train: 9014 images
- Test: 1001 images
- 7 classes: MEL, NV, BCC, AKIEC, BKL, DF, VASC

## Methodology
The cGAN model consists of two main components:
1. **Generator:** Takes a sketch and class labels as input to generate images.
2. **Discriminator:** Differentiates between real and generated images.

### Generator Architecture
- **U-Net Architecture:** Helps in better preservation of details in the generated images through high-resolution features.
- **PatchGAN Architecture:** Classifies whether each patch in an image is real or fake, capturing high-frequency details.

### Discriminator Architecture
- **PatchGAN Architecture:** Focuses on different parts of the image, providing a matrix of outputs representing patches of the image.

## Training Process
1. **Generator Training:**
   - Generates images from sketches and labels.
   - Uses BCE for loss calculation and Adam optimizer.
   - L1 loss encourages the generator to create images close to the real ones.

2. **Discriminator Training:**
   - Trained with real and generated images to classify them as real or fake.
   - Aims to reach a balanced equilibrium with a loss of approximately 0.5.

3. **Adversarial Loss Calculation:**
   - Discriminator's adversarial loss uses BCE.
   - Generator's adversarial loss encourages realistic outputs.

4. **Backpropagation and Weight Update:**
   - Uses Adam optimizer with a learning rate of 0.0001 and beta parameter of 0.5.

5. **Equilibrium Seeking:**
   - Ensures a balance between generator and discriminator.
   - Periodic evaluation for quality improvement.

## Results
- **Loss Metrics:** Indicate balanced training with improved image quality over epochs.
- **Image Generation:** Shows epoch-by-epoch improvement in detail and color accuracy.
- **Evaluation Metrics:**
  - **Frechet Inception Distance (FID):** Lower score indicates close resemblance to real images.
  - **Inception Score (IS):** Higher score suggests greater diversity and clarity.

## Observations and Conclusions
- cGAN demonstrated potential in generating realistic images from sketches.
- Challenges included maintaining equilibrium and preventing discriminator overpowering.
- Future work could explore more complex datasets and architectures for further improvements.

## References
- [GeeksforGeeks - Conditional Generative Adversarial Network](https://www.geeksforgeeks.org/conditional-generative-adversarial-network/)
- [Machine Learning Mastery - Developing cGAN from Scratch](https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch/)
- [Google Colab - cGAN Tutorial](https://colab.research.google.com/github/tensorflow/gan/blob/master/tensorflow_gan/examples/colab_notebooks/tfgan_tutorial.ipynb)
- [PyTorch - DCGAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Dr. Rajendra Nagar for his guidance.
- ISIC HAM10000 dataset providers.

## Links
- [Project Code](https://github.com/bhaveshkhatri81/Computer-Vision-Project-)
- [Project Outputs](https://drive.google.com/drive/folders/1T5eUQBFuY71mDw6Q9gPolfRhxmexVMUi?usp=sharing)

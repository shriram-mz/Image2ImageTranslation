# Sim2Real ModelZoo

This repository provides a framework to run inferencee for various image to image translation models. <br>
Done by MinusZero! <br>

# Getting Started
### Dependencies
- Install python3 and pytorch.
- install other requirements using 

```
pip install -r requirements.txt
```

### Dataset
We use the RGB Camera images from CARLA Simulator for testing the models. If you wish to train/fine-tune the models, you can download the GTA-Cityscape dataset below!

Download the dataset by running 
```
bash ./get_dataset.sh
```
This will place the dataset in a foder named 'dataset' in the root directory.

# Pretrained Models
Download pretrained models using
``` 
bash ./get_pretrained_models.sh
```
This will create a folder named saved_models and will download pre-trained models which will be used by the test and demo notebooks.
The models part of this folder will be : CycleGAN unsupervised and semisupervised, and DualGAN unsupervised and semisupervised.

# Description
```
cycle_gan.py : Contains the class which implements the CycleGAN architecture.
data_loader.py : Contains the DataLoader class which has utility functions to load and see our dataset. Also has functions to save and display images.
demo.ipynb : Ipython notebook to run a demo with our pre-trained models and display results.
dual_gans.py : Contains the class which implements DualGAN architecture.
logger.py : Contains utility functions to display and format logs.
networks.py : Contains implementations of all building blocks used by our GAN's. Has implementations of different Generator and Discriminator architectures.
params.yaml : Contains hypermarameters used by loss functions and optimizers.
semantics_test.ipynb : Ipython notebook to test the semantic segmentation model using which we compare the results of our models.
test_cycle_gan.py : Test CycleGAN model using pre-trained models.
test_dual_gans.py : Test DualGAN model using pre-trained models.
train.ipynb : Ipython notebook to replicate the training.
train_cycle_gan.py : train our CycleGAN model.
train_dual_gans.py : train our DualGAN model.
train_test.p : Pickle file containing the train-test indexes of images.
utils.py : Utility functions used across different files.
```

# Running the ModelZoo
To run the ModelZoo, use the following command :
```
python model_zoo.py --model cycle_gan_un --data_root /path/to/your/images --image_size 512 --output_path /path/to/save/images
```
--model : Enter the model here, the models available are : dual_gans_un, dual_gans_semi, cycle_gan_un, cycle_gan_semi, cycle_gan_512
--image_size : The default size is 256

Inside model_zoo.py, one can find the ModelZoo class whose object is initialized in the main function. The load model function takes us to the ModelLoader class inside model_loader.py where the models are initalized and loaded. The infer() in the ModelZoo calls the necessary methods to get the final output and save it to the desired directory.  

# Results
- Sample Output
<img src="output.png">

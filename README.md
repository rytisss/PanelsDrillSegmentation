# Drilled holes segmentation in the furniture panels 📸 -> 🕳️

Article '***Segmentation of Drilled Holes in Texture Wooden Furniture Panels Using Deep Neural Network***' by ***Rytis Augustauskas***, ***Arūnas Lipnickas*** and ***Tadas Surgailis*** is printed in ***MDPI Sensors*** journal.

Research paper link: [https://www.mdpi.com/1424-8220/21/11/3633](https://www.mdpi.com/1424-8220/21/11/3633)

<img src="https://github.com/rytisss/PanelsDrillSegmentation/blob/main/res/hole_segmentation_preview.gif" width="1000"/>

# Rendered videos :vhs: comparisson list [here](https://www.youtube.com/watch?v=gaAVMjaxfc4&list=PL5dj7GxMk-6x0BqM7zSg5lopu1lOHPwNl&index=1&t=264s)

# Code usage  

Implementation made in Tensorflow 2.5.0

To train model use **train.py** script. Different convolutional neural network architectures can made with parameters.  
  
**use_se** - ***Squeeze and excitation blocks***  
**use_aspp** - ***Atrous spatial pyramid pooling***  
**use_residual_connetions** - ***Residual blocks/residual connections***  
**use_coord_conv** - ***CoordConv layer***  
**downscale_times** - ***How many time we want to downscale? More downscales = more convolutions or 1 downscale = 2 x Conv2D layers**  

```
    model = unet_autoencoder(filters_in_input=16,
                             input_size=(image_width, image_width, image_channels),
                             loss_function=Loss.CROSSENTROPY50DICE50,
                             learning_rate=1e-3,
                             use_se=True,
                             use_aspp=True,
                             use_coord_conv=True,
                             use_residual_connections=True,
                             downscale_times=4,
                             leaky_relu_alpha=0.1)
```

Use **predict.py** to test or perform prediction. Model can be constructed is the same way as shown above. Pass weights files path to the neural network:  
  
**pretrained_weights** - weights path ('*.hdf5' file)
```
    model = unet_autoencoder(filters_in_input=16,
                             input_size=(image_width, image_width, image_channels),
                             loss_function=Loss.CROSSENTROPY50DICE50,
                             learning_rate=1e-3,
                             use_se=True,
                             use_aspp=True,
                             use_coord_conv=True,
                             use_residual_connections=True,
                             downscale_times=4,
                             leaky_relu_alpha=0.1,
                             pretrained_weights=weight_path)
```

# If you find code useful, consider citing the following research:  
  
***Augustauskas, R.; Lipnickas, A.; Surgailis, T. Segmentation of Drilled Holes in Texture Wooden Furniture Panels Using Deep Neural Network. Sensors 2021, 21, 3633. https://doi.org/10.3390/s21113633***
  
***Augustauskas, R.; Lipnickas, A. Improved Pixel-Level Pavement-Defect Segmentation Using a Deep Autoencoder. Sensors 2020, 20, 2557. https://doi.org/10.3390/s20092557***

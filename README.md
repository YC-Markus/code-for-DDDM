# Code for DDDM

This is the code for paper *A Dual-domain Diffusion Model for Sparse-view CT Reconstruction*.

The pre-processing code and MATLAB fan2parallel code will be released later. 

## Installation SUM

Clone this repository. 

Navigate to SUM folder in your terminal.

Then run:

```
pip install -e .
```

This should install the `improved_diffusion` python package that the scripts depend on.

Note that some packages need to be installed manually.

Note that the SUM and IRM are now trained and inferenced **separately**. Thus, each time you change the working module, please  run  `pip install -e .` **again** under corresponding file path.



## Preparing Data for SUM

Regardless of the type of CT data, appropriate preprocessing is required. SUM requires specially "normalized" parallel-beam sinograms as input (Given the high dynamic range of CT data, we set the "black" area of the sinogram to -1, and let the mean of all sinograms for the same organ to linearly vary to 0, allowing the maximum to drift. Please see xxx for details).  Also, you can use fan-beam sinograms as input, with appropriate change in  `reco_train.py` to meet the demand of FBP.



## Training SUM

The default hyperparameters are appropriately set with adequate testing. However, some hyperparameters related to U-Net may need to be changed in line with your dataset and task.



## Sampling SUM

The above training script saves checkpoints to `.pt` files in the logging directory. These checkpoints will have names like `ema_0.9999_200000.pt` and `model200000.pt`. You will likely want to sample from the EMA models, since those produce much better samples.

Once you have a path to your model, you can generate a single batch of sample like so:

```
python SUM/scripts/super_res_sample.py
```

Or generate a large batch of samples like so:

```
python SUM/scripts/multi_super_res.py
```

Note that for SUM, the sampling strategy is exclusive.



## Installation IRM

Clone this repository. 

Navigate to IRM folder in your terminal.

Then run:

```
pip install -e .
```

This should install the `improved_diffusion` python package that the scripts depend on.

Note that some packages need to be installed manually.

Note that the SUM and IRM are now trained and inferenced separately. Thus, each time you change the working module, please  run  `pip install -e .` again under corresponding file path.



## Preparing Data for IRM

Once the training process of SUM has completed, all the training & test & validation dataset should be sampled, transferred by FBP as the input of IRM. 

Note that the CT image data should also be “normalized” followed by the identical strategy as that in SUM.



## Training IRM

The default hyperparameters are appropriately set with adequate testing. However, some hyperparameters related to U-Net may need to be changed in line with your dataset and task.



## Sampling IRM

The above training script saves checkpoints to `.pt` files in the logging directory. These checkpoints will have names like `ema_0.9999_200000.pt` and `model200000.pt`. You will likely want to sample from the EMA models, since those produce much better samples.

Once you have a path to your model, you can generate a single batch of sample like so:

```
python IRM/scripts/img_super_res_sample.py
```

Or generate a large batch of samples like so:

```
python IRM/scripts/img_multi_sample.py
```

Note that for IRM, strides IDDPM and DDIM are optional. We recommend to use DDIM and set parameter  `--timestep_respacing` to 2 as more timesteps may lead to unexpected faked details.

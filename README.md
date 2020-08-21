# Multi-source Human Image Generation
Implementation of Attention-based Fusion for Multi-source Human Image Generation, S. Lathuilière, E. Sangineto, A. Siarohin, N. Sebe, WACV 2020


### Requirments
The code requires the same libraries as in [Deformable GAN](https://github.com/AliaksandrSiarohin/pose-gan)

### Training

First, you need to download the datasets. To do so, follow the instructions of the [Deformable GAN repo](https://github.com/AliaksandrSiarohin/pose-gan).

Then you need to create the training and test tuples:
```nbinput=12 && python create_n_tuples_dataset.py --nb_inputs nbinput```

Note that, with this command we generate tuples as if we were considering 12 input images. In this step, we recommend to keep ```nbinput=12```, since, in the training step, only the first images of the 12-tuples will be considered if you specify fewer inputs. 

Then you can train your model with the following command:
```nbinput=3 && CUDA_VISIBLE_DEVICES=1 python train.py --output_dir output/model_$nbinput --checkpoints_dir output/model_$nbinput --warp_skip mask --dataset market --nb_inputs $nbinput  --l1_penalty_weight 0.001 --nn_loss_area_size 3 --batch_size 4 --content_loss_layer block1_conv2 --number_of_epochs 12 --dmax 6 --kernel_size_last=1 --fusion_type att_dec_rec --return_att 1  --gan_penalty_weight 0.1 ```

The details of all the options are provided in the file ```cmd.py```

### Test

Simply run the following command with the correct path to the generator:
```nbinput=3 && CUDA_VISIBLE_DEVICES=1 python test.py --warp_skip mask --dataset market --nb_inputs $nbinput  --dmax 6 --kernel_size_last=1 --fusion_type att_dec_rec --return_att 1 -generator_checkpoint path/to/generator/checkpoint```

### For help
This repo is based on the [Deformable GAN repo](https://github.com/AliaksandrSiarohin/pose-gan). You may find some help in this repo. If you don't, you can contact me on my telecom-paris.fr address (check my [website](stelat.eu)). 

import argparse

def args():
    """
        Define args that is used in project
    """
    parser = argparse.ArgumentParser(description="Pose guided image generation usign deformable skip layers")
    parser.add_argument("--output_dir", default='output/displayed_samples', help="Directory with generated sample images")
    parser.add_argument("--batch_size", default=4, type=int, help='Size of the batch')
    parser.add_argument("--training_ratio", default=1, type=int,
                        help="The training ratio is the number of discriminator updates per generator update.")

    parser.add_argument("--train_dec_only", default=0, type=int,
                        help="Train only the decoder")
    parser.add_argument("--l1_penalty_weight", default=100, type=float, help='Weight of l1 loss')
    parser.add_argument('--gan_penalty_weight', default=1, type=float, help='Weight of GAN loss')
    parser.add_argument('--tv_penalty_weight', default=0, type=float, help='Weight of total variation loss')
    parser.add_argument('--smooth_penalty_weight', default=0, type=float, help='Weight of smoothness loss')
    parser.add_argument('--grad_penalty_weight', default=0, type=float, help='Weight of gradness loss')

    parser.add_argument('--lstruct_penalty_weight', default=0, type=float, help="Weight of lstruct")
    
    parser.add_argument("--number_of_epochs", default=500, type=int, help="Number of training epochs")

    parser.add_argument("--content_loss_layer", default='none', help='Name of content layer (vgg19)'
                                                                     ' e.g. block4_conv1 or none')

    parser.add_argument("--checkpoints_dir", default="output/checkpoints", help="Folder with checkpoints")
    parser.add_argument("--checkpoint_ratio", default=30, type=int, help="Number of epochs between consecutive checkpoints")
    parser.add_argument("--generator_checkpoint", default=None, help="Previosly saved model of generator")
    parser.add_argument("--discriminator_checkpoint", default=None, help="Previosly saved model of discriminator")
    parser.add_argument("--nn_loss_area_size", default=1, type=int, help="Use nearest neighbour loss")
    parser.add_argument("--use_validation", default=0, type=int, help="Use validation")

    parser.add_argument('--dataset', default='market', choices=['market', 'fasion', 'prw', 'fasion128', 'fasion128128','forVid'],
                        help='Market, fasion or prw')


    parser.add_argument("--display_ratio", default=1, type=int,  help='Number of epochs between ploting')
    parser.add_argument("--start_epoch", default=0, type=int, help='Start epoch for starting from checkpoint')
    parser.add_argument("--pose_estimator", default='pose_estimator.h5',
                            help='Pretrained model for cao pose estimator')

    parser.add_argument("--images_for_test", default=12000, type=int, help="Number of images for testing")

    parser.add_argument("--use_input_pose", default=True, type=int, help='Feed to generator input pose')
    parser.add_argument("--use3D", default=True, type=int, help='3D attention')

    
    parser.add_argument("--warp_skip", default='stn', choices=['none', 'full', 'mask', 'stn'],
                        help="Type of warping skip layers to use.")
    parser.add_argument("--warp_agg", default='max', choices=['max', 'avg'],
                        help="Type of aggregation.")
    parser.add_argument("--dmax", default=6, type=int, help='decoder parameter, number of layers before using upsampling for attention')
    parser.add_argument("--kernel_size_last", default=1, type=int, help='attention parameter, kernel size of the last convolution')
    
    parser.add_argument("--disc_type", default='call', choices=['call', 'sim', 'warp'],
                        help="Type of discriminator call - concat all, sim - siamease, sharewarp - warp.")
    
    parser.add_argument("--use_bg", default=0, type=int, help='Use background images only for prw dataset')
    parser.add_argument("--return_att", default=0, type=int, help='return attention mask')

    parser.add_argument("--nb_inputs", default=2, type=int, help='How many input appearence are used')
    parser.add_argument("--nb_rec", default=1, type=int, help='How many recurences of the recursive model')


    
    parser.add_argument("--fusion_type", default="avg", choices=['att_dec','att_simple','att_dec_rec', 'avg'], help='operation used to fuse the encoders')

    parser.add_argument("--pose_rep_type", default='hm', choices=['hm', 'stickman'],
        help='Representation of the pose, hm - heatmap, stickman - like in vunet (https://github.com/CompVis/vunet).')
    parser.add_argument("--cache_pose_rep", default=1, type=int, help="Cache pose representation on disk.")

    parser.add_argument("--generated_images_dir", default='output/generated_images',
                        help='Folder with generated images from training dataset')

    parser.add_argument('--load_generated_images', default=0, type=int,
                        help='Load images from generated_images_dir or generate')

    parser.add_argument('--use_dropout_test', default=0, type=int,
                        help='To use dropout when generate images')

    args = parser.parse_args()

    args.images_dir_train = 'data/' + args.dataset + '-dataset/train'
    args.images_dir_test = 'data/' + args.dataset + '-dataset/test'

    args.annotations_file_train = 'data/' + args.dataset + '-annotation-train.csv'
    args.annotations_file_test = 'data/' + args.dataset + '-annotation-test.csv'
    
    inputType=([str(i)+"_tuples" for i in range(3,100)])[args.nb_inputs-1]
    
    args.file_train = 'data/' + args.dataset + '-'+inputType+'-train.csv'
    if args.dataset=='market':
        args.file_test = 'data/' + args.dataset + '-12_tuples-test.csv'
        # args.file_test = 'data/market-export-test.csv'
    else:
        args.file_test = 'data/' + args.dataset + '-6_tuples-test.csv'


    
    args.bg_images_dir_train = 'data/' + args.dataset + '-dataset/train-bg'
    args.bg_images_dir_test = 'data/' + args.dataset + '-dataset/test-bg'

        
    if args.dataset in ['fasion','forVid']:
        args.image_size = (256, 256)
    elif args.dataset == 'fasion128128':
        args.image_size = (128, 128)
    else:
        args.image_size = (128, 64)
    
    args.tmp_pose_dir = 'tmp/' + args.dataset + '/'

    del args.dataset

    return args

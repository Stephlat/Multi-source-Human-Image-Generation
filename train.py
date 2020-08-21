from conditional_gan import make_generator, make_discriminator, CGAN
import cmd
from gan.train import Trainer
import keras.backend as K
from pose_dataset import PoseHMDataset


def main():
    args = cmd.args()

    generator = make_generator(args.image_size, args.nb_inputs, args.use_input_pose, args.warp_skip, args.disc_type, args.warp_agg,
                               args.use_bg, args.pose_rep_type,args.fusion_type,args.return_att,args.nb_rec,args.dmax,args.kernel_size_last,args.use3D)
    if args.generator_checkpoint is not None:
        generator.load_weights(args.generator_checkpoint,by_name=True)

    if args.fusion_type in ["avg","att_simple"]:
        nbadditional=0
        assert(args.nb_rec==0)
    elif args.fusion_type in ["att_dec"]:
        nbadditional=args.nb_inputs
        assert(args.nb_rec==1)
    else:
        nbadditional=args.nb_inputs+(args.nb_rec-1)

    discriminator = make_discriminator(args.image_size,args.nb_inputs, args.use_input_pose, args.warp_skip, args.disc_type,
                                       args.warp_agg, args.use_bg, args.pose_rep_type,args.return_att,nbadditional)
    if args.discriminator_checkpoint is not None:
        discriminator.load_weights(args.discriminator_checkpoint)
    
    dataset = PoseHMDataset(test_phase=False, **vars(args))

    gan = CGAN(generator, discriminator, **vars(args))
    trainer = Trainer(dataset, gan, **vars(args))
    
    trainer.train()
    
if __name__ == "__main__":
    main()

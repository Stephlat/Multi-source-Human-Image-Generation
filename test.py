import os

from conditional_gan import make_generator
import cmd
from pose_dataset import PoseHMDataset

from gan.inception_score import get_inception_score

from skimage.io import imread, imsave
from skimage.measure import compare_ssim

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tqdm import tqdm
import re

def l1_score(generated_images, reference_images):
    score_list = []
    for reference_image, generated_image in zip(reference_images, generated_images):
        score = np.abs(2 * (reference_image/255.0 - 0.5) - 2 * (generated_image/255.0 - 0.5)).mean()
        score_list.append(score)
    return np.mean(score_list)


def ssim_score(generated_images, reference_images):
    ssim_score_list = []
    for reference_image, generated_image in zip(reference_images, generated_images):
        ssim = compare_ssim(reference_image, generated_image, gaussian_weights=True, sigma=1.5,
                            use_sample_covariance=False, multichannel=True,
                            data_range=generated_image.max() - generated_image.min())
        ssim_score_list.append(ssim)
    return np.mean(ssim_score_list),np.std(ssim_score_list)


def save_images(input_images, att_images,target_images, generated_images, names, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if len(att_images)>0:
        iterList=zip(map(list, zip(*input_images)),map(list, zip(*att_images)), target_images, generated_images, names)
    else:
        iterList=zip(map(list, zip(*input_images)), target_images, generated_images, names)

    for images in iterList:
        res_name = str('_'.join(images[-1])) + '.png'
        if len(att_images)>0:
            imagesList=images[0]+images[1]+list(images[1+1:])
        else:
            imagesList=images[0]+list(images[1:])

        imsave(os.path.join(output_folder, res_name), np.concatenate(imagesList[:-1], axis=1))


def create_masked_image(names, images, annotation_file):
    import pose_utils
    masked_images = []
    df = pd.read_csv(annotation_file, sep=':')
    for name, image in zip(names, images):
        to = name[1]
        ano_to = df[df['name'] == to].iloc[0]

        kp_to = pose_utils.load_pose_cords_from_strings(ano_to['keypoints_y'], ano_to['keypoints_x'])

        mask = pose_utils.produce_ma_mask(kp_to, image.shape[:2])
        masked_images.append(image * mask[..., np.newaxis])

    return masked_images


def load_generated_images(images_folder):
    input_images = []
    target_images = []
    generated_images = []
    names = []
    for img_name in os.listdir(images_folder):
        img = imread(os.path.join(images_folder, img_name))
        h = img.shape[1] / 3
        input_images.append(img[:, :h])
        target_images.append(img[:, h:2*h])
        generated_images.append(img[:, 2*h:])

        m = re.match(r'([A-Za-z0-9_]*.jpg)_([A-Za-z0-9_]*.jpg)', img_name)
        fr = m.groups()[0]
        to = m.groups()[1]
        names.append([fr, to])

    return input_images, target_images, generated_images, names



def generate_images(dataset, generator,  use_input_pose, nb_inputs=2,nbAtt=0):
    input_images = [[] for i in range(nb_inputs)]
    att_images = [[] for i in range(nb_inputs)]
    target_images = []
    generated_images = []
    names = []
    

    def deprocess_image(img):
        return (255 * ((img + 1) / 2.0)).astype(np.uint8)
    colormap=[tuple([int(255*x) for x in plt.get_cmap('jet')(i)[:-1]]) for i in range(255)]

    
    def colorizeGray(img,colmap):

        
        output=np.empty(img.shape[0:2]+(3,),dtype=np.uint8)
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                output[i,j,:]=colmap[int(img[i,j])]
        return output


    for _ in tqdm(range(dataset._file_test.shape[0])):
        batch, name = dataset.next_generator_sample_test(with_names=True)
        out = generator.predict(batch)
        out_index = 2*nb_inputs if use_input_pose else nb_inputs

        
        for i in range(nb_inputs):
            input_images[i].append(deprocess_image(batch[i]))
            if nbAtt>0:
                att_im=colorizeGray(deprocess_image(np.squeeze(out[2+out_index+i]-0.5)).astype(np.uint8),colormap)
                att_images[i].append(att_im.reshape(input_images[i][-1].shape))

        # out_index = 2 if use_input_pose else 1
        out_index = 2*nb_inputs if use_input_pose else nb_inputs
        
        target_images.append(deprocess_image(batch[out_index]))
        generated_images.append(deprocess_image(out[out_index]))

        names.append([name.iloc[0]['from_0'],name.iloc[0]['from_1'], name.iloc[0]['to']])

        
    input_array = [np.concatenate(input_img, axis=0) for input_img in input_images]
    if len(att_images[0])>1:
        att_array = [np.concatenate(input_img, axis=0) for input_img in att_images]
    else:
        att_array= []
    target_array = np.concatenate(target_images, axis=0)
    generated_array = np.concatenate(generated_images, axis=0)
    print [x.shape for x in input_array]
    print [x.shape for x in att_array]
    return input_array, att_array,target_array, generated_array, names


def test():
    args = cmd.args()
    if args.load_generated_images:
        print ("Loading images...")
        input_images, target_images, generated_images, names,att_images = load_generated_images(args.generated_images_dir)
    else:
        print ("Generate images...")
        from keras import backend as K
        if args.use_dropout_test:
            K.set_learning_phase(1)
        dataset = PoseHMDataset(test_phase=True, **vars(args))
        
        generator = make_generator(args.image_size, args.nb_inputs, args.use_input_pose, args.warp_skip, args.disc_type, args.warp_agg,
                                   args.use_bg, args.pose_rep_type,args.fusion_type,args.return_att,args.nb_rec,args.dmax,args.kernel_size_last,args.res_att,args.use3D,args.resDec)

        assert (args.generator_checkpoint is not None)
        generator.load_weights(args.generator_checkpoint)
        input_images, att_images, target_images, generated_images, names = generate_images(dataset, generator, args.use_input_pose,nb_inputs=args.nb_inputs,nbAtt=(args.nb_inputs if args.return_att else 0))
        print ("Save images to %s..." % (args.generated_images_dir, ))
        save_images(input_images,att_images, target_images, generated_images, names,
                        args.generated_images_dir)

    print ("Compute inception score...")
    inception_score = get_inception_score(generated_images)
    print ("Inception score %s +- %s" % (inception_score[0], inception_score[1]))

    print ("Compute structured similarity score (SSIM)...")
    structured_score,structured_score_std = ssim_score(generated_images, target_images)
    print ("SSIM score %s +- %s" % (structured_score,structured_score_std))

    print ("Compute l1 score...")
    norm_score = l1_score(generated_images, target_images)
    print ("L1 score %s" % norm_score)

    print ("Compute masked inception score...")
    generated_images_masked = create_masked_image(names, generated_images, args.annotations_file_test)
    reference_images_masked = create_masked_image(names, target_images, args.annotations_file_test)
    inception_score_masked = get_inception_score(generated_images_masked)

    print ("Inception score masked %s +- %s" % (inception_score_masked[0], inception_score_masked[1]))
    print ("Compute masked SSIM...")
    structured_score_masked,    structured_score_masked_std = ssim_score(generated_images_masked, reference_images_masked)
    print ("SSIM score masked %s +- %s " % (structured_score_masked, structured_score_masked_std))

    print ("Inception score = %s, masked = %s; SSIM score = %s, masked = %s; l1 score = %s" %
           (inception_score, inception_score_masked, structured_score, structured_score_masked, norm_score))



if __name__ == "__main__":
    test()



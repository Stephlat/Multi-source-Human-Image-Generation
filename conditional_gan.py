from keras.models import Model, Input, Sequential
from keras.layers import Flatten, Concatenate, Activation, Dropout, Dense, Average, Add, Lambda, Multiply,UpSampling2D,AveragePooling2D,UpSampling2D,BatchNormalization,GlobalAveragePooling2D,Add
from keras import regularizers
from keras.layers.convolutional import Conv2D, Conv2DTranspose, ZeroPadding2D, Cropping2D
from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers.advanced_activations import LeakyReLU
import keras.backend as K
from keras.activations import softmax
from keras.backend import tf as ktf
import tensorflow as tf
from gan.gan import GAN
from gan.layer_utils import content_features_model

from keras.optimizers import Adam
from pose_transform import AffineTransformLayer

inputsIdx=0
epsilon=1e-7

def gradient_x(img):
     gx = img[:,:,:-1,:] - img[:,:,1:,:]
     return gx

def gradient_y(img):
     gy = img[:,:-1,:,:] - img[:,1:,:,:]
     return gy


def loss_grad(x,target):
    target=[target]
    x=[x]
        
    disp_gradients_x = [gradient_x(d) for d in x]
    disp_gradients_y = [gradient_y(d) for d in x]

    image_gradients_x = [gradient_x(img) for img in target]
    image_gradients_y = [gradient_y(img) for img in target]

    weights_x = [K.exp(-K.mean(K.abs(g), 3, keepdims=True)) for g in image_gradients_x]
    weights_y = [K.exp(-K.mean(K.abs(g), 3, keepdims=True)) for g in image_gradients_y]

    smoothness_x = [disp_gradients_x[i] * weights_x[i] for i in range(len(target))]
    smoothness_y = [disp_gradients_y[i] * weights_y[i] for i in range(len(target))]

    return K.mean(K.abs(smoothness_x))+K.mean(K.abs(smoothness_y))

def entropyReg(x):
     return -K.mean(x*K.log(x)+(1-x)*K.log(1-x))
        
def block(out, nkernels, down=True, bn=True, dropout=False, leaky=True,name=None):
    if leaky:
        out = LeakyReLU(0.2)(out)
    else:
        out = Activation('relu')(out)

    if name is not None:
        name+="_conv_"+str(nkernels)

    if down:
        out = ZeroPadding2D((1, 1))(out)
        
        out = Conv2D(nkernels, kernel_size=(4, 4), strides=(2, 2), use_bias=False,name=name)(out)
    else:

        # out = Conv2DTranspose(nkernels, kernel_size=(4, 4), strides=(2, 2), use_bias=False)(out)
        out = UpSampling2D()(out)
        out = Conv2D(nkernels, kernel_size=(3, 3),padding='same',name=name)(out)

    if bn:
        out = InstanceNormalization()(out)
    if dropout:
        out = Dropout(0.5)(out)
    return out


def blockNet(inShape, nkernels,outDepth, bn=True, leaky=True,kernel_size_last=1,channelDepthFactor=4,use3D=True):
    inp=Input(inShape[1:])
    out=inp

    channelDepth=outDepth/channelDepthFactor
    # for nf in nkernels:
    #     out = Conv2D(nf, kernel_size=(3, 3), padding='same',activation='linear')(out)
    #     if bn:
    #          # out = BatchNormalization()(out)

    #          out = InstanceNormalization()(out)
    #     if leaky:
    #         out = LeakyReLU(0.2)(out)
    #     else:
    #         out = Activation('relu')(out)

    out2D=Conv2D(1, kernel_size=(kernel_size_last,kernel_size_last), use_bias=True, padding='same',activation='sigmoid',name='att2D_'+"_".join([str(x) for x in inShape]))(out)        


     # att_sum_layers=[Lambda(lambda x: tf.Print(x, [tf.shape(x),x]))(att_layers_d) for att_layers_d in att_sum_layers]            
    # print K.int_shape(out2D)
    # print  K.int_shape(Lambda(lambda x: K.sum(x,axis=[-2,-3],keepdims=True))(out2D))
    # out2D=Lambda(lambda x: K.clip(x,epsilon,1-epsilon))(out2D)
    # out2D=Lambda(lambda x: x/K.sum(x,axis=[-2,-3],keepdims=True))(out2D)
    # print K.int_shape(out2D)
    

    #to test none
    # out1D=GlobalAveragePooling2D()(out)
    # out1D=Dense(channelDepth)(out1D)
    # out1D=Dense(outDepth,activation='softmax')(out1D)


    #3D_relu

    out = InstanceNormalization()(out)
    out1D=GlobalAveragePooling2D()(out)
    out1D=Dense(channelDepth,activation='relu')(out1D)
    out1D=Dense(outDepth,activation='sigmoid')(out1D)
    if not use3D:
         out1D=Lambda(lambda x: x*0+1)(out1D)
    #full3D
    # out = InstanceNormalization()(out)
    # out1D=GlobalAveragePooling2D()(out)
    # out1D=Dense(outDepth,activation='softmax')(out1D)

    # print K.int_shape(out1D)
    # out1D=Lambda(lambda x :softmax(x,axis=-1))(out1D)
    # out1D=Lambda(lambda x :K.expand_dims(x,axis=-2))(out1D)
    # out1D=Lambda(lambda x :K.expand_dims(x,axis=-2))(out1D)

    out=Multiply()([out1D,out2D])
    
    att_mod=Model(input=inp,output=out)

    return att_mod

def UpNet(inShape):
    inp=Input(inShape[1:])
    out=inp
    out=Conv2D(1, kernel_size=(3,3), use_bias=True, padding='same',activation='sigmoid')(out)        
    return Model(input=inp,output=out)


def encoder(inps, nfilters=(64, 128, 256, 512, 512, 512)):
    layers = []
    if len(inps) != 1:
        global inputsIdx
        out = Concatenate(axis=-1,name="concatEncoder_"+str(inputsIdx))(inps)
        inputsIdx+=1
    else:
        out = inps[0]
    for i, nf in enumerate(nfilters):
        if i == 0:
            out = Conv2D(nf, kernel_size=(3, 3), padding='same',name="encoder_conv"+str(0))(out)
        elif i == len(nfilters) - 1:
            out = block(out, nf, bn=False,name="encoder_block_"+str(i))
        else:
            out = block(out, nf,name="encoder_block_"+str(i))
        layers.append(out)
    return layers

def encoderModel(appearenceShape,poseShape,avg=False, nfilters=(64, 128, 256, 512, 512, 512)):
    
    appearenceInput=Input(appearenceShape)
    if poseShape is not None:
        poseInput=Input(poseShape)
    else:
        poseInput=[]
    if avg:
        outposeInput=[Input(poseShape)]
    else:
        outposeInput=[]
    return Model(inputs=[appearenceInput,poseInput]+outposeInput,output=encoder([appearenceInput,poseInput]+outposeInput,nfilters))
    
# def decoder(skips, nfilters=(512, 512, 512, 256, 128, 3)):
#     out = None
#     for i, (skip, nf) in enumerate(zip(skips, nfilters)):
#         if 0 < i < 3:
#             out = Concatenate(axis=-1)([out, skip])
#             out = block(out, nf, down=False, leaky=False, dropout=True)
#         elif i == 0:
#             out = block(skip, nf, down=False, leaky=False, dropout=True)
#         elif i == len(nfilters) - 1:
#             out = Concatenate(axis=-1)([out, skip])
#             out = Activation('relu')(out)
#             out = Conv2D(nf, kernel_size=(3, 3), use_bias=True, padding='same')(out)
#         else:
#             out = Concatenate(axis=-1)([out, skip])
#             out = block(out, nf, down=False, leaky=False)
#     out = Activation('tanh')(out)
#     return out
def decoder_Model(skips, nfilters=(512, 512, 512, 256, 128, 3)):
    outList = []
    inputList = []
    out=None
    for i, (skip, nf) in enumerate(zip(skips, nfilters)):
        inputSkip=Input(K.int_shape(skip)[1:])
        if 0 < i < 3:
            out = Concatenate(axis=-1)([out, inputSkip])
            out = block(out, nf, down=False, leaky=False, dropout=True)
        elif i == 0:
            outList.append(inputSkip)
            out = block(inputSkip, nf, down=False, leaky=False, dropout=True)
            
        elif i == len(nfilters) - 1:
            out = Concatenate(axis=-1)([out, inputSkip])
            out = Activation('relu')(out)
            out = Conv2D(nf, kernel_size=(3, 3), use_bias=True, padding='same')(out)
            out = Activation('tanh')(out)
        else:
            out = Concatenate(axis=-1)([out, inputSkip])
            out = block(out, nf, down=False, leaky=False)
        outList.append(out)
        inputList.append(inputSkip)
    return Model(inputs=inputList,outputs=outList[::-1],name="modeldecoder")


def decoder_Attention(skips,dmax,kernel_size_last,convList=None, nfilters=(512, 512, 512, 256, 128, 3),use3D=True):
     out = None
     
     previousAtt=None
     for d, (skip, nf) in enumerate(zip(skips, nfilters)):

          if d==0:
               out=Average()(skip)
               att=[Concatenate(axis=-1)([out,skip_i]) for skip_i in skip]
               conv = blockNet(K.int_shape(att[0]),[],outDepth=K.int_shape(skip[0])[-1],kernel_size_last=kernel_size_last,use3D=use3D)
          elif previousAtt!=None and d<dmax:
               previousAtt=[UpSampling2D(size=(2, 2),interpolation='bilinear')(previousAtt_i) for previousAtt_i in previousAtt]
               att=[Concatenate(axis=-1)([out,skip_i,previousAtt_i]) for skip_i,previousAtt_i in zip(skip,previousAtt)]
               conv = blockNet(K.int_shape(att[0]),[],outDepth=K.int_shape(skip[0])[-1],kernel_size_last=kernel_size_last,use3D=use3D)
          else:
               previousAtt=[UpSampling2D(size=(2, 2))(previousAtt_i) for previousAtt_i in previousAtt]
               att=previousAtt

               conv = UpNet(K.int_shape(att[0]))
          skip_merge, att=applyAttentionSingle(conv,att,skip,name="att_"+str(d))

          
          previousAtt=att

          out = Concatenate(axis=-1)([out, skip_merge])

          
          
          if d == len(nfilters) - 1:
               # out = Concatenate(axis=-1)([out, skip_merge])
               out = Activation('relu')(out)
               out = Conv2D(nf, kernel_size=(3, 3), use_bias=True, padding='same',name="decoder_conv_"+str(d))(out)
          else:
               # out = Concatenate(axis=-1)([out, skip_merge])
               out = block(out, nf, down=False, leaky=False,name="decoder_"+str(d))
     out = Activation('tanh')(out)
     return out,att



from stn import SpatialTransformer
import numpy as np


def concatenate_skips(skips_app, skips_pose, warp, image_size, warp_agg, warp_skip):
    skips = []
    if warp_skip == 'stn':
        b = np.zeros((2, 3), dtype='float32')
        b[0, 0] = 1
        b[1, 1] = 1
        W = np.zeros((32, 6), dtype='float32')
        weights = [W, b.flatten()]

        locnet = Sequential()
        locnet.add(Dense(64, input_shape=(72, )))
        locnet.add(LeakyReLU(0.2))
        locnet.add(Dense(32))
        locnet.add(LeakyReLU(0.2))
        locnet.add(Dense(6, weights=weights))

    for i, (sk_app, sk_pose) in enumerate(zip(skips_app, skips_pose)):
        if i < 4:
            if warp_skip != 'stn':
                out = AffineTransformLayer(10 if warp_skip == 'mask' else 1, warp_agg, image_size)([sk_app] + warp)
            else:
                out = SpatialTransformer(locnet, K.int_shape(sk_app)[1:3])(warp + [sk_app])
            out = Concatenate(axis=-1)([out, sk_pose])
        else:
            out = Concatenate(axis=-1)([sk_app, sk_pose])
        skips.append(out)
    return skips



def applyAttentionSingle(conv,att_layers,enc_layers,name=""):

     att_layers = [conv(att_layers_i) for att_layers_i in att_layers]

     
     att_layers=[Lambda(lambda x: K.clip(x,epsilon,1-epsilon))(att_layers_i) for att_layers_i in att_layers]            
     att_sum_layers = Add()(att_layers)


     att_normed_layers = [Lambda(lambda x: x[0]/x[1])([att_layers_i,att_sum_layers]) for att_layers_i in att_layers]


     attenc=[Multiply()([att_normed_layers_i,enc_layers_i])
             for att_normed_layers_i,enc_layers_i in zip(att_normed_layers,enc_layers)]


     fused_enc_layers= Add()(attenc)
     return fused_enc_layers, att_normed_layers





def make_generator(image_size,nbInput, use_input_pose, warp_skip, disc_type, warp_agg, use_bg, pose_rep_type,fusion_type,return_Att,nb_rec,dmax,kernel_size_last,use3D):
     use_warp_skip = warp_skip != 'none'
     appearenceShape=list(image_size) + [3]
     input_img = [Input(appearenceShape) for i in range(nbInput)]
 
     output_pose = Input(list(image_size) + [18 if pose_rep_type == 'hm' else 3])
     output_img = Input(appearenceShape)
     bg_img = Input(appearenceShape) 
    
     nfilters_decoder = (512, 512, 512, 256, 128, 3) if max(image_size) == 128 else (512, 512, 512, 512, 256, 128, 3)
     nfilters_encoder = (64, 128, 256, 512, 512, 512) if max(image_size) == 128 else (64, 128, 256, 512, 512, 512, 512)
 
     convList=None
     
     if warp_skip == 'full':
         warp = [[Input((1, 8))] for i in range(nbInput)]
     elif warp_skip == 'mask':
         warp = [[Input((10, 8)),Input((10, image_size[0], image_size[1]))] for i in range(nbInput)]
         flat_warp = [item for sublist in warp for item in sublist]
     elif warp_skip == 'stn':
         warp = [[Input((72,))] for i in range(nbInput)]
     else:
         warp = []
         flat_warp=warp
 
     if use_input_pose:
         inputPoseShape=list(image_size) + [18 if pose_rep_type == 'hm' else 3]
         input_pose = [Input(inputPoseShape) for i in range(nbInput)]
     else:
         input_pose = []
         inputPoseShape=None
         
     if use_bg:
        bg_img = [bg_img]
     else:
        bg_img = [] 
 
     warp_in_disc = [] if disc_type != 'warp' else warp

     
     if use_warp_skip:
         encoderShared=encoderModel(appearenceShape,inputPoseShape,avg=False,  nfilters=nfilters_encoder)
         encoderPose=encoderModel(inputPoseShape,inputPoseShape,avg=False,  nfilters=nfilters_encoder)
         enc_app_layers = [encoderShared([input_img_i] +  [input_pose_i]) for input_img_i ,input_pose_i in zip(input_img,input_pose)]
         # enc_tg_layers = encoder([output_pose] + bg_img, nfilters=nfilters_encoder)
         enc_tg_layers = [encoderPose([input_pose_i,output_pose])for input_pose_i in input_pose]
         enc_layers =[ concatenate_skips(enc_app_layers_d, enc_tg_layers_i, warp_i, image_size, warp_agg, warp_skip) for enc_app_layers_d,enc_tg_layers_i,warp_i in zip(enc_app_layers,enc_tg_layers,warp)]
     else:
         encoderShared=encoderModel(appearenceShape,inputPoseShape, avg=True,nfilters=nfilters_encoder)

         enc_layers = [encoderShared([input_img_i] + [input_pose_i]+[output_pose]) for input_img_i, input_pose_i in zip(input_img,input_pose)]


     kernel_size=(1,)*6
     
     enc_layers_swaped = [[enc_layers_d[i] for enc_layers_d in enc_layers] for i in range(len(enc_layers[0]))]

     if fusion_type=="avg":
         decoderModel=decoder_Model(enc_layers[0][::-1], nfilters_decoder)        
         if nbInput==1:
              fused_enc_layers= enc_layers[0]
         else:
              fused_enc_layers =  [Average(name='AverageEnc_'+str(i))([enc_layers_d[i] for enc_layers_d in enc_layers]) for i in range(len(enc_layers[0]))]
         out = decoderModel(fused_enc_layers[::-1])[0]
         att_mask= [] 

     else:

          out,att_mask = decoder_Attention(enc_layers_swaped[::-1],nfilters=nfilters_decoder,dmax=dmax,kernel_size_last=kernel_size_last,use3D=use3D)
          att_mask=[Lambda(lambda x: K.mean(x,axis=-1,keepdims=True))(att_mask_i) for att_mask_i in att_mask]
     # out = decoder(fused_enc_layers[::-1], nfilters_decoder)
 
     warp_in_disc = [] if disc_type != 'warp' else flat_warp

     if (not return_Att or fusion_type=='avg') :
          att_mask= [] 
     generatorModel= Model(inputs=input_img + input_pose + [output_img, output_pose] + bg_img+ flat_warp,
                               outputs=input_img + input_pose + [out, output_pose]+att_mask  + bg_img ,name="modelGenerator")

     # generatorModel.summary()
     return generatorModel

def indiv_disc(image_size,use_input_pose, warp_skip, disc_type, warp_agg, use_bg, pose_rep_type,nbInput):

     input_img = Input(list(image_size) + [3]) 
     output_pose = Input(list(image_size) + [18 if pose_rep_type == 'hm' else 3])
     input_pose = Input(list(image_size) + [18 if pose_rep_type == 'hm' else 3])
     output_img = Input(list(image_size) + [3])

     if warp_skip == 'full':
         warp = [Input((10, 8))]
     elif warp_skip == 'mask':
         warp = [Input((10, 8)), Input((10, image_size[0], image_size[1]))]
     else:
         warp = []
 
     if use_input_pose:
         input_pose = input_pose
     else:
         input_pose = []
 
     if use_bg:
        bg_img = [bg_img]
     else:
        bg_img = []
 
 
     assert (not use_bg) or (disc_type == 'call')

     out = Concatenate(axis=-1,name="ConcatenateDiscInputs")([input_img, input_pose, output_img, output_pose] + bg_img)
     out = Conv2D(64, kernel_size=(4, 4), strides=(2, 2))(out)
     out = block(out, 128,name="disc_"+str(128))
     out = block(out, 256,name="disc_"+str(256))
     out = block(out, 512,name="disc_"+str(512))
     out = block(out, 1, bn=False,name="disc_"+str(1))
     out = Activation('sigmoid')(out)
     out = Flatten()(out)

     return Model(inputs=[input_img,input_pose,output_img, output_pose] + bg_img, outputs=[out],name="model_Disc_single")

  
def make_discriminator(image_size, nbInput,use_input_pose, warp_skip, disc_type, warp_agg, use_bg, pose_rep_type, return_Att,nbadditional):
    input_img = [Input(list(image_size) + [3]) for i in range(nbInput)]
    output_pose = Input(list(image_size) + [18 if pose_rep_type == 'hm' else 3])
    input_pose = [Input(list(image_size) + [18 if pose_rep_type == 'hm' else 3]) for i in range(nbInput)]
    output_img = Input(list(image_size) + [3])
    bg_img = Input(list(image_size) + [3]) 
    additional_generated = [Input(list(image_size) + [3]) for i in range(nbadditional)] 
    att_mask = [Input(list(image_size) + [1]) for i in range(nbInput)] 

    

    if warp_skip == 'full':
        warp = [Input((10, 8))]
    elif warp_skip == 'mask':
        warp = [Input((10, 8)), Input((10, image_size[0], image_size[1]))]
    else:
        warp = []

    if use_input_pose:
        input_pose = input_pose
    else:
        input_pose = []

    if use_bg:
       bg_img = [bg_img]
    else:
       bg_img = []

    if not return_Att:
        att_mask = []

    assert (not use_bg) or (disc_type == 'call')

    singleDisc=indiv_disc(image_size,use_input_pose, warp_skip, disc_type, warp_agg, use_bg, pose_rep_type,nbInput)
    # outList=[[singleDisc([img, inpose] + [outimg, output_pose] + bg_img) for img,inpose in zip(input_img,input_pose)] for outimg in [output_img]+additional_generated]
    outList=[singleDisc([img, inpose] + [output_img, output_pose] + bg_img) for img,inpose in zip(input_img,input_pose)]

    # out = Average()([item for sublist in outList for item in sublist])
    # out = Concatenate(axis=-1)([item for sublist in outList for item in sublist])
    if nbInput==1:
         out =outList[0]
    else:
         out = Concatenate(axis=-1)(outList)

    
    # out = outList[0][0]

    if not return_Att:
         att_mask=[]
    model= Model(inputs=input_img + input_pose + [output_img, output_pose]+att_mask+ bg_img, outputs=out,name="model_Disc_call")

    return model


def total_variation_loss(x, image_size):
    img_nrows, img_ncols = image_size
    assert K.ndim(x) == 4
    if K.image_data_format() == 'channels_first':
        a = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, 1:, :img_ncols - 1])
        b = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, :img_nrows - 1, 1:])
    else:
        a = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
        b = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


def nn_loss(reference, target, neighborhood_size=(3, 3)):
    v_pad = neighborhood_size[0] / 2
    h_pad = neighborhood_size[1] / 2
    val_pad = ktf.pad(reference, [[0, 0], [v_pad, v_pad], [h_pad, h_pad], [0, 0]],
                      mode='CONSTANT', constant_values=-10000)
    reference_tensors = []
    for i_begin in range(0, neighborhood_size[0]):
        i_end = i_begin - neighborhood_size[0] + 1
        i_end = None if i_end == 0 else i_end
        for j_begin in range(0, neighborhood_size[1]):
            j_end = j_begin - neighborhood_size[0] + 1
            j_end = None if j_end == 0 else j_end
            sub_tensor = val_pad[:, i_begin:i_end, j_begin:j_end, :]
            reference_tensors.append(ktf.expand_dims(sub_tensor, -1))
    reference = ktf.concat(reference_tensors, axis=-1)
    target = ktf.expand_dims(target, axis=-1)

    abs = ktf.abs(reference - target)
    norms = ktf.reduce_sum(abs, reduction_indices=[-2])
    loss = ktf.reduce_min(norms, reduction_indices=[-1])

    return loss


class CGAN(GAN):
    def __init__(self, generator, discriminator, l1_penalty_weight, gan_penalty_weight, use_input_pose, image_size,
                 content_loss_layer, tv_penalty_weight, nn_loss_area_size, lstruct_penalty_weight, smooth_penalty_weight,grad_penalty_weight,**kwargs):
        super(CGAN, self).__init__(generator, discriminator, generator_optimizer=Adam(2e-4, 0.5, 0.999),
                                   discriminator_optimizer=Adam(2e-4, 0.5, 0.999), **kwargs)
        # generator.summary()
        # discriminator.summary()

        self._l1_penalty_weight = l1_penalty_weight
        self.generator_metric_names = ['gan_loss','l1_loss', 'tv_loss', 'lstruct','lsmooth','lgrad']
        self._use_input_pose = use_input_pose
        self._image_size = image_size
        self._content_loss_layer = content_loss_layer
        self._gan_penalty_weight = gan_penalty_weight
        self._tv_penalty_weight = tv_penalty_weight
        self._smooth_penalty_weight = smooth_penalty_weight
        self._grad_penalty_weight = grad_penalty_weight
        self._nn_loss_area_size = nn_loss_area_size
        if lstruct_penalty_weight != 0:
            from keras.models import load_model
            self._pose_estimator = load_model(kwargs['pose_estimator'])
        self._lstruct_penalty_weight = lstruct_penalty_weight
        self.nb_inputs= kwargs['nb_inputs']

        self.nb_rec= kwargs['nb_rec']

        self.fusion_type= kwargs['fusion_type']
        
    def _compile_generator_loss(self):
        if self.fusion_type not in ["avg","att_simple"]:
             image_index=2*self.nb_inputs

        else:
             image_index = 2*self.nb_inputs if self._use_input_pose else self.nb_inputs
             # image_index = 2 if self._use_input_pose else 1
        # image_indexes=[2*self.nb_inputs]

        def st_loss(a, b):
            if self._nn_loss_area_size > 1:
                return nn_loss(a, b, (self._nn_loss_area_size, self._nn_loss_area_size))
            else:
                return K.mean(K.abs(a - b))

        if self._content_loss_layer != 'none':
            layer_name = self._content_loss_layer.split(',')
            cf_model = content_features_model(self._image_size, layer_name)
            reference = cf_model(self._generator_input[image_index])
            target = cf_model(self._discriminator_fake_input[image_index])
            l1_loss = K.constant(0)
            if type(reference) != list:
                reference = [reference]
                target = [target]
            for a, b in zip(reference, target):
                l1_loss = l1_loss + self._l1_penalty_weight * st_loss(a, b)

                
        else:
            l1_loss = K.constant(0)
             
            reference = self._generator_input[image_index]
            target = self._discriminator_fake_input[image_index]
            l1_loss = l1_loss + self._l1_penalty_weight * st_loss(reference, target)

            # l1_loss = self._l1_penalty_weight * st_loss(reference, target)

            
                
        if self._lstruct_penalty_weight != 0:
            target_struct = self._pose_estimator(self._generator_input[image_index][..., ::-1] / 2)[1][..., :18]
            struct = self._pose_estimator(self._discriminator_fake_input[image_index][..., ::-1] / 2)[1][..., :18]
            struct_loss = self._lstruct_penalty_weight * K.mean((target_struct - struct) ** 2)
        else:
            struct_loss = K.constant(0)
     
        if self._smooth_penalty_weight!=0:
             smooth_loss=K.constant(0)
             for i in range(self.nb_inputs):
                  att=self._discriminator_fake_input[2*self.nb_inputs+2+i]
                  smooth_loss+=self._smooth_penalty_weight*(entropyReg(att))

        else:
             smooth_loss=K.constant(0)


        if self._grad_penalty_weight!=0:
             att1=self._discriminator_fake_input[2*self.nb_inputs+2]
             att2=self._discriminator_fake_input[2*self.nb_inputs+2+1]
             grad_loss=self._grad_penalty_weight*(loss_grad(att1,reference)+loss_grad(att2,reference))

        else:
             grad_loss=K.constant(0)

             
             
        def smooth_loss_fn(y_true, y_pred):
             return smooth_loss

        def grad_loss_fn(y_true, y_pred):
             return grad_loss

        
        def struct_loss_fn(y_true, y_pred):
            return struct_loss

        def tv_loss(y_true, y_pred):
            return self._tv_penalty_weight * total_variation_loss(self._discriminator_fake_input[image_index],
                                                                  self._image_size)

        def l1_loss_fn(y_true, y_pred):
            return l1_loss

        def gan_loss_fn(y_true, y_pred):
            loss =super(CGAN, self)._compile_generator_loss()[0](y_true, y_pred)

            return K.constant(0) if self._gan_penalty_weight == 0 else self._gan_penalty_weight * loss


        def generator_loss(y_true, y_pred):
            return gan_loss_fn(y_true, y_pred) + l1_loss_fn(y_true, y_pred) + tv_loss(y_true, y_pred) + struct_loss_fn(y_true, y_pred)+ smooth_loss_fn(y_true, y_pred)+ grad_loss_fn(y_true, y_pred)

       
        return generator_loss, [gan_loss_fn, l1_loss_fn, tv_loss, struct_loss_fn,smooth_loss_fn,grad_loss_fn]
       

        

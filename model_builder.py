
# ssds part
#from lib.modeling.ssds import ssd
#from lib.modeling.ssds import ssd_lite
import rfb


#from lib.modeling.ssds import rfb_lite
#from lib.modeling.ssds import fssd
#from lib.modeling.ssds import fssd_lite
#from lib.modeling.ssds import yolo

ssds_map = {
                'rfb': rfb.build_rfb,
            }

# nets part
#from lib.modeling.nets import vgg
import resnet
#from lib.modeling.nets import mobilenet
#from lib.modeling.nets import darknet
networks_map = {
                    'resnet_18': resnet.resnet_18,
                    'resnet_34': resnet.resnet_34,
                    'resnet_50': resnet.resnet_50,
                    'resnet_101': resnet.resnet_101
               }

from prior_box import PriorBox
import torch

def _forward_features_size(model, img_size):
    model.eval()
    x = torch.rand(1, 3, img_size[0], img_size[1])
    x = torch.autograd.Variable(x, volatile=True) #.cuda()
    feature_maps = model(x, phase='feature')
    return [(o.size()[2], o.size()[3]) for o in feature_maps]


def create_model():
    '''
    '''
    ASPECT_RATIOS = [[2,3], [2, 3], [2, 3], [2, 3], [2], [2]]
    IMAGE_SIZE = [300, 300]
    FEATURE_LAYER = [[22, 34, 'S', 'S', '', ''], [512, 1024, 512, 256, 256, 256]]
    NUM_CLASSES = 21
    SIZES = [0.2, 0.95]
    STEPS = []
    CLIP = True
    #
    base = networks_map['resnet_50']
    number_box= [2*len(aspect_ratios) if isinstance(aspect_ratios[0], int) else len(aspect_ratios) for aspect_ratios in ASPECT_RATIOS]  
        
    model = ssds_map['rfb'](base=base, feature_layer=FEATURE_LAYER, mbox=number_box, num_classes= NUM_CLASSES)
    #
    print(model)
    feature_maps = _forward_features_size(model, IMAGE_SIZE)
    print('==>Feature map size:')
    print(feature_maps)
    # 
    priorbox = PriorBox(image_size=IMAGE_SIZE, feature_maps=feature_maps, aspect_ratios=ASPECT_RATIOS, 
                    scale=SIZES, archor_stride=STEPS, clip=CLIP)
    # priors = Variable(priorbox.forward(), volatile=True)

    return model, priorbox
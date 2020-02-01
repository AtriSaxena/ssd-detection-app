import os
import argparse
import torch
from torch.autograd import Variable
import cv2
import imageio

#from ssd import build_ssd
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

def dectect(img, neural_net, transform):
    """Detects objects on an image and draws rectangles around them
    :param img: image/frame of a video stream
    :param neural_net: pretrained SSD neural network
    :param transform: transforms the images to have the right format for the neural network
    :type img: <class 'numpy.ndarray'>
    :type neural_net: <class 'ssd.SSD'>
    :type transform: <class 'data.BaseTransform'>
    :return: image with rectangles around detected objects
    :rtype: <class 'numpy.ndarray'>
    """
    height, width = img.shape[:2]
    ## convert from numpy array to torch tensor
    transformed_img = transform(img)[0]
    ## turn transformed image from a numpy array to a torch tensor
    ## and switch/permute the color channels from RGB (0, 1, 2) to BRG (2, 0, 1)
    x = torch.from_numpy(transformed_img).permute(2, 0, 1)
    ## add fake dimension corresponding to the batch and turn it into a torch Variable
    x = Variable(x.unsqueeze(0))
    y = neural_net(x)  # feed torch Variable into the neural network
    ## get values of output y (torch Tensor)
    # detections = [batch, num of classes, num of occurence of classes, (score, x0, y0, x1, y1)]
    detections = y.data
    ## [width, height, width, height] = upper-left corner to lower-right corner
    scale = torch.Tensor([width, height, width, height])
    for i in range(detections.size(1)):
        occur = 0
        while detections[0, i, occur, 0] >= 0.6:
            ## get the points from the detection box
            pt = (detections[0, i, occur, 1:] * scale).numpy()
            ## draw the rectangle on the object in the image
            cv2.rectangle(img, (int(pt[0]), int(pt[1])),
                          (int(pt[2]), int(pt[3])), COLORS[i % 3], 2)
            ## display the label on top
            cv2.putText(img, labelmap[i - 1], (int(pt[0]), int(pt[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            occur += 1
    return img


# def load_neural_net(model, phase='train'):
#     """Loads a SSD neural network
#     :param model: path of the pretrained model
#     :param phase: 'train' for training or 'test' for testing the neural network (Default: 'train')
#     :type model: str
#     :type phase: str
#     :return: returns the SSD neural network
#     :rtype: <class 'ssd.SSD'>
#     """
#     neural_net = build_ssd(phase)  # 2 possible args: 'train' and 'test'
#     neural_net.load_state_dict(torch.load(
#         model, map_location=lambda storage, loc: storage))
#     return neural_net


def main(args):
    """Instantiates the neural network, draws rectangles on detected objects on a video stream 
    and saves this video.
    :param args: Namespace of positional and optional arguments
    :type args: <class 'argparse.Namespace'>
    """
    video = getattr(args, 'v_input')
    #model = getattr(args, 'SSD model')
    #neural_net = load_neural_net(model, phase='test')
    net = build_ssd('test', 300, 25)
    net.load_state_dict(torch.load(args.weights))
    output_name = 'output.mp4'
    output_dir = 'videos'
    possible_formats = ('.mov', '.avi', '.mpg',
                        '.mpeg', '.mp4', '.mkv', '.wmv')
    if args.output and args.output.lower().endswith(possible_formats):
        output_name = args.output

    store_path = os.path.join(output_dir, output_name)
    ## Create the transformation
    transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))

    ## object dtection on video
    reader = imageio.get_reader(video)
    fps = reader.get_meta_data()['fps']
    writer = imageio.get_writer(store_path, fps=fps)
    for frame in reader:
        frame = dectect(frame, net.eval(), transform)
        writer.append_data(frame)
    writer.close()
    reader.close()


if __name__ == '__main__':
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from data import BaseTransform, VOC_CLASSES as labelmap
    from ssd import build_ssd

    PARSER = argparse.ArgumentParser(
        description='Object detection on video stream')
    PARSER.add_argument('--weights', default='D:/CODE/SSD_Animal/weights/animal_ssd.pth',
                    type=str, help='Trained state_dict file path')
    PARSER.add_argument('--v_input', type=str,
                        help='Path to your video where the object detection should be done with.')
    PARSER.add_argument('-o', '--output', type=str,
                        help='Name of your output file (including file extension e.g.: myvideo.mp4).')
    ARGS = PARSER.parse_args()
    main(ARGS)

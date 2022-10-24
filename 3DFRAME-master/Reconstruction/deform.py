"""
Example 2. Optimizing vertices.
"""
from __future__ import division
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import glob

import torch
import torch.nn as nn
import numpy as np
from skimage.io import imread, imsave
import tqdm
import imageio
from utils.myloss import KeyPointLoss, shape_loss_fcn, iou_loss_fcn, img_loss_fcn, sym_loss_fcn
import neural_renderer as nr
from utils.Laplacianloss import Laplacianloss
from utils.FFD import FFD
from utils.FFD import FFDStructureLoss
import logging

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')


def findKeypoints(vertices):
    index = np.loadtxt("./index.txt", dtype=np.uint16)
    # print(index,index.shape)
    kp3d = torch.zeros((42, 3), dtype=torch.float32)
    for i, j in enumerate(index):
        kp3d[i] = vertices[0][j]
    kp3d = kp3d[None, :, :]
    # print("kp3d:",kp3d,kp3d.shape)
    return kp3d.cuda()


def coord_transform(kp3d, renderer):
    kp3d = nr.look(kp3d, renderer.eye, renderer.camera_direction)
    kp3d = nr.perspective(kp3d, renderer.viewing_angle)
    x = (kp3d[:, :, 0] + 1) / 2
    y = 1 - (kp3d[:, :, 1] + 1) / 2
    kp2d = torch.stack((x, y), dim=2)
    return kp2d


class Deformation_Model(nn.Module):
    def __init__(self, filename_obj, filename_ref="./ref_image.jpg", kp=None):
        super(Deformation_Model, self).__init__()

        # load .obj
        vertices, faces, vt_min, vt_max1, vt_max2 = nr.load_obj(filename_obj)
        self.face2save = faces
        vertices_ref = vertices.detach().clone()
        self.vertices_ref = vertices_ref
        self.vertices_ref.requires_grad = False
        self.register_buffer('faces', faces[None, :, :])

        # create textures
        texture_size = 2
        textures = torch.ones(1, self.faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32)
        self.register_buffer('textures', textures)

        # load reference image
        image_ref = torch.from_numpy(imread(filename_ref).astype(np.float32) / 255.)[None, ::]
        self.register_buffer('image_ref', image_ref)
        # kp2d_ref = torch.from_numpy(np.loadtxt("./data/coordinates_71.txt").astype(np.float32)).cuda()
        # kp2d_ref = kp2d_ref[None, :, :]
        self.kp2d_ref = kp

        # create FFD function
        self.FFD = FFD(vertices, 20, 15, 4)
        self.vertices = self.FFD()
        # create loss functions
        self.lploss_fcn = Laplacianloss(self.faces, self.vertices)
        self.FFDStructureLoss_fcn = FFDStructureLoss(self.FFD.p)

        # setup renderer
        renderer = nr.Renderer(camera_mode='look', camera_direction=[-0.0078,  0.2845,  2.4126])
        self.renderer = renderer

        # create log file
        # if os.path.exists("./loss.txt"):
        #     os.remove("./loss.txt")
        # logging.basicConfig(filename="./loss.txt", level=logging.DEBUG)

    def forward(self):
        # self.renderer.eye = torch.tensor([0.0113, -0.2886, -2.8484]).cuda()
        vt_ffd = self.FFD()
        image = self.renderer(vt_ffd, self.faces, mode='silhouettes')
        # compute keypointloss
        kp3d = findKeypoints(vt_ffd)
        kp2d = coord_transform(kp3d, self.renderer)
        kp_loss = KeyPointLoss(kp2d, self.kp2d_ref).cuda()
        #compute sym loss
        sym_loss = sym_loss_fcn(kp3d)
        # compute laplacianloss
        lp_loss = self.lploss_fcn()
        # compute shapeloss
        shape_loss = shape_loss_fcn(self.vertices_ref, vt_ffd)
        # compute iou loss
        iou_loss = iou_loss_fcn(self.image_ref, image).cuda()
        # compute image_loss
        img_loss = torch.sum((image - self.image_ref) ** 2)
        # compute FFD_loss
        ffdstructure_loss = self.FFDStructureLoss_fcn()
        # print(self.vertices)
        # loss = kp_loss + lp_loss + shape_loss + iou_loss + img_loss + ffdstructure_loss
        loss = kp_loss + sym_loss
        # log = "total_loss = " + str(loss.data) + "\r\n" + \
        #       "lp_loss = " + str(lp_loss.data) + "\r\n" + \
        #       "shape_loss = " + str(shape_loss.data) + "\r\n" + \
        #       "iou_loss = " + str(iou_loss.data) + "\r\n" + \
        #       "img_loss = " + str(img_loss.data) + "\r\n" + \
        #       "kp_loss = " + str(kp_loss.data) + "\r\n" + \
        #       "ffdstructure_loss = " + str(ffdstructure_loss) + "\r\n"
        #
        # logging.debug(log)
        # print("shape_loss = ", shape_loss)
        print("loss = ", loss)
        return loss


def make_gif(filename):
    with imageio.get_writer(filename, mode='I') as writer:
        for filename in sorted(glob.glob('/tmp/_tmp_*.png')):
            writer.append_data(imageio.imread(filename))
            os.remove(filename)
    writer.close()


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-io', '--filename_obj', type=str, default=os.path.join(data_dir, 'glass_average1.obj'))
#     parser.add_argument('-ir', '--filename_ref', type=str, default=os.path.join(data_dir, 'glass_ref.jpg'))
#     parser.add_argument(
#         '-oo', '--filename_output_optimization', type=str, default=os.path.join(data_dir, 'example2_optimization.gif'))
#     parser.add_argument(
#         '-or', '--filename_output_result', type=str, default=os.path.join(data_dir, 'example2_result.gif'))
#     parser.add_argument('-g', '--gpu', type=int, default=0)
#     args = parser.parse_args()
#
#     model = Deformation_Model(args.filename_obj, args.filename_ref).cuda()
#     # model.cuda()
#
#     optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
#     # optimizer.setup(model)
#     loop = tqdm.tqdm(range(90))
#     for i in loop:
#         loop.set_description('Optimizing')
#         # optimizer.target.cleargrads()
#         optimizer.zero_grad()
#         loss = model()
#         loss.backward(retain_graph=True)
#         optimizer.step()
#         images = model.renderer(model.FFD(), model.faces, mode='silhouettes')
#         image = images.detach().cpu().numpy()[0]
#         imsave('/tmp/_tmp_%04d.png' % i, image)
#     make_gif(args.filename_output_optimization)
#     nr.save_obj("./data/reconstruction_result.obj",torch.squeeze(model.FFD(),dim=0),model.face2save)
#
#     # draw object
#     loop = tqdm.tqdm(range(0, 360, 4))
#     draw_renderer = nr.Renderer(camera_mode="look_at")
#     model.renderer = draw_renderer
#     for num, azimuth in enumerate(loop):
#         loop.set_description('Drawing')
#         model.renderer.eye = nr.get_points_from_angles(2.732, 0, azimuth)
#         images = model.renderer(model.FFD(), model.faces, model.textures)
#         image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))
#         imsave('/tmp/_tmp_%04d.png' % num, image)
#     make_gif(args.filename_output_result)
#
#
# if __name__ == '__main__':
#     main()

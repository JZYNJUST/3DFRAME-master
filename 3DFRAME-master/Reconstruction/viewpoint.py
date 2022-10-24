"""
Example 4. Finding camera parameters.
"""
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import glob
import cv2
import torch
import torch.nn as nn
import numpy as np
from skimage.io import imread, imsave
import tqdm
import imageio
from utils.myloss import iou_loss_fcn
import neural_renderer as nr
import torchvision.transforms as transforms

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')
T = transforms.Compose([transforms.ToTensor()])

def findKeypoints(vertices):
    index = np.loadtxt("./index.txt", dtype=np.uint16)
    # print(index,index.shape)
    kp3d = torch.zeros((42, 3), dtype=torch.float32)
    for i, j in enumerate(index):
        kp3d[i] = vertices[j]
    kp3d = kp3d[None, :, :]
    # print("kp3d:",kp3d,kp3d.shape)
    return kp3d.cuda()

def my_normalize(z):
    # require z.size() = bxn
    # and z is a tensor
    batch_num = z.size()[0]
    point_num = z.size()[1]
    new_z = torch.zeros(batch_num, point_num)
    for batch_idx in range(batch_num):
        zs = z[batch_idx]
        new_zs = torch.zeros(point_num)
        z_max = torch.max(zs)
        z_min = torch.min(zs)
        for point_idx in range(point_num):
            new_zs[point_idx] = (zs[point_idx] - z_min) / (z_max - z_min)
        new_z[batch_idx] = new_zs
    return new_z


def kp_loss(ref, pre, p):
    # ref and pre are bxnx2
    batch_size = ref.size()[0]
    point_num = ref.size()[1]
    loss = torch.zeros(batch_size, point_num).cuda()
    for batch_idx in range(batch_size):
        for point_idx in range(point_num):
            loss[batch_idx][point_idx] = p[batch_idx][point_idx] ** 1 * \
                                         torch.sqrt((ref[batch_idx][point_idx][0] - pre[batch_idx][point_idx][0]) ** 2
                                                    + (ref[batch_idx][point_idx][1] - pre[batch_idx][point_idx][
                                             1]) ** 2)
    return torch.sum(loss)


class Model_ViewPoint(nn.Module):
    def __init__(self, filename_obj, filename_ref=None, kp3d_path="./3Dpoints.txt", weight_path="./weight.txt",
                 ref_points=None):
        # kp3d_path is the file path of 3d position of keypoints on average model
        # weight_path is the file path of the weight of keypoint loss function
        # refpoint is the keypoint detected in reference image: 42x2

        super(Model_ViewPoint, self).__init__()
        # load .obj
        vertices, faces, vt_min, vt_max1, vt_max2 = nr.load_obj(filename_obj)
        # load 3d keypoints
        # kp3d = torch.from_numpy(np.loadtxt(kp3d_path).astype(np.float32)).cuda()
        kp3d = findKeypoints(vertices)
        weight = torch.from_numpy(np.loadtxt(weight_path).astype(np.float32)).cuda()
        self.weight = weight[None, :]
        # img_ref = cv2.imread("./data/yanjing-16_17.jpg",0)
        # img_ref = T(img_ref)
        # self.img_ref = img_ref.cuda()
        # normalize 3d keypoints
        # kp3d -= vt_min
        # kp3d /= vt_max1
        # kp3d *= 2
        # kp3d -= vt_max2
        # # reshape kp3d
        # kp3d = kp3d[None, :, :]
        self.ref_points = ref_points
        self.kp3d = kp3d
        self.register_buffer('vertices', vertices[None, :, :])
        self.register_buffer('faces', faces[None, :, :])

        # create textures
        texture_size = 2
        textures = torch.ones(1, self.faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32)
        self.register_buffer('textures', textures)

        # load reference image
        # image_ref = torch.from_numpy((imread(filename_ref).max(-1) != 0).astype(np.float32))
        # self.register_buffer('image_ref', image_ref)

        # camera parameters
        self.camera_position = nn.Parameter(torch.from_numpy(np.array([1, 10, -14], dtype=np.float32)))
        self.camera_direction = nn.Parameter(torch.from_numpy(np.array([0, 0, 1], dtype=np.float32)))
        # self.P = nn.Parameter(torch.from_numpy(np.array([[[1,0,0,0],[0,1,0,0],[0,0,1,-14]]], dtype=np.float32)))
        # self.dist_coeffs = torch.cuda.FloatTensor([[0., 0., 0., 0., 0.]]).repeat(self.P.shape[0], 1)
        # setup renderer
        renderer = nr.Renderer(camera_mode='look', camera_direction=self.camera_direction)
        renderer.eye = self.camera_position
        self.renderer = renderer

    def forward(self):
        # kp3d = nr.projection(vertices=self.kp3d,P=self.P,dist_coeffs=self.dist_coeffs,orig_size=256)
        # kp3d = nr.look_at(self.kp3d, self.renderer.eye)
        kp3d = nr.look(self.kp3d, self.renderer.eye, self.camera_direction)
        kp3d = nr.perspective(kp3d, self.renderer.viewing_angle)
        # make projected 3d keypoints distributed from 0 to 1
        image = self.renderer(self.vertices, self.faces, mode='silhouettes')
        # image = self.renderer(self.vertices, self.faces, textures=self.textures, mode=None)
        x = (kp3d[:, :, 0] + 1) / 2
        y = 1 - (kp3d[:, :, 1] + 1) / 2
        z = kp3d[:, :, 2]
        z = my_normalize(z)
        # print("z:",z.size(),z)
        kp2d = torch.stack((x, y), dim=2)
        loss = kp_loss(kp2d, self.ref_points, self.weight)
        # kp_l = torch.mul((kp2d-self.ref_points) ** 2,z)
        # iou_l =iou_loss(image,self.img_ref)
        print("self.renserer.eye=", self.renderer.eye)
        print("self.renderer.camera_position=", self.camera_position)
        print("self.renderer.camera_direction", self.camera_direction)
        # with open("./data/camera_para.txt", 'w') as camera_file:
        #     np_eye = self.renderer.eye.detach().cpu().numpy()
        #     np_campos = self.camera_position.detach().cpu().numpy()
        #     np_camdir = self.camera_direction.detach().cpu().numpy()
        #     camera_file.write(str(np_eye) + "\n")
        #     camera_file.write(str(np_campos) + "\n")
        #     camera_file.write(str(np_camdir) + "\n")
        #     camera_file.close()
        return loss


def make_gif(filename):
    with imageio.get_writer(filename, mode='I') as writer:
        for filename in sorted(glob.glob('./tmp/_tmp_*.png')):
            writer.append_data(imread(filename))
            os.remove(filename)
    writer.close()


def make_reference_image(filename_ref, filename_obj):
    model = Model_ViewPoint(filename_obj)
    model.cuda()

    model.renderer.eye = nr.get_points_from_angles(2.732, 30, -15)
    images = model.renderer.render(model.vertices, model.faces, torch.tanh(model.textures))
    image = images.detach().cpu().numpy()[0]
    imsave(filename_ref, image)

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-io', '--filename_obj', type=str, default=os.path.join(data_dir, 'glass_average1.obj'))
#     parser.add_argument('-ir', '--filename_ref', type=str, default=os.path.join(data_dir, 'example4_ref.png'))
#     parser.add_argument('-or', '--filename_output', type=str, default=os.path.join(data_dir, 'example4_result.gif'))
#     parser.add_argument('-mr', '--make_reference_image', type=int, default=0)
#     parser.add_argument('-g', '--gpu', type=int, default=0)
#     args = parser.parse_args()
#
#     if args.make_reference_image:
#         make_reference_image(args.filename_ref, args.filename_obj)
#
#     model = Model(args.filename_obj, args.filename_ref)
#     model.cuda()
#
#     # optimizer = chainer.optimizers.Adam(alpha=0.1)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
#     loop = tqdm.tqdm(range(800))
#     for i in loop:
#         optimizer.zero_grad()
#         loss = model()
#         loss.backward()
#         optimizer.step()
#         images = model.renderer(model.vertices, model.faces, torch.tanh(model.textures))
#         image = images.detach().cpu().numpy()[0].transpose(1,2,0)
#         imsave('/tmp/_tmp_%04d.png' % i, image)
#         loop.set_description('Optimizing (loss %.4f)' % loss.data)
#         # if loss.item() < 70:
#         #     break
#     make_gif(args.filename_output)
#
#
# if __name__ == '__main__':
#     main()

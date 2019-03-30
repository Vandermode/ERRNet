# import tensorflow as tf
import torch
import numpy as np
# import sklearn.manifold.t_sne

class TensorAxis:
    N = 0
    C = 1
    H = 2
    W = 3


class CSFlow:
    def __init__(self, sigma=float(0.1), b=float(1.0)):
        self.b = b
        self.sigma = sigma

    def __calculate_CS(self, scaled_distances, axis_for_normalization=TensorAxis.C):
        self.scaled_distances = scaled_distances
        self.cs_weights_before_normalization = torch.exp((self.b - scaled_distances) / self.sigma)
        # self.cs_weights_before_normalization = 1 / (1 + scaled_distances)
        self.cs_NHWC = CSFlow.sum_normalize(self.cs_weights_before_normalization, axis_for_normalization)
        
        # print('cs_NHWC:{}'.format(self.cs_NHWC.sum()))

        # self.cs_NHWC = self.cs_weights_before_normalization

    # def reversed_direction_CS(self):
    #     cs_flow_opposite = CSFlow(self.sigma, self.b)
    #     cs_flow_opposite.raw_distances = self.raw_distances
    #     work_axis = [TensorAxis.H, TensorAxis.W]
    #     relative_dist = cs_flow_opposite.calc_relative_distances(axis=work_axis)
    #     cs_flow_opposite.__calculate_CS(relative_dist, work_axis)
    #     return cs_flow_opposite

    # --
    @staticmethod
    def create_using_L2(I_features, T_features, sigma=float(0.5), b=float(1.0)):
        cs_flow = CSFlow(sigma, b)
        sT = T_features.shape
        sI = I_features.shape # N, C, H, W

        Ivecs = torch.reshape(I_features, (sI[0], sI[1], -1))
        Tvecs = torch.reshape(T_features, (sI[0], sT[1], -1))
        r_Ts = torch.sum(Tvecs * Tvecs, 1) # N C P
        r_Is = torch.sum(Ivecs * Ivecs, 1)

        # print('r_Ts:{}'.format(r_Ts.sum()))
        # print('r_Is:{}'.format(r_Is.sum()))
        
        raw_distances_list = []
        for i in range(sT[0]):
            Ivec, Tvec, r_T, r_I = Ivecs[i], Tvecs[i], r_Ts[i], r_Is[i]
            A = torch.transpose(Tvec, 0, 1) @ Ivec  # (matrix multiplication)
            cs_flow.A = A
            # A = tf.matmul(Tvec, tf.transpose(Ivec))
            r_T = torch.reshape(r_T, [-1, 1])  # turn to column vector
            dist = r_T - 2 * A + r_I
            
            # print('dist:{}'.format(dist.mean()))
            
            dist = torch.reshape(dist, shape=(1, dist.shape[0], sI[2], sI[3]))
            # protecting against numerical problems, dist should be positive
            dist = torch.clamp(dist, min=float(0.0))
            # dist = tf.sqrt(dist)
            raw_distances_list += [dist]

        cs_flow.raw_distances = torch.cat(raw_distances_list)

        # print('raw_distances:{}'.format(cs_flow.raw_distances.mean()))

        relative_dist = cs_flow.calc_relative_distances()
        
        # print('relative_dist:{}'.format(relative_dist.mean()))

        cs_flow.__calculate_CS(relative_dist)
        return cs_flow

    # --
    @staticmethod
    def create_using_L1(I_features, T_features, sigma=float(0.5), b=float(1.0)):
        cs_flow = CSFlow(sigma, b)
        sT = T_features.shape
        sI = I_features.shape

        Ivecs = torch.reshape(I_features, (sI[0], -1, sI[3]))
        Tvecs = torch.reshape(T_features, (sI[0], -1, sT[3]))
        raw_distances_list = []
        for i in range(sT[0]):
            Ivec, Tvec = Ivecs[i], Tvecs[i]
            dist = torch.abs(torch.sum(Ivec.unsqueeze(1) - Tvec.unsqueeze(0), dim=2))
            dist = torch.reshape(torch.transpose(dist, 0, 1), shape=(1, sI[1], sI[2], dist.shape[0]))
            # protecting against numerical problems, dist should be positive
            dist = torch.clamp(dist, min=float(0.0))
            # dist = tf.sqrt(dist)
            raw_distances_list += [dist]

        cs_flow.raw_distances = torch.cat(raw_distances_list)

        relative_dist = cs_flow.calc_relative_distances()
        cs_flow.__calculate_CS(relative_dist)
        return cs_flow

    # --
    @staticmethod
    def create_using_dotP(I_features, T_features, sigma=float(0.5), b=float(1.0)):
        cs_flow = CSFlow(sigma, b)
        # prepare feature before calculating cosine distance
        T_features, I_features = cs_flow.center_by_T(T_features, I_features)
        T_features = CSFlow.l2_normalize_channelwise(T_features)
        I_features = CSFlow.l2_normalize_channelwise(I_features)

        # work seperatly for each example in dim 1
        cosine_dist_l = []
        N = T_features.size()[0]
        for i in range(N):
            T_features_i = T_features[i, :, :, :].unsqueeze_(0) 
            I_features_i = I_features[i, :, :, :].unsqueeze_(0)
            patches_PC11_i = cs_flow.patch_decomposition(T_features_i)  # 1CHW --> PC11, with P=H*W (C_out x C_in x H x W)
            cosine_dist_i = torch.nn.functional.conv2d(I_features_i, patches_PC11_i)
            # cosine_dist_1HWC = cosine_dist_i.permute((0, 2, 3, 1))
            cosine_dist_l.append(cosine_dist_i) # 1PHW

        cs_flow.cosine_dist = torch.cat(cosine_dist_l, dim=0)

        cs_flow.raw_distances = - (cs_flow.cosine_dist - 1) / 2  ### why -

        relative_dist = cs_flow.calc_relative_distances()
        cs_flow.__calculate_CS(relative_dist)
        return cs_flow

    def calc_relative_distances(self, axis=TensorAxis.C):
        epsilon = 1e-5
        div = torch.min(self.raw_distances, dim=axis, keepdim=True)[0]
        relative_dist = self.raw_distances / (div + epsilon)
        return relative_dist

    @staticmethod
    def sum_normalize(cs, axis=TensorAxis.C):
        reduce_sum = torch.sum(cs, dim=axis, keepdim=True)
        cs_normalize = torch.div(cs, reduce_sum)
        return cs_normalize

    def center_by_T(self, T_features, I_features):
        # assuming both input are of the same size
        # calculate stas over [batch, height, width], expecting 1x1xDepth tensor
        axes = [0, 1, 2]
        self.meanT = T_features.mean(TensorAxis.N, keepdim=True).mean(TensorAxis.H, keepdim=True).mean(TensorAxis.W, keepdim=True)
        # self.varT = T_features.var(0, keepdim=True).var(2, keepdim=True).var(3, keepdim=True)
        self.T_features_centered = T_features - self.meanT
        self.I_features_centered = I_features - self.meanT

        return self.T_features_centered, self.I_features_centered

    @staticmethod
    def l2_normalize_channelwise(features):
        norms = features.norm(p=2, dim=TensorAxis.C, keepdim=True)
        features = features.div(norms)
        return features

    def patch_decomposition(self, T_features):
        # 1HWC --> 11PC --> PC11, with P=H*W
        (_, C, H, W) = T_features.shape
        P = H * W
        patches_PC11 = T_features.reshape(shape=(C, P, 1, 1)).permute(dims=(1, 0, 2, 3))
        return patches_PC11

    @staticmethod
    def pdist2(x, keepdim=False):
        sx = x.shape
        x = x.reshape(shape=(sx[0], sx[1] * sx[2], sx[3]))
        differences = x.unsqueeze(2) - x.unsqueeze(1)
        distances = torch.sum(differences**2, -1)
        if keepdim:
            distances = distances.reshape(shape=(sx[0], sx[1], sx[2], sx[3]))
        return distances

    @staticmethod
    def calcR_static(sT, order='C', deformation_sigma=0.05):
        # oreder can be C or F (matlab order)
        pixel_count = sT[0] * sT[1]

        rangeRows = range(0, sT[1])
        rangeCols = range(0, sT[0])
        Js, Is = np.meshgrid(rangeRows, rangeCols)
        row_diff_from_first_row = Is
        col_diff_from_first_col = Js

        row_diff_from_first_row_3d_repeat = np.repeat(row_diff_from_first_row[:, :, np.newaxis], pixel_count, axis=2)
        col_diff_from_first_col_3d_repeat = np.repeat(col_diff_from_first_col[:, :, np.newaxis], pixel_count, axis=2)

        rowDiffs = -row_diff_from_first_row_3d_repeat + row_diff_from_first_row.flatten(order).reshape(1, 1, -1)
        colDiffs = -col_diff_from_first_col_3d_repeat + col_diff_from_first_col.flatten(order).reshape(1, 1, -1)
        R = rowDiffs ** 2 + colDiffs ** 2
        R = R.astype(np.float32)
        R = np.exp(-(R) / (2 * deformation_sigma ** 2))
        return R






# --------------------------------------------------
#           CX loss
# --------------------------------------------------



def CX_loss(I_features, T_features, deformation=False, dis=False):
    # T_features = tf.convert_to_tensor(T_features, dtype=tf.float32)
    # I_features = tf.convert_to_tensor(I_features, dtype=tf.float32)
    # since this is a convertion of tensorflow to pytorch we permute the tensor from
    # T_features = normalize_tensor(T_features)
    # I_features = normalize_tensor(I_features)

    # since this originally Tensorflow implemntation
    # we modify all tensors to be as TF convention and not as the convention of pytorch.
    # def from_pt2tf(Tpt):
    #     Ttf = Tpt.permute(0, 2, 3, 1)
    #     return Ttf
    # N x C x H x W --> N x H x W x C
    # T_features_tf = from_pt2tf(T_features)
    # I_features_tf = from_pt2tf(I_features)

    cs_flow = CSFlow.create_using_dotP(I_features, T_features, sigma=1.0)
    # cs_flow = CSFlow.create_using_L2(I_features, T_features, sigma=1.0)
    # sum_normalize:
    # To:
    cs = cs_flow.cs_NHWC

    # print('cs:{}'.format(cs.std()))

    if deformation:
        deforma_sigma = 0.001
        sT = T_features_tf.shape[1:2 + 1]
        R = CSFlow.calcR_static(sT, deformation_sigma=deforma_sigma)
        cs *= torch.Tensor(R).unsqueeze(dim=0).cuda()

    if dis:
        CS = []
        k_max_NC = torch.max(torch.max(cs, dim=1)[1], dim=1)[1]
        indices = k_max_NC.cpu()
        N, C = indices.shape
        for i in range(N):
            CS.append((C - len(torch.unique(indices[i, :]))) / C)
        score = torch.FloatTensor(CS)
    else:
        # reduce_max X and Y dims
        # cs = CSFlow.pdist2(cs,keepdim=True) N C H W
        k_max_NC = torch.max(torch.max(cs, dim=2)[0], dim=2)[0]
        # reduce mean over C(H*W) dim 
        CS = torch.mean(k_max_NC, dim=1)
        # score = 1/CS
        # score = torch.exp(-CS*10)
        score = -torch.log(CS)
    # reduce mean over N dim
    # CX_loss = torch.mean(CX_loss)
    return score


def symetric_CX_loss(T_features, I_features):
    score = (CX_loss(T_features, I_features) + CX_loss(I_features, T_features)) / 2
    return score

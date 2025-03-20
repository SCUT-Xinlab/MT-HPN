import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import config

lamda = config.lamda
SE = config.SE
node = config.node  # number of rois

class MT_HGPC_SE(nn.Module):
    def __init__(self, in_dim, out_dim, node_list):
        super(MT_HGPC_SE, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.node_types = len(node_list)
        self.node_list = node_list

        self.convs_1 = []
        self.convs_2 = []

        for i in range(self.node_types):
            self.convs_1.append(nn.Conv2d(in_dim, out_dim // 2, (1, len(self.node_list[i]))))
            nn.init.normal_(self.convs_1[-1].weight, std=math.sqrt(
                2 * (1 - lamda) / (len(self.node_list[i]) * in_dim + len(self.node_list[i]) * out_dim // 2)))
        self.convs_1 = nn.ModuleList(self.convs_1)

        for i in range(self.node_types):
            self.convs_2.append(nn.Conv2d(in_dim, out_dim // 2, (1, len(self.node_list[i]))))
            nn.init.normal_(self.convs_2[-1].weight, std=math.sqrt(
                2 * (1 - lamda) / (len(self.node_list[i]) * in_dim + len(self.node_list[i]) * out_dim // 2)))
        self.convs_2 = nn.ModuleList(self.convs_2)

        self.trans_1 = nn.Conv2d(out_dim // 2, out_dim // 4, 1)
        nn.init.normal_(self.trans_1.weight, std=math.sqrt(4 / (out_dim // 2 + out_dim // 4)))
        self.trans_2 = nn.Conv2d(out_dim // 2, out_dim // 4, 1)
        nn.init.normal_(self.trans_2.weight, std=math.sqrt(4 / (out_dim // 2 + out_dim // 4)))

        self.trans_11 = nn.Conv2d(out_dim// 2, out_dim // 4, 1) #
        nn.init.normal_(self.trans_11.weight, std=math.sqrt(4 / (out_dim // 2 + out_dim // 4)))
        self.trans_12 = nn.Conv2d(out_dim // 2, out_dim // 4, 1)
        nn.init.normal_(self.trans_12.weight, std=math.sqrt(4 / (out_dim // 2 + out_dim // 4)))

        self.trans_21 = nn.Conv2d(out_dim // 2, out_dim // 4, 1)
        nn.init.normal_(self.trans_21.weight, std=math.sqrt(4 / (out_dim // 2 + out_dim // 4)))
        self.trans_22 = nn.Conv2d(out_dim // 2, out_dim // 4, 1)
        nn.init.normal_(self.trans_22.weight, std=math.sqrt(4 / (out_dim // 2 + out_dim // 4)))

        self.convres_1 = nn.Conv2d(in_dim, out_dim, 1)
        nn.init.normal_(self.convres_1.weight, std=math.sqrt(4 * lamda / (in_dim + out_dim)))
        self.convres_2 = nn.Conv2d(in_dim, out_dim, 1)
        nn.init.normal_(self.convres_2.weight, std=math.sqrt(4 * lamda / (in_dim + out_dim)))

        self.sed_1 = nn.Linear(out_dim, int(out_dim / SE), False)
        self.seu_1 = nn.Linear(int(out_dim / SE), out_dim, False)

        self.sed_2 = nn.Linear(out_dim, int(out_dim / SE), False)
        self.seu_2 = nn.Linear(int(out_dim / SE), out_dim, False)

    def forward(self, y, z):

        res_y = self.convres_1(y)
        res_z = self.convres_2(z)

        batchsize = y.size(0)
        y_area = self.convs_1[0](y[:, :, :, self.node_list[0]])
        for i in range(1, self.node_types):
            y_self = self.convs_1[i](y[:, :, :, self.node_list[i]])
            y_area = torch.cat([y_area, y_self], dim=-1)

        z_area = self.convs_2[0](z[:, :, :, self.node_list[0]])
        for i in range(1, self.node_types):
            z_self = self.convs_2[i](z[:, :, :, self.node_list[i]])
            z_area = torch.cat([z_area, z_self], dim=-1)

        q_y = F.normalize(self.trans_1(y_area), dim=1)
        q_y = q_y.permute(0, 2, 3, 1)
        k_y = q_y.permute(0, 1, 3, 2)
        v_y = y_area.permute(0, 2, 3, 1)

        q_z = F.normalize(self.trans_2(z_area), dim=1)
        q_z = q_z.permute(0, 2, 3, 1)
        k_z = q_z.permute(0, 1, 3, 2)
        v_z = z_area.permute(0, 2, 3, 1)

        qk_ys = q_y @ k_y / math.sqrt(self.node_types)
        y_area= (qk_ys @ v_y).permute(0, 3, 1, 2)
        y_c = y_area[:, :, :, 0].unsqueeze(-1)
        qk_zs = q_z @ k_z / math.sqrt(self.node_types)
        z_area = (qk_zs @ v_z).permute(0, 3, 1, 2)
        z_c = z_area[:, :, :, 0].unsqueeze(-1)

        qe_y = F.normalize(self.trans_11(y_c), dim=1)
        qe_y = qe_y.permute(0, 3, 2, 1)
        ke_y = F.normalize(self.trans_12(y_c), dim=1)
        ke_y = ke_y.permute(0, 3, 1, 2)
        ve_y = y_c.permute(0, 3, 2, 1)

        qe_z = F.normalize(self.trans_21(z_c), dim=1)
        qe_z = qe_z.permute(0, 3, 2, 1)
        ke_z = F.normalize(self.trans_22(z_c), dim=1)
        ke_z = ke_z.permute(0, 3, 1, 2)
        ve_z = z_c.permute(0, 3, 2, 1)

        qke_y = qe_y @ ke_z / math.sqrt(node)
        qke_ys = qe_y @ ke_y / math.sqrt(node)
        qke_y = F.softmax(qke_y,dim=3)
        qke_ys = F.softmax(qke_ys, dim=3)
        y_c_p = (qke_y @ ve_z).permute(0, 3, 2, 1)
        y_c_n = (qke_ys @ ve_y).permute(0, 3, 2, 1)
        y_c = torch.cat((y_c_p, y_c_n), dim = 1)
        y_C = y_c.expand(batchsize, self.out_dim, node, node).contiguous()
        y_R = y_C.permute(0, 1, 3, 2)
        y = y_C + y_R

        qke_z = qe_z @ ke_y / math.sqrt(node)
        qke_zs = qe_z @ ke_z / math.sqrt(node)
        qke_z = F.softmax(qke_z, dim=3)
        qke_zs = F.softmax(qke_zs, dim=3)
        z_c_p = (qke_z @ ve_y).permute(0, 3, 2, 1)
        z_c_n = (qke_zs @ ve_z).permute(0, 3, 2, 1)
        z_c = torch.cat((z_c_p, z_c_n), dim=1)
        z_C = z_c.expand(batchsize, self.out_dim, node, node).contiguous()
        z_R = z_C.permute(0, 1, 3, 2)
        z = z_C + z_R

        se_y = torch.mean(y, (2, 3))
        se_y = self.sed_1(se_y)
        se_y = F.relu(se_y)
        se_y = self.seu_1(se_y)
        se_y = torch.sigmoid(se_y)
        se_y = se_y.unsqueeze(2).unsqueeze(3)

        y = y.mul(se_y)
        y = y + res_y

        se_z = torch.mean(z, (2, 3))
        se_z = self.sed_2(se_z)
        se_z = F.relu(se_z)
        se_z = self.seu_2(se_z)
        se_z = torch.sigmoid(se_z)
        se_z = se_z.unsqueeze(2).unsqueeze(3)

        z = z.mul(se_z)
        z = z + res_z

        return y,z

class MT_GPC_SE(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MT_GPC_SE, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv_1 = nn.Conv2d(in_dim, out_dim // 2, (1, node))
        nn.init.normal_(self.conv_1.weight, std=math.sqrt(2 * (1 - lamda) / (node * in_dim + node * out_dim // 2)))
        self.conv_2 = nn.Conv2d(in_dim, out_dim // 2, (1, node))
        nn.init.normal_(self.conv_2.weight, std=math.sqrt(2 * (1 - lamda) / (node * in_dim + node * out_dim // 2)))

        self.trans_11 = nn.Conv2d(out_dim // 2, out_dim // 4, 1)
        nn.init.normal_(self.trans_11.weight, std=math.sqrt(4 / (out_dim // 2 + out_dim // 4)))
        self.trans_12 = nn.Conv2d(out_dim // 2, out_dim // 4, 1)
        nn.init.normal_(self.trans_12.weight, std=math.sqrt(4 / (out_dim // 2 + out_dim // 4)))
        self.trans_21 = nn.Conv2d(out_dim // 2, out_dim // 4, 1)
        nn.init.normal_(self.trans_21.weight, std=math.sqrt(4 / (out_dim // 2 + out_dim // 4)))
        self.trans_22 = nn.Conv2d(out_dim // 2, out_dim // 4, 1)
        nn.init.normal_(self.trans_22.weight, std=math.sqrt(4 / (out_dim // 2 + out_dim // 4)))

        self.convres_1 = nn.Conv2d(in_dim, out_dim, 1)
        nn.init.normal_(self.convres_1.weight, std=math.sqrt(4 * lamda / (in_dim + out_dim)))
        self.convres_2 = nn.Conv2d(in_dim, out_dim, 1)
        nn.init.normal_(self.convres_2.weight, std=math.sqrt(4 * lamda / (in_dim + out_dim)))

        self.sed_1 = nn.Linear(out_dim, int(out_dim / SE), False)
        self.seu_1 = nn.Linear(int(out_dim / SE), out_dim, False)

        self.sed_2 = nn.Linear(out_dim, int(out_dim / SE), False)
        self.seu_2 = nn.Linear(int(out_dim / SE), out_dim, False)

    def forward(self, y, z):

        res_y = self.convres_1(y)
        res_z = self.convres_2(z)

        batchsize = y.size(0)
        y_c = self.conv_1(y)
        z_c = self.conv_2(z)

        q_y = F.normalize(self.trans_11(y_c), dim=1)
        q_y = q_y.permute(0, 3, 2, 1)
        k_y = F.normalize(self.trans_12(y_c), dim=1)
        k_y = k_y.permute(0, 3, 1, 2)
        v_y = y_c.permute(0, 3, 2, 1)

        q_z = F.normalize(self.trans_21(z_c), dim=1)
        q_z = q_z.permute(0, 3, 2, 1)
        k_z = F.normalize(self.trans_22(z_c), dim=1)
        k_z = k_z.permute(0, 3, 1, 2)
        v_z = z_c.permute(0, 3, 2, 1)

        qk_y = q_y @ k_z / math.sqrt(node)
        qk_ys = q_y @ k_y / math.sqrt(node)
        qk_y = torch.softmax(qk_y, 3)
        qk_ys = torch.softmax(qk_ys, 3)
        y_c_pos = (qk_y @ v_z).permute(0, 3, 2, 1)
        y_c_neg = (qk_ys @ v_y).permute(0, 3, 2, 1)
        y_c = torch.cat((y_c_pos, y_c_neg), dim=1)
        y_c = y_c[:, :, :, 0].unsqueeze(-1)
        y_C = y_c.expand(batchsize, self.out_dim, node, node).contiguous()
        y_R = y_C.permute(0, 1, 3, 2)
        y = y_C + y_R

        qk_z = q_z @ k_y / math.sqrt(node)
        qk_zs = q_z @ k_z / math.sqrt(node)
        qk_z = torch.softmax(qk_z, 3)
        qk_zs = torch.softmax(qk_zs, 3)
        z_c_pos = (qk_z @ v_y).permute(0, 3, 2, 1)
        z_c_neg = (qk_zs @ v_z).permute(0, 3, 2, 1)
        z_c = torch.cat((z_c_pos, z_c_neg), dim=1)
        z_c = z_c[:, :, :, 0].unsqueeze(-1)
        z_C = z_c.expand(batchsize, self.out_dim, node, node).contiguous()
        z_R = z_C.permute(0, 1, 3, 2)
        z = z_C + z_R

        se_y = torch.mean(y, (2, 3))
        se_y = self.sed_1(se_y)
        se_y = F.relu(se_y)
        se_y = self.seu_1(se_y)
        se_y = torch.sigmoid(se_y)
        se_y = se_y.unsqueeze(2).unsqueeze(3)

        y = y.mul(se_y)
        y = y + res_y

        se_z = torch.mean(z, (2, 3))
        se_z = self.sed_2(se_z)
        se_z = F.relu(se_z)
        se_z = self.seu_2(se_z)
        se_z = torch.sigmoid(se_z)
        se_z = se_z.unsqueeze(2).unsqueeze(3)

        z = z.mul(se_z)
        z = z + res_z

        return y, z

class EP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(EP, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, (1, node))
        nn.init.normal_(self.conv.weight, std=math.sqrt(4/(node*in_dim+out_dim)))

    def forward(self, x):
        x = self.conv(x)
        return x

class NP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NP, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, (node, 1))
        nn.init.normal_(self.conv.weight, std=math.sqrt(4/(node*in_dim+out_dim)))

    def forward(self, x):
        x = self.conv(x)
        return x

class MT_HPN(nn.Module):        # our main model, PB-HPC + PB-CAB
    def __init__(self, E2E_1, E2E_2, E2E_3, E2N, N2G):
        super(MT_HPN, self).__init__()

        whole = list(range(node))
        # This is the partition prior for AAL90 atlas and should be adjusted to match the corresponding prior of the selected atlas.
        frontal = list(range(20)) + list(range(22, 28)) + [30, 31, 32, 33] + [54, 55] + [68, 69]
        temporal = [20, 21] + list(range(36, 42)) + list(range(70, 76)) + list(range(78, 90))
        parietal = [34, 35] + list(range(56, 70))
        occipital = list(range(42, 56))
        insula = [28, 29]

        # This is the partition prior for Destrieux atlas and should be adjusted to match the corresponding prior of the selected atlas.
        #frontal = [0, 4, 5, 7, 11, 12, 13, 14, 15, 23, 28, 30, 31, 39, 40, 44, 45, 51, 52, 53, 61, 62, 63, 65, 67, 68,
        #           69, 74, 78, 79, 81, 85, 86, 87, 88, 89, 97, 102, 104, 105, 113, 114, 118, 119, 125, 126, 127, 135,
        #           136, 137, 139, 141, 142, 143]
        #temporal = [20, 22, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 49, 50, 71, 72, 73, 94, 96, 106, 107, 108, 109, 110,
        #            111, 112, 113, 114, 116, 123, 124, 145, 146, 147]
        #parietal = [2, 3, 8, 9, 24, 25, 26, 27, 29, 39, 40, 54, 55, 64, 66, 70, 76, 77, 82, 83, 98, 99, 100, 101, 103,
        #            113, 114, 128, 129, 138, 140, 144]
        #occipital = [1, 10, 18, 19, 20, 21, 41, 43, 56, 57, 58, 59, 60, 64, 75, 84, 92, 93, 94, 95, 115, 117, 130, 131,
        #             132, 133, 134, 138]
        #insula = [16, 17, 46, 47, 48, 90, 91, 120, 121, 122]

        self.node_list = [whole, frontal, temporal, parietal, occipital, insula]

        self.GPC_Cross_1 = MT_HGPC_SE(1, E2E_1, self.node_list)
        self.GPC_Cross_2 = MT_HGPC_SE(E2E_1, E2E_2, self.node_list)
        self.GPC_Cross_3 = MT_HGPC_SE(E2E_2, E2E_3, self.node_list)

        self.convN_1 = nn.Conv2d(E2E_3, E2N, (1, node))
        nn.init.normal_(self.convN_1.weight, std=math.sqrt(4 / (node * E2E_3 + E2N)))
        self.convN_2 = nn.Conv2d(E2E_3, E2N, (1, node))
        nn.init.normal_(self.convN_2.weight, std=math.sqrt(4 / (node * E2E_3 + E2N)))

        self.convG_1 = nn.Conv2d(E2N, N2G, (node, 1))
        nn.init.normal_(self.convG_1.weight, std=math.sqrt(4 / (node * E2N + N2G)))
        self.convG_2 = nn.Conv2d(E2N, N2G, (node, 1))
        nn.init.normal_(self.convG_2.weight, std=math.sqrt(4 / (node * E2N + N2G)))

        self.fc_1 = nn.Linear(N2G, 128)
        nn.init.constant_(self.fc_1.bias, 0)
        self.fc_2 = nn.Linear(N2G, 128)
        nn.init.constant_(self.fc_2.bias, 0)
        self.fc_3 = nn.Linear(128, 2)
        nn.init.constant_(self.fc_3.bias, 0)
        self.fc_4 = nn.Linear(128, 2)
        nn.init.constant_(self.fc_4.bias, 0)

    def forward(self, x):
        batchsize = x.size(0)

        y, z = self.GPC_Cross_1(x,x)
        y = F.relu(y)
        z = F.relu(z)

        y, z = self.GPC_Cross_2(y, z)
        y = F.relu(y)
        z = F.relu(z)

        y, z = self.GPC_Cross_3(y,z)
        y = F.relu(y)
        z = F.relu(z)

        y = self.convN_1(y)
        y = F.relu(y)
        z = self.convN_2(z)
        z = F.relu(z)

        y = self.convG_1(y)
        y = F.relu(y)
        z = self.convG_2(z)
        z = F.relu(z)

        y = y.view(batchsize, -1)
        z = z.view(batchsize, -1)

        y = self.fc_1(y)
        y = F.relu(y)
        z = self.fc_2(z)
        z = F.relu(z)

        y = self.fc_3(y)
        z = self.fc_4(z)

        return y, z

class MT_PCN(nn.Module):    # MT_HPN only base, BC-GCN + PB-CAB  homogeneous path convolution with multi-task structure
    def __init__(self, E2E_1, E2E_2, E2E_3, E2N, N2G):
        super(MT_PCN, self).__init__()

        self.GPC_Cross_1 = MT_GPC_SE(1, E2E_1)
        self.GPC_Cross_2 = MT_GPC_SE(E2E_1, E2E_2)
        self.GPC_Cross_3 = MT_GPC_SE(E2E_2, E2E_3)

        self.convN_1 = nn.Conv2d(E2E_3, E2N, (1, node))
        nn.init.normal_(self.convN_1.weight, std=math.sqrt(4 / (node * E2E_3 + E2N)))
        self.convN_2 = nn.Conv2d(E2E_3, E2N, (1, node))
        nn.init.normal_(self.convN_2.weight, std=math.sqrt(4 / (node * E2E_3 + E2N)))

        self.convG_1 = nn.Conv2d(E2N, N2G, (node, 1))
        nn.init.normal_(self.convG_1.weight, std=math.sqrt(4 / (node * E2N + N2G)))
        self.convG_2 = nn.Conv2d(E2N, N2G, (node, 1))
        nn.init.normal_(self.convG_2.weight, std=math.sqrt(4 / (node * E2N + N2G)))

        self.fc_1 = nn.Linear(N2G, 128)
        nn.init.constant_(self.fc_1.bias, 0)
        self.fc_2 = nn.Linear(N2G, 128)
        nn.init.constant_(self.fc_2.bias, 0)
        self.fc_3 = nn.Linear(128, 2)
        nn.init.constant_(self.fc_3.bias, 0)
        self.fc_4 = nn.Linear(128, 2)
        nn.init.constant_(self.fc_4.bias, 0)

    def forward(self, x):
        batchsize = x.size(0)

        y, z = self.GPC_Cross_1(x,x)
        y = F.relu(y)
        z = F.relu(z)

        y, z = self.GPC_Cross_2(y, z)
        y = F.relu(y)
        z = F.relu(z)

        y, z = self.GPC_Cross_3(y,z)
        y = F.relu(y)
        z = F.relu(z)

        y = self.convN_1(y)
        y = F.relu(y)
        z = self.convN_2(z)
        z = F.relu(z)

        y = self.convG_1(y)
        y = F.relu(y)
        z = self.convG_2(z)
        z = F.relu(z)

        y = y.view(batchsize, -1)
        z = z.view(batchsize, -1)

        y = self.fc_1(y)  # 性别
        y = F.relu(y)
        z = self.fc_2(z)  # 年龄
        z = F.relu(z)

        y = self.fc_3(y)  # 性别
        z = self.fc_4(z)  # 年龄

        return y, z

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        #nn.init.kaiming_normal_(m.weight, mode='fan_out')
		#nn.init.xavier_uniform_(m.weight)
		nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        #nn.init.constant_(m.bias, 0)

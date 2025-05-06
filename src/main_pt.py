# import hydra
# import numpy as np
# import torch
# import torch.nn.functional as F
# import torchvision.transforms as T
# from numpy.random import random
# from omegaconf import DictConfig
# from PIL import Image


# def lbp_torch(x):
#     #pad image for 3x3 mask size
#     x = F.pad(input=x, pad = [1, 1, 1, 1], mode='constant')
#     b=x.shape
#     M=b[1] # pylint: disable=invalid-name
#     N=b[2] # pylint: disable=invalid-name

#     y=x
#     #select elements within 3x3 mask
#     y00=y[:,0:M-2, 0:N-2]
#     y01=y[:,0:M-2, 1:N-1]
#     y02=y[:,0:M-2, 2:N  ]
#     #
#     y10=y[:,1:M-1, 0:N-2]
#     y11=y[:,1:M-1, 1:N-1]
#     y12=y[:,1:M-1, 2:N  ]
#     #
#     y20=y[:,2:M, 0:N-2]
#     y21=y[:,2:M, 1:N-1]
#     y22=y[:,2:M, 2:N ]

#     # Apply comparisons and multiplications
#     bit=torch.ge(y00,y11)
#     tmp=torch.mul(bit,torch.tensor(1))

#     bit=torch.ge(y01,y11)
#     val=torch.mul(bit,torch.tensor(2))
#     val=torch.add(val,tmp)

#     bit=torch.ge(y02,y11)
#     tmp=torch.mul(bit,torch.tensor(4))
#     val=torch.add(val,tmp)

#     bit=torch.ge(y12,y11)
#     tmp=torch.mul(bit,torch.tensor(8))
#     val=torch.add(val,tmp)

#     bit=torch.ge(y22,y11)
#     tmp=torch.mul(bit,torch.tensor(16))
#     val=torch.add(val,tmp)

#     bit=torch.ge(y21,y11)
#     tmp=torch.mul(bit,torch.tensor(32))
#     val=torch.add(val,tmp)

#     bit=torch.ge(y20,y11)
#     tmp=torch.mul(bit,torch.tensor(64))
#     val=torch.add(val,tmp)

#     bit=torch.ge(y10,y11)
#     tmp=torch.mul(bit,torch.tensor(128))
#     val=torch.add(val,tmp)

#     return val


# def uav_traj_gen(img_tensor, seq_len: int, patch_size: int):
#     _, height, width = img_tensor.shape
#     start_point = np.random.randint(0 + int(patch_size / 2) + 1, height - int(patch_size / 2) - 1)
#     end_point = np.random.randint(0 + int(patch_size / 2) + 1, height - int(patch_size / 2) - 1)
#     coord_heights = np.linspace(start_point, end_point, seq_len)
#     coord_widths = np.linspace(0 + int(patch_size / 2) + 1, width - int(patch_size / 2) - 1, seq_len)
#     coordinates = []
#     for i in range(seq_len):
#         coordinates.append((int(coord_widths[i]), int(coord_heights[i])))
#     return np.asarray(coordinates)


# def initialize_particles(img_tensor, particle_number: int, patch_size: int):
#     _, height, width = img_tensor.shape
#     particle_init_y = np.random.randint(0 + int(patch_size / 2) + 1, height - int(patch_size / 2) - 1,
#                                         size=particle_number)
#     particle_init_x = np.random.randint(0 + int(patch_size / 2) + 1, width - int(patch_size / 2) - 1,
#                                         size=particle_number)
#     particle_init = []
#     for i in range(particle_number):
#         particle_init.append([particle_init_x[i], particle_init_y[i]])
#     return np.asarray(particle_init)


# def get_patch_at_coords(img_tensor, point, patch_size: int):
#     x = point[0]
#     y = point[1]
#     patch = img_tensor[:, y - int(patch_size / 2):y + int(patch_size / 2) + 1,
#             x - int(patch_size / 2):x + int(patch_size / 2) + 1]
#     return patch


# def collect_particle_desriptors(img_tensor, coordinate_list, patch_size):
#     descriptors = []
#     for i, _ in enumerate(coordinate_list):
#         x = coordinate_list[i][0]
#         y = coordinate_list[i][1]
#         descriptors.append(img_tensor[:, y - int(patch_size / 2):y + int(patch_size / 2) + 1,
#                            x - int(patch_size / 2):x + int(patch_size / 2) + 1])
#     descriptors = torch.stack(descriptors)
#     return descriptors


# def match_patches_batch(descriptors: torch.Tensor, template: torch.Tensor, num_bins=16, lbp_bins=32):
#     """
#     descriptors: [N, C, H, W] - tensor patches (candidates)
#     template: [C, H, W]        - single patch (template)
#     """
#     N, _, _, _ = descriptors.shape # pylint: disable=invalid-name

#     template_batch = template.unsqueeze(0).expand(N, -1, -1, -1)

#     def compute_histograms(tensor, color_bins=num_bins, lbp_bins=lbp_bins):
#         """
#         tensor: [N, C, H, W]
#         Returns: [N, C*color_bins + lbp_bins]
#         """
#         N, C, _, _ = tensor.shape # pylint: disable=invalid-name
#         hist_list = []

#         for c in range(C):
#             ch = tensor[:, c, :, :].reshape(N, -1)
#             hist = torch.stack([torch.histc(ch[i], bins=color_bins, min=0.0, max=1.0) for i in range(N)])
#             hist = hist + 1e-6
#             hist = hist / hist.sum(dim=1, keepdim=True)
#             hist_list.append(hist)

#         gray = 0.114 * tensor[:, 0, :, :] + 0.587 * tensor[:, 1, :, :] + 0.299 * tensor[:, 2, :, :]
#         gray = gray.unsqueeze(1)

#         lbp_list = []
#         for i in range(N):
#             lbp = lbp_torch(gray[i])
#             lbp = lbp.flatten()
#             lbp_hist = torch.histc(lbp.float(), bins=lbp_bins, min=0, max=lbp_bins-1)
#             lbp_hist = lbp_hist + 1e-6
#             lbp_hist = lbp_hist / lbp_hist.sum()
#             lbp_list.append(lbp_hist)

#         lbp_hist_tensor = torch.stack(lbp_list, dim=0)
#         hist_list.append(lbp_hist_tensor)

#         return torch.cat(hist_list, dim=1)

#     desc_hist = compute_histograms(descriptors)
#     templ_hist = compute_histograms(template_batch)

#     diff = torch.abs(desc_hist - templ_hist)

#     scores = 1.0 - diff.sum(dim=1)

#     return scores

# def systematic_resample(weights):
#     length = len(weights)
#     positions = (random() + np.arange(length)) / length
#     indexes = np.zeros(length, 'i')
#     cumulative_sum = np.cumsum(weights)
#     i, j = 0, 0
#     while i < length:
#         if positions[i] < cumulative_sum[j]:
#             indexes[i] = j
#             i += 1
#         else:
#             j += 1
#     return indexes

# @hydra.main(config_path='configs', config_name='config_pt.yaml', version_base=None)
# def main(cfg: DictConfig):
#     # map_img = Image.open(cfg.map_path).convert("RGB")
#     # img_tensor = T.ToTensor()(map_img)

#     # uav_traj = uav_traj_gen(img_tensor, cfg.traj_len, cfg.patch_size)
#     uav_loc = 0
#     # uav_descriptor = get_patch_at_coords(img_tensor, uav_traj[uav_loc], cfg.patch_size)
#     uav_loc += 1

#     # particles = initialize_particles(img_tensor, cfg.particles_num, cfg.patch_size)
#     # descriptors = collect_particle_desriptors(img_tensor, particles, cfg.patch_size)


#     # scores = match_patches_batch(descriptors, uav_descriptor)


# if __name__ == "main_pt":
#     main()

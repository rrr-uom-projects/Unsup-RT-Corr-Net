from model.interpolation_net import *
from utils.arap import ArapInterpolationEnergy
from data.data import *
from utils.utils import k_fold_split_train_val_test, ParamBase, getDirs, getFiles
from neuromorph_imaging_loss.train import NetParam
import sys
import os
import pickle

class HypParam(ParamBase):
    def __init__(self):
        self.increase_thresh = 30

        self.method = "arap"
        self.in_mod = InterpolationModGeoEC

        self.load_dist_mat = True
        self.load_sub = True

def save_seq_collection_hard_correspondences(interpol, model_path, res_name):
    """Save test correspondences on a shape"""

    data_folder_out = os.path.join(model_path, "out")

    if not os.path.isdir(os.path.join(data_folder_out, res_name)):
        os.makedirs(os.path.join(data_folder_out, res_name), exist_ok=True)

    if not os.path.isdir(os.path.join(data_folder_out, res_name, "corrs")):
        os.makedirs(os.path.join(data_folder_out, res_name, "corrs"), exist_ok=True)

    with torch.no_grad():
        for i, data in enumerate(tqdm(interpol.val_loader)):
            shape_x = batch_to_shape(data["X"])
            shape_y = batch_to_shape(data["Y"])

            point_pred = interpol.interp_module.get_pred(shape_x, shape_y)
            point_pred = point_pred.cpu().numpy()

            corr_out = interpol.interp_module.match(shape_x, shape_y)
            assignment = corr_out.argmax(dim=1).cpu().numpy()
            assignmentinv = corr_out.argmax(dim=0).cpu().numpy()

            verts_x = shape_x.verts.cpu().numpy()
            verts_y = shape_y.verts.cpu().numpy()
            triangles_x = shape_x.triangles.cpu().numpy()
            triangles_y = shape_y.triangles.cpu().numpy()

            result = {}
            result["assignment"] = assignment
            result["assignmentinv"] = assignmentinv
            result["X"] = {"verts": verts_x, "triangles": triangles_x}
            result["Y"] = {"verts": verts_y, "triangles": triangles_y}
            result["inter_verts"] = point_pred
            result["fname_x"] = data["X"]["fname"]
            result["fname_y"] = data["Y"]["fname"]
            #result["corr_out"] = corr_out.cpu().numpy()
            
            with open(os.path.join(data_folder_out, res_name, "corrs/", f"seq_{i}.pkl"), "wb") as f:
                pickle.dump(result, f)

def create_interpol(dataset, dataset_val=None, folder_weights_load=None, time_stamp=None, param=NetParam(), hyp_param=None):
    if time_stamp is None:
        time_stamp = get_timestr()

    interpol_energy = ArapInterpolationEnergy()

    interpol_module = InterpolationModGeoEC(interpol_energy, param).to(device)

    preproc_mods = []
    
    settings_module = timestep_settings(increase_thresh=30)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False)

    interpol = InterpolNet(interpol_module, train_loader, val_loader=val_loader, time_stamp=time_stamp, preproc_mods=preproc_mods, settings_module=settings_module)

    if folder_weights_load is not None:
        interpol.load_self(save_path(folder_str=folder_weights_load))

    interpol.i_epoch = 0

    return interpol

def run_test(fold_num, lambd_imaging=1000):
    # find the model to load
    models_dir = ""
    model_name = list(filter(lambda x: f"imaging_loss_fold{fold_num}" in x, getDirs(models_dir)))[0]

    hyp_param = HypParam()

    data_folder = ""

    testing_datasets = []
    structs = ["brainstem", "mandible", "parotid_lt", "parotid_rt", "spinal_cord", "submandibular_lt", "submandibular_rt"]
    for struct in structs:
        # determine which meshes to use
        all_fnames = list(sorted(filter(lambda x: struct in x, getFiles(join(data_folder, "meshes/")))))
        _, _, test_inds = k_fold_split_train_val_test(dataset_size=len(all_fnames), fold_num=int(fold_num), seed=100457)
        test_fnames = [all_fnames[i].replace('.ply','') for i in test_inds]
        testing_datasets += [general_dataset(data_folder, fnames=test_fnames, load_dist_mat=True)]

    fused_dataset_test = torch.utils.data.ConcatDataset(testing_datasets)

    hyp_param.rot_mod = 0

    interpol = create_interpol(dataset=fused_dataset_test, dataset_val=fused_dataset_test, time_stamp=model_name, hyp_param=hyp_param)

    interpol.load_self(os.path.join(models_dir, model_name))

    interpol.interp_module.param.num_timesteps = 7
    
    res_name = f""
    save_seq_collection_hard_correspondences(interpol, os.path.join(models_dir, model_name), res_name)

if __name__ == "__main__":
    run_test(sys.argv[1])
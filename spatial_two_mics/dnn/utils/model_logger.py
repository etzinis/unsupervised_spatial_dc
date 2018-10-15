"""!
@brief Model logger in order to be able to load the model and test it
on different data.

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of illinois at Urbana Champaign
"""
import os
import sys
import torch
import datetime
import glob2

root_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '../../../')
sys.path.insert(0, root_dir)
from spatial_two_mics.config import MODELS_DIR


def save(model,
         optimizer,
         args,
         epoch,
         performance_dic,
         dataset_id,
         max_models_per_dataset=10):
    state = {
        'epoch': epoch,
        'val_performance': performance_dic,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'args': args
    }
    sdr_str = str(round(performance_dic['sdr'], 3))
    folder_name = os.path.join(MODELS_DIR, dataset_id)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    available_models = glob2.glob(folder_name + '/*.pt')

    if len(available_models) > max_models_per_dataset:
        sdr_and_model_path = [os.path.basename(path)
                              for path in available_models]
        sdr_and_model_path = [float(path.split("_")[1])
                              for path in sdr_and_model_path]
        sdr_and_model_path = zip(sdr_and_model_path, available_models)
        sdr_sorted_models = sorted(sdr_and_model_path,
                                   key=lambda x: x[0])[::-1]
        for sdr, path in sdr_sorted_models[max_models_per_dataset:]:
            try:
                os.remove(path)
            except:
                print("Error in removing {} ...".format(path))

    ts = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%s")
    filename = "SDR_{}_{}.pt".format(sdr_str, ts)
    file_path = os.path.join(folder_name, filename)
    torch.save(state, file_path)


def load(model,
         optimizer,
         dataset_id,
         filename=None):

    folder_name = os.path.join(MODELS_DIR, dataset_id)
    if filename is None:
        available_models = glob2.glob(folder_name + '/*.pt')
        file_path = os.path.join(folder_name, available_models[0])
    else:
        file_path = os.path.join(folder_name, filename)

    loaded_state = torch.load(file_path)
    model.load_state_dict(loaded_state['model_state'])
    optimizer.load_state_dict(loaded_state['optimizer_state'])
    epoch = loaded_state['epoch']
    val_performance = loaded_state['val_performance']
    args = loaded_state['args']

    return model, optimizer, epoch, val_performance, args

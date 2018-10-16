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
import torch.nn as nn

root_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '../../../')
sys.path.insert(0, root_dir)
from spatial_two_mics.config import MODELS_DIR
import spatial_two_mics.dnn.models.simple_LSTM_encoder as LSTM_builder


def save(model,
         optimizer,
         args,
         epoch,
         performance_dic,
         dataset_id,
         mean_tr,
         std_tr,
         max_models_per_dataset=20):
    state = {
        'epoch': epoch,
        'val_performance': performance_dic,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'args': args,
        'mean_tr': mean_tr,
        'std_tr': std_tr
    }
    sdr_str = str(round(performance_dic['sdr'], 3))
    sar_str = str(round(performance_dic['sar'], 3))
    sir_str = str(round(performance_dic['sir'], 3))
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
    filename = "SDR_{}_SIR_{}_SAR_{}_{}.pt".format(sdr_str,
                                                   sir_str,
                                                   sar_str,
                                                   ts)
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
    mean_tr = loaded_state['mean_tr']
    std_tr = loaded_state['std_tr']

    return (model, optimizer, epoch, val_performance,
            args, mean_tr, std_tr)


def load_and_create_the_model(model_path):

    loaded_state = torch.load(model_path)
    epoch = loaded_state['epoch']
    val_performance = loaded_state['val_performance']
    args = loaded_state['args']
    mean_tr = loaded_state['mean_tr']
    std_tr = loaded_state['std_tr']

    model = LSTM_builder.BLSTMEncoder(num_layers=args.n_layers,
                                      hidden_size=args.hidden_size,
                                      embedding_depth=args.embedding_depth,
                                      bidirectional=args.bidirectional,
                                      dropout=args.dropout)
    model = nn.DataParallel(model).cuda()

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate,
                                 betas=(0.9, 0.999))

    model.load_state_dict(loaded_state['model_state'])
    optimizer.load_state_dict(loaded_state['optimizer_state'])

    return (model, optimizer, epoch, val_performance,
            args, mean_tr, std_tr)

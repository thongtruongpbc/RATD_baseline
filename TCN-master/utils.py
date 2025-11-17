import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import os
import datetime

def pkl_load(fname):
    """Load pickle file"""
    with open(fname, 'rb') as f:
        return pickle.load(f)

def pad_nan_to_target(array, target_length, axis=0):
    """Pad array with NaN to target length"""
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array
    
    pad_width = [(0, 0)] * array.ndim
    pad_width[axis] = (0, pad_size)
    return np.pad(array, pad_width, mode='constant', constant_values=np.nan)

def init_dl_program(name, device_name, seed, debug=False):
    """Initialize deep learning program"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if device_name == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    return device_name

def name_with_datetime(prefix='output'):
    """Generate name with datetime"""
    now = datetime.datetime.now()
    return prefix + '_' + now.strftime('%Y%m%d_%H%M%S')

def pkl_save(obj, path):
    """Save object to pickle file"""
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def data_dropout(data, p=0.1):
    """Randomly drop data points"""
    mask = torch.rand_like(data) > p
    return data * mask

def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=20,
    foldername="",
):
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    if foldername != "":
        output_path = foldername + "/model.pth"

    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    best_valid_loss = 1e10
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()

                loss = model(train_batch)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
                if batch_no >= config["itr_per_epoch"]:
                    break

            lr_scheduler.step()
        
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            torch.save(model.state_dict(), output_path)

    if foldername != "":
        torch.save(model.state_dict(), output_path)

def evaluate(model, test_loader, nsample=100, scaler=1, mean_scaler=0, foldername=""):
    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0
        x=np.linspace(0,100,100)
        mse_list=np.zeros(100)
        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                output = model.evaluate(test_batch, nsample)

                samples, c_target, eval_points, observed_points, observed_time, cut_length = output
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)

                samples_median = samples.median(dim=1)
                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)
                
                mse_current = (
                    ((samples_median.values - c_target) * eval_points) ** 2
                ) * (scaler ** 2)
                mae_current = (
                    torch.abs((samples_median.values - c_target) * eval_points) 
                ) * scaler
                if batch_no<100:
                    mse_list[batch_no-1]=mse_current.sum().item()/eval_points.sum().item()
                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                evalpoints_total += eval_points.sum().item()

                it.set_postfix(
                    ordered_dict={
                        "rmse_total": np.sqrt(mse_total / evalpoints_total),
                        "mae_total": mae_total / evalpoints_total,
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )
            
            with open(
                foldername + "/generated_outputs_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                all_target = torch.cat(all_target, dim=0)
                all_evalpoint = torch.cat(all_evalpoint, dim=0)
                all_observed_point = torch.cat(all_observed_point, dim=0)
                all_observed_time = torch.cat(all_observed_time, dim=0)
                all_generated_samples = torch.cat(all_generated_samples, dim=0)

                pickle.dump(
                    [
                        all_generated_samples,
                        all_target,
                        all_evalpoint,
                        all_observed_point,
                        all_observed_time,
                    ],
                    f,
                )

        return np.sqrt(mse_total / evalpoints_total), mae_total / evalpoints_total

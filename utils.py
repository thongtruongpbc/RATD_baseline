import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

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
        accumulation_steps = 8
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                loss = model(train_batch)
                loss = loss / accumulation_steps
                loss.backward()
                avg_loss += loss.item()
                if (batch_no + 1) % accumulation_steps == 0:
                    
                    optimizer.step()
                    optimizer.zero_grad()
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

        best_valid_loss = float('inf')

        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0.0

            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        loss = model(valid_batch, is_train=0)
                        avg_loss_valid += loss.item()

                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )

            avg_loss_valid /= batch_no

            if avg_loss_valid < best_valid_loss:
                best_valid_loss = avg_loss_valid
                torch.save(model.state_dict(), output_path)
                print(f"\n[Best Model Updated] loss = {best_valid_loss:.6f} at epoch {epoch_no}")



def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )


def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))


def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):

    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)

def calc_quantile_CRPS_sum(target, forecast, eval_points, mean_scaler, scaler):

    eval_points = eval_points.mean(-1)
    target = target * scaler + mean_scaler
    target = target.sum(-1)
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = torch.quantile(forecast.sum(-1),quantiles[i],dim=1)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)

# def evaluate(model, test_loader, nsample=1, scaler=1, mean_scaler=0, foldername=""):
#     with torch.no_grad():
#         model.eval()
#         mse_total = 0.0
#         mae_total = 0.0
#         evalpoints_total = 0.0
#         device = 'cuda:0'
#         mse_list = torch.zeros(100, device=device)

#         with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
#             for batch_no, test_batch in enumerate(it, start=1):

#                 samples, target, eval_points, observed_points, observed_time = model.evaluate(
#                     test_batch, nsample
#                 )
#                 samples = samples.to(device)                # (B, nsample, L, K)
#                 target = target.to(device)                  # (B, L, K)
#                 eval_points = eval_points.to(device)        # (B, L, K)

#                 samples = samples.permute(0, 1, 3, 2)       # (B, nsample, K, L)
#                 target = target.permute(0, 2, 1)            # (B, K, L)
#                 eval_points = eval_points.permute(0, 2, 1)  # (B, K, L)

#                 samples_median = samples.median(dim=1).values  # (B, K, L)

#                 diff = (samples_median - target) * eval_points
#                 mse_current = (diff ** 2) * (scaler ** 2)
#                 mae_current = torch.abs(diff) * scaler
#                 mse_sum = mse_current.sum()
#                 mae_sum = mae_current.sum()
#                 eval_sum = eval_points.sum()

#                 mse_total += mse_sum
#                 mae_total += mae_sum
#                 evalpoints_total += eval_sum
    

                
#             with open(
#                 foldername + "/generated_outputs_nsample" + str(nsample) + ".pk", "wb"
#             ) as f:
#                 all_target = torch.cat(all_target, dim=0)
#                 all_evalpoint = torch.cat(all_evalpoint, dim=0)
#                 all_observed_point = torch.cat(all_observed_point, dim=0)
#                 all_observed_time = torch.cat(all_observed_time, dim=0)
#                 all_generated_samples = torch.cat(all_generated_samples, dim=0)

#                 pickle.dump(
#                     [
#                         all_generated_samples,
#                         all_target,
#                         all_evalpoint,
#                         all_observed_point,
#                         all_observed_time,
#                         scaler,
#                         mean_scaler,
#                     ],
#                     f,
#                 )

#             CRPS = calc_quantile_CRPS(
#                 all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
#             )
#             CRPS_sum = calc_quantile_CRPS_sum(
#                 all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
#             )

#             with open(
#                 foldername + "/result_nsample" + str(nsample) + ".pk", "wb"
#             ) as f:
#                 pickle.dump(
#                     [
#                         np.sqrt(mse_total / evalpoints_total),
#                         mae_total / evalpoints_total,
#                         CRPS,
#                     ],
#                     f,
#                 )
#                 print("RMSE:", np.sqrt((mse_total / evalpoints_total).cpu().item()))
#                 print("MAE:", mae_total / evalpoints_total)
#                 print("CRPS:", CRPS)
#                 print("CRPS_sum:", CRPS_sum)
    
#     rmse = np.sqrt(mse_total / evalpoints_total)
#     rmse = np.sqrt((mse_total / evalpoints_total).cpu().item())
#     mae = (mae_total / evalpoints_total).cpu().item()

#     if torch.is_tensor(CRPS):
#         CRPS = CRPS.cpu().item()

#     if torch.is_tensor(CRPS_sum):
#         CRPS_sum = CRPS_sum.cpu().item()


#     return rmse, mae, CRPS, CRPS_sum


def evaluate(model, test_loader, nsample=3, scaler=1, mean_scaler=0, foldername=""):
    with torch.no_grad():
        model.eval()
        mse_total = 0.0
        mae_total = 0.0
        evalpoints_total = 0.0
        device = 'cuda:0'
        mse_list = torch.zeros(100, device=device)

        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):

                samples, target, eval_points, observed_points, observed_time = model.evaluate(
                    test_batch, nsample
                )
                samples = samples.to(device)                # (B, nsample, L, K)
                target = target.to(device)                  # (B, L, K)
                eval_points = eval_points.to(device)        # (B, L, K)

                samples = samples.permute(0, 1, 3, 2)       # (B, nsample, K, L)
                target = target.permute(0, 2, 1)            # (B, K, L)
                eval_points = eval_points.permute(0, 2, 1)  # (B, K, L)

                samples_median = samples.median(dim=1).values  # (B, K, L)

                diff = (samples_median - target) * eval_points
                mse_current = (diff ** 2) * (scaler ** 2)
                mae_current = torch.abs(diff) * scaler
                mse_sum = mse_current.sum()
                mae_sum = mae_current.sum()
                eval_sum = eval_points.sum()

                mse_total += mse_sum
                mae_total += mae_sum
                evalpoints_total += eval_sum

    
    rmae = (mae_total / evalpoints_total).cpu().item()
    rmse = np.sqrt((mse_total / evalpoints_total).cpu().item())
    mae = (mae_total / evalpoints_total).cpu().item()
    mse = (mse_total / evalpoints_total).cpu().item()



    return mae, mse, rmse,  rmae

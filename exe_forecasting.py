import argparse
import torch
import datetime
import json
import yaml
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import csv
from main_model import RATD_Forecasting
from dataset_forecasting import get_dataloader
from utils import train, evaluate
import gc
gc.collect()
torch.cuda.empty_cache()

parser = argparse.ArgumentParser(description="RATD")
parser.add_argument("--config", type=str, default="base_forecasting.yaml")
parser.add_argument("--datatype", type=str, default="electricity")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=10)
# parser.add_argument("--target_dim", type=int, default=7)
parser.add_argument("--h_size", type=int, default=96)
parser.add_argument("--ref_size", type=int, default=168)

args = parser.parse_args()
print(args)
if args.datatype == 'electricity':
        target_dim = 321
elif args.datatype == 'weather':
        target_dim = 21
else:
        target_dim = 7
path = "/home/cds/ntthu/VARDiff/RATD_baseline/config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
config["diffusion"]["h_size"] = args.h_size
config["diffusion"]["ref_size"] = args.ref_size
print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = "./save/forecasting_" + args.datatype + '_' + current_time + "/"
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)


csv_path = 'results.csv'
def save_final_result(csv_path, mae, mse, rmse,  rmae, h_size, ref_size, datatype):
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["mae", "mse", "rmse",  "rmae", "h_size", "ref_size", "datatype"])
        writer.writerow([mae, mse, rmse,  rmae, h_size, ref_size, datatype])


train_loader, valid_loader, test_loader = get_dataloader(
    L = args.h_size,
    H = args.ref_size,
    device= args.device,
    datatype=args.datatype,
    batch_size=config["train"]["batch_size"],
)

model = RATD_Forecasting(config, args.device, target_dim).to(args.device)

if args.modelfolder == "":
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
    )
else:
    model.load_state_dict(torch.load("/data/RATD/save/" + args.modelfolder + "/model.pth"))
model.target_dim = target_dim
model.eval()
mae, mse, rmse,  rmae = evaluate(
    model,
    test_loader,
    nsample=args.nsample,
    scaler=1,
    mean_scaler=0,
    foldername=foldername,
)
del model
gc.collect()
save_final_result(csv_path, mae, mse, rmse,  rmae, args.h_size, args.ref_size, args.datatype)
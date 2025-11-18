import torch
import numpy as np
import datautils
from utils import init_dl_program, name_with_datetime, pkl_save, data_dropout
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import argparse
import yaml
from TCN.word_cnn.model import TCN_AE
import datautils

def all_retrieval(model, num, config, L, H):
    datatype = config['path']['datatype']
    vec_train_path =  f'/home/cds/ntthu/VARDiff/RATD_baseline/data/TCN/{datatype}_{L}_{H}_trainvec_list.pt'
    train_set = datautils.Dataset_retrieval(root_path=config["path"]["dataset_path"],flag='train',data_path=f'{args.datatype}.csv', size=[L, 0, H])
    val_set = datautils.Dataset_retrieval(root_path=config["path"]["dataset_path"],flag='val', data_path=f'{args.datatype}.csv', size=[L, 0, H])
    test_set = datautils.Dataset_retrieval(root_path=config["path"]["dataset_path"],flag='test',data_path=f'{args.datatype}.csv', size=[L, 0, H])

    all_repr=torch.load(vec_train_path)
    all_repr_len = len(train_set)
    train_references=[]
    val_references = []
    test_references = []
    ref_train_path =  f'/home/cds/ntthu/VARDiff/RATD_baseline/data/TCN/{datatype}_{L}_{H}_train_id_list.pt'
    ref_val_path =  f'/home/cds/ntthu/VARDiff/RATD_baseline/data/TCN/{datatype}_{L}_{H}_val_id_list.pt'
    ref_test_path = f'/home/cds/ntthu/VARDiff/RATD_baseline/data/TCN/{datatype}_{L}_{H}_test_id_list.pt'

    # train
    with torch.no_grad():
        for i in range(len(train_set)):
            x=train_set.data_x[i:i+L]
            
            x=x[np.newaxis, :, :]
            x=torch.tensor(x).to(config["retrieval"]["device"])
            x_vec = model.encode(x)
            k=x_vec.shape[-2]
            l=x_vec.shape[-1]
            x_vec=x_vec.reshape(1,k*l)
            all_repr = all_repr.reshape(all_repr_len,k*l)
            distances=torch.norm(x_vec.cpu() - all_repr,dim=1)
            _, idx=torch.topk(-1*distances, num)
            train_references.append(idx.int())
        train_references = torch.cat(train_references, dim=0)
        print(train_references.shape)
        torch.save(train_references, ref_train_path)

    #val
    with torch.no_grad():
        for i in range(len(val_set)):
            x=val_set.data_x[i:i+L]

            x=x[np.newaxis, :, :]
            x=torch.tensor(x).to(config["retrieval"]["device"])
            x_vec = model.encode(x)
            k=x_vec.shape[-2]
            l=x_vec.shape[-1]
            x_vec=x_vec.reshape(1,k*l)
            all_repr = all_repr.reshape(all_repr_len,k*l)
            distances=torch.norm(x_vec.cpu() - all_repr,dim=1)
            _, idx=torch.topk(-1*distances, num)
            val_references.append(idx.int())
        val_references = torch.cat(val_references, dim=0)
        print(val_references.shape)
        torch.save(val_references, ref_val_path)

    #test
    with torch.no_grad():
        for i in range(len(test_set)):
            x=test_set.data_x[i:i+L]
            x=x[np.newaxis, :, :]
            x=torch.tensor(x).to(config["retrieval"]["device"])
            x_vec = model.encode(x)
            k=x_vec.shape[-2]
            l=x_vec.shape[-1]
            x_vec=x_vec.reshape(1,k*l)
            all_repr = all_repr.reshape(all_repr_len, k*l)
            distances=torch.norm(x_vec.cpu() - all_repr,dim=1)
            _, idx=torch.topk(-1*distances, num)
            test_references.append(idx.int())
        test_references = torch.cat(test_references, dim=0)
        print(test_references.shape)
        torch.save(test_references, ref_test_path)


    # return train_references

def all_encode(model,config, L, H):
    train_vec_list=[]
    datatype = config['path']['datatype']
    train_set = datautils.Dataset_retrieval(root_path=config["path"]["dataset_path"],flag='train',data_path=f'{datatype}.csv', size=[L, 0, H])
    
    vec_train_path =  f'/home/cds/ntthu/VARDiff/RATD_baseline/data/TCN/{datatype}_{L}_{H}_trainvec_list.pt'
    # train
    with torch.no_grad():
        for i in range(len(train_set)):
            x=train_set.data_x[i:i+L]
            y=train_set.data_x[i+L:i+L+H]
            x=x[np.newaxis, :, :]
            x=torch.tensor(x).to(config["retrieval"]["device"])
            x = x.float()
            x_vec = model.encode(x)
            train_vec_list.append(x_vec.cpu())
    train_vec_list = torch.cat(train_vec_list, dim=0)
    torch.save(train_vec_list.float(), vec_train_path)

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0

    for x in loader:
        x = x.to(device).float()       # shape: (B, seq_len, input_size)

        optimizer.zero_grad()
        output = model(x.float())      # shape: (B, output_size)
        #print(x.dtype, output.dtype)
        loss = criterion(output.float(), x)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)


def validation(model, loader, criterion, device):
    model.eval()
    loss_total = 0

    with torch.no_grad():
        for x in loader:
            x = x.to(device)

            output = model(x)
            loss = criterion(output, x)
            loss_total += loss.item()

    return loss_total / len(loader)
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CSDI")
    parser.add_argument("--datatype", type=str, default="electricity")
    parser.add_argument("--modelfolder", type=str, default="")
    parser.add_argument("--L", type=int, default=720, help='history length')
    parser.add_argument("--H", type=int, default=720, help='pred length')
    parser.add_argument(
        "--type", type=str, default="encode", choices=["encode", "retrieval"]
    )
    parser.add_argument("--encoder", default="TCN")

    args = parser.parse_args()
    L=args.L  # his_len
    H=args.H # hoziron pred_len
    print(args)

    path = "config/" + f"retrieval_{args.datatype}.yaml"
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    config["retrieval"]["encoder"] = args.encoder
    #setting
    num_epochs = 30
    batch_size = 32
    criterion = torch.nn.MSELoss()
        
    model = TCN_AE(
            input_size=config["retrieval"]["dim"],
            latent_size=config["retrieval"]["latent_size"], num_channels=[config["retrieval"]["dim"]] * (config["retrieval"]["level"]) + [config["retrieval"]["dim"]],
        ).to(config["retrieval"]["device"])
    model.float()
        #model=torch.load( config["path"]["encoder_path"])
    if args.type == 'encode':
        # Load data
        
        
        train_set = datautils.Dataset_TCN(root_path=config["path"]["dataset_path"],flag='train', data_path=f'{args.datatype}.csv',size=[L, 0, H])
        val_set = datautils.Dataset_TCN(root_path=config["path"]["dataset_path"],flag='val', data_path=f'{args.datatype}.csv',size=[L, 0, H])

        train_loader = DataLoader(
            train_set, batch_size = batch_size, shuffle=1)
        val_loader = DataLoader(
            val_set, batch_size = batch_size, shuffle=0)

        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(num_epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device = 'cuda:0')
            val_loss = validation(model, val_loader, criterion, device = 'cuda:0')
            print(f"Epoch {epoch+1}/{num_epochs}  Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}")

        torch.save(model.state_dict(), f"{args.datatype}_{L}_{H}_tcn_model.pth")

        # all encode
        all_encode(model,config, L, H)
    if args.type == 'retrieval':
        model.load_state_dict(torch.load(f"{args.datatype}_{L}_{H}_tcn_model.pth", map_location='cuda:0'))
        model.to('cuda:0')
        model.eval()
        all_retrieval(model, config["retrieval"]["level"], config, L, H)


    

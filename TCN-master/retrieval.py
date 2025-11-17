import torch
import numpy as np
from TCN.word_cnn.model import TCN
import datautils
from utils import init_dl_program, name_with_datetime, pkl_save, data_dropout

import torch
import numpy as np
import argparse
import yaml
from TCN.word_cnn.model import TCN
import datautils

def all_retrieval(model, num, config):
    x=torch.from_numpy(x).to(config["retrieval"]["device"])
    L=config["retrieval"]["L"]
    H=config["retrieval"]["H"]
    train_set = datautils.Dataset_Electricity(root_path=config["path"]["dataset_path"],flag='train',size=[L, 0, H])
    val_set = datautils.Dataset_Electricity(root_path=config["path"]["dataset_path"],flag='val',size=[L, 0, H])
    test_set = datautils.Dataset_Electricity(root_path=config["path"]["dataset_path"],flag='test',size=[L, 0, H])

    all_repr=torch.load(config["path"]["vec_train_path"])
    train_references=[]
    val_references = []
    test_references = []

    # train
    with torch.no_grad():
        for i in range(len(train_set) - L - H + 1):
            x=train_set.data_x[i:i+L]
            
            x=x[np.newaxis, :, :]
            x=torch.tensor(x).transpose(1, 2).to(config["retrieval"]["device"])
            x_vec = model.encode(x)
            k=x_vec.shape[-2]
            l=x_vec.shape[-1]
            x_vec=x_vec.reshape(1,k*l)
            all_repr = all_repr.reshape(config["retrieval"]["length"],k*l)
            distances=torch.norm(x_vec.cpu() - all_repr,dim=1)
            _, idx=torch.topk(-1*distances, num)
            train_references.append(idx.int())
        train_references = torch.cat(train_references, dim=0)
        print(train_references.shape)
        torch.save(train_references, config["path"]["ref_train_path"])

    #val
    with torch.no_grad():
        for i in range(len(val_set) - L - H + 1):
            x=val_set.data_x[i:i+L]

            x=x[np.newaxis, :, :]
            x=torch.tensor(x).transpose(1, 2).to(config["retrieval"]["device"])
            x_vec = model.encode(x)
            k=x_vec.shape[-2]
            l=x_vec.shape[-1]
            x_vec=x_vec.reshape(1,k*l)
            all_repr = all_repr.reshape(config["retrieval"]["length"],k*l)
            distances=torch.norm(x_vec.cpu() - all_repr,dim=1)
            _, idx=torch.topk(-1*distances, num)
            val_references.append(idx.int())
        val_references = torch.cat(val_references, dim=0)
        print(val_references.shape)
        torch.save(val_references, config["path"]["ref_val_path"])

    #test
    with torch.no_grad():
        for i in range(len(test_set) - L - H + 1):
            x=test_set.data_x[i:i+L]
            x=x[np.newaxis, :, :]
            x=torch.tensor(x).transpose(1, 2).to(config["retrieval"]["device"])
            x_vec = model.encode(x)
            k=x_vec.shape[-2]
            l=x_vec.shape[-1]
            x_vec=x_vec.reshape(1,k*l)
            all_repr = all_repr.reshape(config["retrieval"]["length"],k*l)
            distances=torch.norm(x_vec.cpu() - all_repr,dim=1)
            _, idx=torch.topk(-1*distances, num)
            test_references.append(idx.int())
        test_references = torch.cat(test_references, dim=0)
        print(test_references.shape)
        torch.save(test_references, config["path"]["ref_test_path"])


    # return train_references

def all_encode(model,config):
    train_vec_list=[]
    val_vec_list=[]
    test_vec_list=[]
    reference_list=[]
    L=config["retrieval"]["L"]
    H=config["retrieval"]["H"]
    train_set = datautils.Dataset_Electricity(root_path=config["path"]["dataset_path"],flag='train',size=[L, 0, H])

    # train
    with torch.no_grad():
        for i in range(len(train_set) - L - H + 1):
            x=train_set.data_x[i:i+L]
            y=train_set.data_x[i+L:i+L+H]
            x=x[np.newaxis, :, :]
            x=torch.tensor(x).transpose(1, 2).to(config["retrieval"]["device"])
            x = x.float()
            x_vec = model.encode(x)
            train_vec_list.append(x_vec.cpu())
    train_vec_list = torch.cat(train_vec_list, dim=0)
    torch.save(train_vec_list.float(), config["path"]["vec_train_path"])

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CSDI")
    parser.add_argument("--config", type=str, default="retrieval_ele.yaml")
    parser.add_argument("--modelfolder", type=str, default="")
    parser.add_argument(
        "--type", type=str, default="encode", choices=["encode", "retrieval"]
    )
    parser.add_argument("--encoder", default="TCN")

    args = parser.parse_args()
    print(args)

    path = "config/" + args.config
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    config["retrieval"]["encoder"] = args.encoder
    model = TCN(
            input_size=config["retrieval"]["length"],
            output_size=config["retrieval"]["length"], num_channels=[config["retrieval"]["length"]] * (config["retrieval"]["level"]) + [config["retrieval"]["length"]],
        ).to(config["retrieval"]["device"])
    #model=torch.load( config["path"]["encoder_path"])
    if args.type == 'encode':
        all_encode(model,config)
    if args.type == 'retrieval':
        all_retrieval(model,config)


    

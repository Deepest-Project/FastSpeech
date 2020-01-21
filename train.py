import os, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import Model
import hparams
from text import *
from torch.utils.tensorboard import SummaryWriter
from loss import TransformerLoss
from plot_image import plot_image
from utils import prepare_dataloaders, save_checkpoint, lr_scheduling


def validate(model, criterion, val_loader, iteration, writer):
    model.eval()
    with torch.no_grad():
        n_data, val_loss = 0, 0
        for i, batch in enumerate(val_loader):
            n_data += len(batch[0])
            text_padded, text_lengths, mel_padded, mel_lengths, align_padded = [
                x.cuda() for x in batch
            ]
            mel_out, durations, durations_out = model(text_padded,
                                                      align_padded,
                                                      text_lengths,
                                                      mel_lengths)
            mel_loss, duration_loss = criterion((mel_out, durations_out),
                                                (mel_padded, durations))
            val_loss += (mel_loss+duration_loss).item()*len(batch[0])
            
        val_loss /= n_data

    writer.add_scalar('val_loss', val_loss,
                      global_step=iteration//hparams.accumulation)
    
    fig = plot_image(mel_padded,
                     mel_out,
                     align_padded,
                     text_padded,
                     mel_lengths, 
                     text_lengths)
    writer.add_figure('Validation plots', fig,
                      global_step=iteration//hparams.accumulation)
    writer.flush()
    
    model.train()
    
    
def main():
    train_loader, val_loader, collate_fn = prepare_dataloaders(hparams)

    model = Model(hparams).cuda()
    
    if hparams.pretrained_embedding==True:
        state_dict = torch.load(hparams.teacher_model)['state_dict']

        for k, v in state_dict.items():
            if k=='alpha1':
                model.alpha.data = v

            if 'Embedding' in k:
                setattr(model, k, v)

            if 'Prenet_E' in k:
                setattr(model, k, v)

            if 'Encoder' in k:
                setattr(model, k, v)
        
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=hparams.lr,
                                 betas=(0.9, 0.98),
                                 eps=1e-09)
    
    criterion = TransformerLoss()


    logging_path=f'{hparams.output_directory}/{hparams.log_directory}'
    if os.path.exists(logging_path):
        raise Exception('The experiment already exists')
    else:
        os.mkdir(logging_path)
        writer = SummaryWriter(logging_path)


    iteration, loss = 0, 0
    model.train()
    print("Training Start!!!")
    while iteration < (hparams.train_steps*hparams.accumulation):
        for i, batch in enumerate(train_loader):
            text_padded, text_lengths, mel_padded, mel_lengths, align_padded = [
                x.cuda() for x in batch
            ]
            mel_out, durations, durations_out = model(text_padded,
                                                      align_padded,
                                                      text_lengths,
                                                      mel_lengths)
            mel_loss, duration_loss = criterion((mel_out, durations_out),
                                                (mel_padded, durations))
            sub_loss = (mel_loss+duration_loss)/4
            sub_loss.backward()
            loss = loss+sub_loss.item()

            iteration += 1
            if iteration%hparams.accumulation == 0:
                lr_scheduling(optimizer, iteration//hparams.accumulation)
                torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.grad_clip_thresh)
                optimizer.step()
                model.zero_grad()
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'],
                                  global_step=iteration//hparams.accumulation)
                writer.add_scalar('mel_loss', mel_loss.item(),
                                  global_step=iteration//hparams.accumulation)
                writer.add_scalar('duration_loss', duration_loss.item(),
                                  global_step=iteration//hparams.accumulation)
                loss=0


            if iteration%(hparams.iters_per_plot*hparams.accumulation)==0:
                fig = plot_image(mel_padded,
                                 mel_out,
                                 align_padded,
                                 text_padded,
                                 mel_lengths,
                                 text_lengths)
                writer.add_figure('Train plots', fig,
                                  global_step=iteration//hparams.accumulation)
                writer.flush()


            if iteration%(hparams.iters_per_checkpoint*hparams.accumulation)==0:
                save_checkpoint(model,
                                optimizer,
                                hparams.lr,
                                iteration//hparams.accumulation,
                                filepath=logging_path)


            if iteration==(hparams.train_steps*hparams.accumulation):
                break

        validate(model, criterion, val_loader, iteration, writer)

    
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=str, default='0')
    args = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)
    
    main()
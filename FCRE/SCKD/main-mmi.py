import argparse
import random
from sampler import data_sampler
from config import Config
import torch
from model.bert_encoder import Bert_EncoderMLM
from model.dropout_layer import Dropout_Layer
from model.classifier import Softmax_Layer, Proto_Softmax_Layer
from data_loader import get_data_loader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.cluster import KMeans
import collections
from copy import deepcopy
import os
from typing import List
import utils
from mixup import mixup_data_augmentation
from add_loss import MultipleNegativesRankingLoss, SupervisedSimCSELoss, ContrastiveLoss, NegativeCosSimLoss
from torch.nn.utils import clip_grad_norm_
import logging
import sys
from sam import SAM
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def train_simple_model(config, encoder, dropout_layer, classifier, training_data, epochs, map_relid2tempid):
    data_loader = get_data_loader(config, training_data, shuffle=True)

    encoder.train()
    dropout_layer.train()
    classifier.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': encoder.parameters(), 'lr': 0.00001},
        {'params': dropout_layer.parameters(), 'lr': 0.00001},
        {'params': classifier.parameters(), 'lr': 0.001}
    ])
    if config.SAM:
        base_optimizer = optim.Adam
        optimizer = SAM(
            params=[{'params': encoder.parameters(), 'lr': 0.00001},
                    {'params': dropout_layer.parameters(), 'lr': 0.00001},
                    {'params': classifier.parameters(), 'lr': 0.001}],
            base_optimizer=optim.Adam,  # Pass the Adam class, not an instance
            rho=config.rho,
            adaptive=True
        )
    for epoch_i in range(epochs):
        losses = []
        for step, batch_data in enumerate(data_loader):
            optimizer.zero_grad()
            labels, _, tokens = batch_data
            for k in tokens.keys():
                tokens[k] = tokens[k].to(config.device) # B, {''ids', 'mask'}
            labels = labels.to(config.device)
            labels = [map_relid2tempid[x.item()] for x in labels]
            labels = torch.tensor(labels).to(config.device)
            
            # tokens = torch.stack([x.to(config.device) for x in tokens],dim=0)
            reps, mask_output = encoder(tokens)
            
            reps, _ = dropout_layer(reps)
            logits = classifier(reps)
            loss = criterion(logits, labels)

            if not config.SAM:
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
            else:
                loss.backward()
                optimizer.first_step(zero_grad=True)
                
                # tokens = torch.stack([x.to(config.device) for x in tokens],dim=0)
                reps, mask_output = encoder(tokens)
                
                reps, _ = dropout_layer(reps)
                logits = classifier(reps)
                loss = criterion(logits, labels)
                losses.append(loss.item())
                
                loss.backward()
                optimizer.second_step(zero_grad=True)
        print(f"loss is {np.array(losses).mean()}")


def compute_jsd_loss(m_input):
    # m_input: the result of m times dropout after the classifier.
    # size: m*B*C
    m = m_input.shape[0]
    mean = torch.mean(m_input, dim=0)
    jsd = 0
    for i in range(m):
        loss = F.kl_div(F.log_softmax(mean, dim=-1), F.softmax(m_input[i], dim=-1), reduction='none')
        loss = loss.sum()
        jsd += loss / m
    return jsd


def contrastive_loss(hidden, labels):

    logsoftmax = nn.LogSoftmax(dim=-1)

    return -(logsoftmax(hidden) * labels).sum() / labels.sum()


def construct_hard_triplets(output, labels, relation_data):
    positive = []
    negative = []
    pdist = nn.PairwiseDistance(p=2)
    for rep, label in zip(output, labels):
        positive_relation_data = relation_data[label.item()]
        negative_relation_data = []
        for key in relation_data.keys():
            if key != label.item():
                negative_relation_data.extend(relation_data[key])
        positive_distance = torch.stack([pdist(rep.cpu(), p) for p in positive_relation_data])
        negative_distance = torch.stack([pdist(rep.cpu(), n) for n in negative_relation_data])
        positive_index = torch.argmax(positive_distance)
        negative_index = torch.argmin(negative_distance)
        positive.append(positive_relation_data[positive_index.item()])
        negative.append(negative_relation_data[negative_index.item()])


    return positive, negative


def train_first(config, encoder, dropout_layer, classifier, training_data, epochs, map_relid2tempid, new_relation_data):
    data_loader = get_data_loader(config, training_data, shuffle=True)

    encoder.train()
    dropout_layer.train()
    classifier.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': encoder.parameters(), 'lr': 0.00001},
        {'params': dropout_layer.parameters(), 'lr': 0.00001},
        {'params': classifier.parameters(), 'lr': 0.001}
    ])
    if config.SAM:
        base_optimizer = optim.Adam
        optimizer = SAM(
            params=[{'params': encoder.parameters(), 'lr': 0.00001},
                    {'params': dropout_layer.parameters(), 'lr': 0.00001},
                    {'params': classifier.parameters(), 'lr': 0.001}],
            base_optimizer=optim.Adam,  # Pass the Adam class, not an instance
            rho=config.rho,
            adaptive=True
        )
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
    for epoch_i in range(epochs):
        losses = []
        for step, (labels, _, tokens) in enumerate(data_loader):
            for k in tokens.keys():
                tokens[k] = tokens[k].to(config.device) # B, {''ids', 'mask'}
            optimizer.zero_grad()

            logits_all = []
            # tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
            labels = labels.to(config.device)
            origin_labels = labels[:]
            labels = [map_relid2tempid[x.item()] for x in labels]
            labels = torch.tensor(labels).to(config.device)
            reps, mask_output = encoder(tokens)
            outputs,_ = dropout_layer(reps)
            positives,negatives = construct_hard_triplets(outputs, origin_labels, new_relation_data)

            for _ in range(config.f_pass):
                output, output_embedding = dropout_layer(reps)
                logits = classifier(output)
                logits_all.append(logits)

            positives = torch.cat(positives, 0).to(config.device)
            negatives = torch.cat(negatives, 0).to(config.device)
            anchors = outputs
            logits_all = torch.stack(logits_all)
            m_labels = labels.expand((config.f_pass, labels.shape[0]))  # m,B
            loss1 = criterion(logits_all.reshape(-1, logits_all.shape[-1]), m_labels.reshape(-1))
            loss2 = compute_jsd_loss(logits_all)
            tri_loss = triplet_loss(anchors, positives, negatives)
            loss = loss1 + loss2 + tri_loss

            if not config.SAM:
                loss.backward()
                losses.append(loss.item())
                optimizer.step()
                
            else:
                loss.backward()
                optimizer.first_step(zero_grad=True)
                reps, mask_output = encoder(tokens)
                outputs,_ = dropout_layer(reps)
                positives,negatives = construct_hard_triplets(outputs, origin_labels, new_relation_data)

                for _ in range(config.f_pass):
                    output, output_embedding = dropout_layer(reps)
                    logits = classifier(output)
                    logits_all.append(logits)

                positives = torch.cat(positives, 0).to(config.device)
                negatives = torch.cat(negatives, 0).to(config.device)
                anchors = outputs
                logits_all = torch.stack(logits_all)
                m_labels = labels.expand((config.f_pass, labels.shape[0]))  # m,B
                loss1 = criterion(logits_all.reshape(-1, logits_all.shape[-1]), m_labels.reshape(-1))
                loss2 = compute_jsd_loss(logits_all)
                tri_loss = triplet_loss(anchors, positives, negatives)
                loss = loss1 + loss2 + tri_loss
                
                losses.append(loss.item())
                loss.backward()
                optimizer.second_step(zero_grad=True)
                
        print(f"loss is {np.array(losses).mean()}")


def train_mem_model(config, encoder, dropout_layer, classifier, training_data, epochs, map_relid2tempid, new_relation_data,
                prev_encoder, prev_dropout_layer, prev_classifier, prev_relation_index , prototype = None ):
    data_loader = get_data_loader(config, training_data, shuffle=True)

    encoder.train()
    dropout_layer.train()
    classifier.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': encoder.parameters(), 'lr': 0.00001},
        {'params': dropout_layer.parameters(), 'lr': 0.00001},
        {'params': classifier.parameters(), 'lr': 0.001}
    ])
    if config.SAM:
        base_optimizer = optim.Adam
        optimizer = SAM(
            params=[{'params': encoder.parameters(), 'lr': 0.00001},
                    {'params': dropout_layer.parameters(), 'lr': 0.00001},
                    {'params': classifier.parameters(), 'lr': 0.001}],
            base_optimizer=optim.Adam,  # Pass the Adam class, not an instance
            rho=0.05,
            adaptive=True
        )
        
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
    distill_criterion = nn.CosineEmbeddingLoss()
    softmax = nn.Softmax(dim=0)
    T = config.kl_temp
    for epoch_i in range(epochs):
        losses = []
        for step, (labels, _, tokens) in enumerate(data_loader):
            for k in tokens.keys():
                tokens[k] = tokens[k].to(config.device) # B, {''ids', 'mask'}
            optimizer.zero_grad()

            logits_all = []
            # tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
            labels = labels.to(config.device)
            origin_labels = labels[:]
            mlm_labels = labels[:]
            mlm_labels = mlm_labels + config.vocab_size + config.marker_size # token [REL{i}]
            
            labels = [map_relid2tempid[x.item()] for x in labels]
            labels = torch.tensor(labels).to(config.device)
            reps , mask_output = encoder(tokens)
            normalized_reps_emb = F.normalize(reps.view(-1, reps.size()[1]), p=2, dim=1)
            outputs,_ = dropout_layer(reps)
            if prev_dropout_layer is not None:
                prev_outputs, _ = prev_dropout_layer(reps)
                positives,negatives = construct_hard_triplets(prev_outputs, origin_labels, new_relation_data)
            else:
                positives, negatives = construct_hard_triplets(outputs, origin_labels, new_relation_data)

            for _ in range(config.f_pass):
                output, output_embedding = dropout_layer(reps)
                logits = classifier(output)
                logits_all.append(logits)

            positives = torch.cat(positives, 0).to(config.device)
            negatives = torch.cat(negatives, 0).to(config.device)
            anchors = outputs
            logits_all = torch.stack(logits_all)
            m_labels = labels.expand((config.f_pass, labels.shape[0]))  # m,B
            loss1 = criterion(logits_all.reshape(-1, logits_all.shape[-1]), m_labels.reshape(-1))
            loss2 = compute_jsd_loss(logits_all)
            tri_loss = triplet_loss(anchors, positives, negatives)
            #compute infoNCE loss
            infoNCE_loss = 0
            for i in range(output.shape[0]):
                neg_prototypes = [prototype[rel_id] for rel_id in prototype.keys() if rel_id != origin_labels[i].item()]
                neg_prototypes = torch.stack(neg_prototypes).to(config.device)
                
                #--- prepare batch of negative samples 
                neg_prototypes.requires_grad_ = False
                neg_prototypes = neg_prototypes.squeeze()
                f_pos = encoder.infoNCE_f(mask_output[i],outputs[i])
                f_neg = encoder.infoNCE_f(mask_output[i],neg_prototypes )

                

                f_concat = torch.cat([f_pos,f_neg.squeeze()],dim=0)
                # quick fix for large number
                f_concat = torch.log(torch.max(f_concat, torch.tensor(1e-9).to(config.device)))

                infoNCE_loss += -torch.log(softmax(f_concat)[0])
                #--- prepare batch of negative samples  
            infoNCE_loss /= output.shape[0]           
            # compute MLM loss
            
            mlm_loss = criterion(mask_output.view(-1, mask_output.size()[-1]), mlm_labels.view(-1))
            
            
            loss = loss1 + loss2 + tri_loss + config.infonce_lossfactor*infoNCE_loss + config.mlm_lossfactor*mlm_loss
            # wandb.log({"loss1": loss1, "loss2": loss2, "tri_loss": tri_loss, "infoNCE_loss": infoNCE_loss , "mlm_loss": mlm_loss})
            
            if prev_encoder is not None:
                prev_reps,_ = prev_encoder(tokens)
                prev_reps.detach()
                normalized_prev_reps_emb = F.normalize(prev_reps.view(-1, prev_reps.size()[1]), p=2, dim=1)

                feature_distill_loss = distill_criterion(normalized_reps_emb, normalized_prev_reps_emb,
                                                         torch.ones(tokens['ids'].size(0)).to(
                                                             config.device))
                loss += feature_distill_loss

            if prev_dropout_layer is not None and prev_classifier is not None:
                prediction_distill_loss = None
                dropout_output_all = []
                prev_dropout_output_all = []
                for i in range(config.f_pass):
                    output, _ = dropout_layer(reps)
                    prev_output, _ = prev_dropout_layer(reps)
                    dropout_output_all.append(output)
                    prev_dropout_output_all.append(output)
                    pre_logits = prev_classifier(output).detach()

                    pre_logits = F.softmax(pre_logits.index_select(1, prev_relation_index) / T, dim=1)

                    log_logits = F.log_softmax(logits_all[i].index_select(1, prev_relation_index) / T, dim=1)
                    if i == 0:
                        prediction_distill_loss = -torch.mean(torch.sum(pre_logits * log_logits, dim=1))
                    else:
                        prediction_distill_loss += -torch.mean(torch.sum(pre_logits * log_logits, dim=1))

                prediction_distill_loss /= config.f_pass
                loss += prediction_distill_loss
                dropout_output_all = torch.stack(dropout_output_all)
                prev_dropout_output_all = torch.stack(prev_dropout_output_all)
                mean_dropout_output_all = torch.mean(dropout_output_all, dim=0)
                mean_prev_dropout_output_all = torch.mean(prev_dropout_output_all,dim=0)
                normalized_output = F.normalize(mean_dropout_output_all.view(-1, mean_dropout_output_all.size()[1]), p=2, dim=1)
                normalized_prev_output = F.normalize(mean_prev_dropout_output_all.view(-1, mean_prev_dropout_output_all.size()[1]), p=2, dim=1)
                hidden_distill_loss = distill_criterion(normalized_output, normalized_prev_output,
                                                         torch.ones(tokens['ids'].size(0)).to(
                                                             config.device))
                loss += hidden_distill_loss
            
            print(f"loss1 is {loss1.item()}, loss2 is {loss2.item()}, tri_loss is {tri_loss.item()}, infoNCE_loss is {infoNCE_loss.item()}, mlm_loss is {mlm_loss.item()}")
            if not config.SAM:
                loss.backward()
                losses.append(loss.item())
                optimizer.step()
            else:
                
                loss.backward()
                optimizer.first_step(zero_grad=True)
                logits_all = []
                
                reps , mask_output = encoder(tokens)
                normalized_reps_emb = F.normalize(reps.view(-1, reps.size()[1]), p=2, dim=1)
                outputs,_ = dropout_layer(reps)
                if prev_dropout_layer is not None:
                    prev_outputs, _ = prev_dropout_layer(reps)
                    positives,negatives = construct_hard_triplets(prev_outputs, origin_labels, new_relation_data)
                else:
                    positives, negatives = construct_hard_triplets(outputs, origin_labels, new_relation_data)

                for _ in range(config.f_pass):
                    output, output_embedding = dropout_layer(reps)
                    logits = classifier(output)
                    logits_all.append(logits)

                positives = torch.cat(positives, 0).to(config.device)
                negatives = torch.cat(negatives, 0).to(config.device)
                anchors = outputs
                logits_all = torch.stack(logits_all)
                m_labels = labels.expand((config.f_pass, labels.shape[0]))  # m,B
                loss1 = criterion(logits_all.reshape(-1, logits_all.shape[-1]), m_labels.reshape(-1))
                loss2 = compute_jsd_loss(logits_all)
                tri_loss = triplet_loss(anchors, positives, negatives)
                #compute infoNCE loss
                infoNCE_loss = 0
                for i in range(output.shape[0]):
                    neg_prototypes = [prototype[rel_id] for rel_id in prototype.keys() if rel_id != origin_labels[i].item()]
                    neg_prototypes = torch.stack(neg_prototypes).to(config.device)
                    
                    #--- prepare batch of negative samples 
                    neg_prototypes.requires_grad_ = False
                    neg_prototypes = neg_prototypes.squeeze()
                    f_pos = encoder.infoNCE_f(mask_output[i],outputs[i])
                    f_neg = encoder.infoNCE_f(mask_output[i],neg_prototypes )

                    

                    f_concat = torch.cat([f_pos,f_neg.squeeze()],dim=0)
                    # quick fix for large number
                    f_concat = torch.log(torch.max(f_concat, torch.tensor(1e-9).to(config.device)))

                    infoNCE_loss += -torch.log(softmax(f_concat)[0])
                    #--- prepare batch of negative samples  
                infoNCE_loss /= output.shape[0]           
                # compute MLM loss
                
                mlm_loss = criterion(mask_output.view(-1, mask_output.size()[-1]), mlm_labels.view(-1))
                
                
                loss = loss1 + loss2 + tri_loss + config.infonce_lossfactor*infoNCE_loss + config.mlm_lossfactor*mlm_loss
                # wandb.log({"loss1": loss1, "loss2": loss2, "tri_loss": tri_loss, "infoNCE_loss": infoNCE_loss , "mlm_loss": mlm_loss})
                
                if prev_encoder is not None:
                    prev_reps,_ = prev_encoder(tokens)
                    prev_reps.detach()
                    normalized_prev_reps_emb = F.normalize(prev_reps.view(-1, prev_reps.size()[1]), p=2, dim=1)

                    feature_distill_loss = distill_criterion(normalized_reps_emb, normalized_prev_reps_emb,
                                                            torch.ones(tokens['ids'].size(0)).to(
                                                                config.device))
                    loss += feature_distill_loss

                if prev_dropout_layer is not None and prev_classifier is not None:
                    prediction_distill_loss = None
                    dropout_output_all = []
                    prev_dropout_output_all = []
                    for i in range(config.f_pass):
                        output, _ = dropout_layer(reps)
                        prev_output, _ = prev_dropout_layer(reps)
                        dropout_output_all.append(output)
                        prev_dropout_output_all.append(output)
                        pre_logits = prev_classifier(output).detach()

                        pre_logits = F.softmax(pre_logits.index_select(1, prev_relation_index) / T, dim=1)

                        log_logits = F.log_softmax(logits_all[i].index_select(1, prev_relation_index) / T, dim=1)
                        if i == 0:
                            prediction_distill_loss = -torch.mean(torch.sum(pre_logits * log_logits, dim=1))
                        else:
                            prediction_distill_loss += -torch.mean(torch.sum(pre_logits * log_logits, dim=1))

                    prediction_distill_loss /= config.f_pass
                    loss += prediction_distill_loss
                    dropout_output_all = torch.stack(dropout_output_all)
                    prev_dropout_output_all = torch.stack(prev_dropout_output_all)
                    mean_dropout_output_all = torch.mean(dropout_output_all, dim=0)
                    mean_prev_dropout_output_all = torch.mean(prev_dropout_output_all,dim=0)
                    normalized_output = F.normalize(mean_dropout_output_all.view(-1, mean_dropout_output_all.size()[1]), p=2, dim=1)
                    normalized_prev_output = F.normalize(mean_prev_dropout_output_all.view(-1, mean_prev_dropout_output_all.size()[1]), p=2, dim=1)
                    hidden_distill_loss = distill_criterion(normalized_output, normalized_prev_output,
                                                            torch.ones(tokens['ids'].size(0)).to(
                                                                config.device))
                    loss += hidden_distill_loss
                    print(f"loss1 is {loss1}, loss2 is {loss2}, tri_loss is {tri_loss}, infoNCE_loss is {infoNCE_loss}, mlm_loss is {mlm_loss}")
                    
                    losses.append(loss.item())
                    loss.backward()
                    optimizer.second_step(zero_grad=True)       
            losses.append(loss.item())
        # print(f"loss is {np.array(losses).mean()}")


def train_mem_model_mixup(config, encoder, dropout_layer, classifier, training_data, epochs, map_relid2tempid, new_relation_data,
                prev_encoder, prev_dropout_layer, prev_classifier, prev_relation_index):
    data_loader = get_data_loader(config, training_data, shuffle=True)

    encoder.train()
    dropout_layer.train()
    classifier.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': encoder.parameters(), 'lr': 0.00001},
        {'params': dropout_layer.parameters(), 'lr': 0.00001},
        {'params': classifier.parameters(), 'lr': 0.001}
    ])
    if config.SAM:
        base_optimizer = optim.Adam
        optimizer = SAM(
            params=[{'params': encoder.parameters(), 'lr': 0.00001},
                    {'params': dropout_layer.parameters(), 'lr': 0.00001},
                    {'params': classifier.parameters(), 'lr': 0.001}],
            base_optimizer=base_optimizer,  # Pass the Adam class, not an instance
            rho=config.rho,
            adaptive=True
        )
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
    distill_criterion = nn.CosineEmbeddingLoss()
    neg_cos_sim_loss = NegativeCosSimLoss()
    T = config.kl_temp
    loss_retrieval = MultipleNegativesRankingLoss()
    for epoch_i in range(epochs):
        losses = []
        for step, (labels, _, tokens) in enumerate(data_loader):
            for k in tokens.keys():
                tokens[k] = tokens[k].to(config.device) # B, {''ids', 'mask'}
            label_first = [temp[0] for temp in labels]
            label_second = [temp[1] for temp in labels]
            merged_labels = label_first + label_second
            merged_labels = torch.tensor(merged_labels)
            
            optimizer.zero_grad()

            logits_all = []
            # tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
            merged_labels = merged_labels.to(config.device)
            origin_labels = merged_labels[:]
            merged_labels = [map_relid2tempid[x.item()] for x in merged_labels]
            merged_labels = torch.tensor(merged_labels).to(config.device)
            reps = encoder.forward_mixup(tokens) # B x 2 x 2H
            # print(f"Rep: {reps.shape}")
            reps_first = reps[:,0,:] # B x 2H
            reps_second = reps[:,1,:] # B x 2H
            merged_reps = torch.cat([reps_first, reps_second], dim=0)
            
            #-----------------loss add 1-----------------
            loss_add1 = neg_cos_sim_loss(reps_first, reps_second)
        
            #-----------------loss add 1-----------------

            #-----------------loss add 2-----------------
            reps_hidden_mean_12 = (reps_first + reps_second) / 2    
            matrix_labels_tensor_mean_12 = np.zeros((reps_hidden_mean_12.shape[0], reps_hidden_mean_12.shape[0]), dtype=float)
            for i1 in range(reps_hidden_mean_12.shape[0]):
                    for j1 in range(reps_hidden_mean_12.shape[0]):
                        if i1 != j1:
                            if label_first[i1] in [label_first[j1], label_second[j1]] and label_second[i1] in [label_first[j1], label_second[j1]]:
                                matrix_labels_tensor_mean_12[i1][j1] = 1.0
            matrix_labels_tensor_mean_12 = torch.tensor(matrix_labels_tensor_mean_12).to(config.device)
            
            loss_add2 = loss_retrieval(reps_hidden_mean_12, reps_hidden_mean_12, matrix_labels_tensor_mean_12)
            #-----------------loss add 2-----------------
            
            
            normalized_reps_emb = F.normalize(merged_reps.view(-1, merged_reps.size()[1]), p=2, dim=1)
            outputs,_ = dropout_layer(merged_reps)
            if prev_dropout_layer is not None:
                prev_outputs, _ = prev_dropout_layer(merged_reps)
                positives,negatives = construct_hard_triplets(prev_outputs, origin_labels, new_relation_data)
            else:
                positives, negatives = construct_hard_triplets(outputs, origin_labels, new_relation_data)

            for _ in range(config.f_pass):
                output, output_embedding = dropout_layer(merged_reps)
                logits = classifier(output)
                logits_all.append(logits)

            positives = torch.cat(positives, 0).to(config.device)
            negatives = torch.cat(negatives, 0).to(config.device)
            anchors = outputs
            logits_all = torch.stack(logits_all)
            m_labels = merged_labels.expand((config.f_pass, merged_labels.shape[0]))  # m,B
            loss1 = criterion(logits_all.reshape(-1, logits_all.shape[-1]), m_labels.reshape(-1))
            loss2 = compute_jsd_loss(logits_all)
            tri_loss = triplet_loss(anchors, positives, negatives)
            loss = loss1 + loss2 + tri_loss

            if prev_encoder is not None:
                prev_reps = prev_encoder.forward_mixup(tokens).detach() # B x 2 x 2H
                prev_reps_first = prev_reps[:,0,:]
                prev_reps_second = prev_reps[:,1,:]
                merged_prev_reps = torch.cat([prev_reps_first, prev_reps_second], dim=0)
                normalized_prev_reps_emb = F.normalize(merged_prev_reps.view(-1, merged_prev_reps.size()[1]), p=2, dim=1)

                feature_distill_loss = distill_criterion(normalized_reps_emb, normalized_prev_reps_emb,
                                                         torch.ones(tokens['ids'].size(0) * 2).to(
                                                             config.device))
                loss += feature_distill_loss

            if prev_dropout_layer is not None and prev_classifier is not None:
                prediction_distill_loss = None
                dropout_output_all = []
                prev_dropout_output_all = []
                for i in range(config.f_pass):
                    output, _ = dropout_layer(merged_reps)
                    prev_output, _ = prev_dropout_layer(merged_reps)
                    dropout_output_all.append(output)
                    prev_dropout_output_all.append(output)
                    pre_logits = prev_classifier(output).detach()

                    pre_logits = F.softmax(pre_logits.index_select(1, prev_relation_index) / T, dim=1)

                    log_logits = F.log_softmax(logits_all[i].index_select(1, prev_relation_index) / T, dim=1)
                    if i == 0:
                        prediction_distill_loss = -torch.mean(torch.sum(pre_logits * log_logits, dim=1))
                    else:
                        prediction_distill_loss += -torch.mean(torch.sum(pre_logits * log_logits, dim=1))

                prediction_distill_loss /= config.f_pass
                loss += prediction_distill_loss
                dropout_output_all = torch.stack(dropout_output_all)
                prev_dropout_output_all = torch.stack(prev_dropout_output_all)
                mean_dropout_output_all = torch.mean(dropout_output_all, dim=0)
                mean_prev_dropout_output_all = torch.mean(prev_dropout_output_all,dim=0)
                normalized_output = F.normalize(mean_dropout_output_all.view(-1, mean_dropout_output_all.size()[1]), p=2, dim=1)
                normalized_prev_output = F.normalize(mean_prev_dropout_output_all.view(-1, mean_prev_dropout_output_all.size()[1]), p=2, dim=1)
                hidden_distill_loss = distill_criterion(normalized_output, normalized_prev_output,
                                                         torch.ones(tokens['ids'].size(0) * 2).to(
                                                             config.device))
                loss += hidden_distill_loss
            if not torch.isnan(loss_add1).any():
                loss += config.loss1_factor*loss_add1
            if not torch.isnan(loss_add2).any():
                loss += config.loss2_factor*loss_add2
            print(f"loss1 is {loss1.item()}, loss2 is {loss2.item()}, tri_loss is {tri_loss.item()}, loss_add1 is {loss_add1.item()}, loss_add2 is {loss_add2.item()}")
            if not config.SAM:
                loss.backward()
                losses.append(loss.item())
                optimizer.step()
            else:
                loss.backward()
                optimizer.first_step(zero_grad=True)
                logits_all = []
                reps = encoder.forward_mixup(tokens) # B x 2 x 2H
                # print(f"Rep: {reps.shape}")
                reps_first = reps[:,0,:] # B x 2H
                reps_second = reps[:,1,:] # B x 2H
                merged_reps = torch.cat([reps_first, reps_second], dim=0)
                
                #-----------------loss add 1-----------------
            
                loss_add1 = neg_cos_sim_loss(reps_first, reps_second)

            
            
                #-----------------loss add 1-----------------

                #-----------------loss add 2-----------------
                reps_hidden_mean_12 = (reps_first + reps_second) / 2    
                
                loss_add2 = loss_retrieval(reps_hidden_mean_12, reps_hidden_mean_12, matrix_labels_tensor_mean_12)
                #-----------------loss add 2-----------------
                
                
                normalized_reps_emb = F.normalize(merged_reps.view(-1, merged_reps.size()[1]), p=2, dim=1)
                outputs,_ = dropout_layer(merged_reps)
                if prev_dropout_layer is not None:
                    prev_outputs, _ = prev_dropout_layer(merged_reps)
                    positives,negatives = construct_hard_triplets(prev_outputs, origin_labels, new_relation_data)
                else:
                    positives, negatives = construct_hard_triplets(outputs, origin_labels, new_relation_data)

                for _ in range(config.f_pass):
                    output, output_embedding = dropout_layer(merged_reps)
                    logits = classifier(output)
                    logits_all.append(logits)

                positives = torch.cat(positives, 0).to(config.device)
                negatives = torch.cat(negatives, 0).to(config.device)
                anchors = outputs
                logits_all = torch.stack(logits_all)
                m_labels = merged_labels.expand((config.f_pass, merged_labels.shape[0]))  # m,B
                loss1 = criterion(logits_all.reshape(-1, logits_all.shape[-1]), m_labels.reshape(-1))
                loss2 = compute_jsd_loss(logits_all)
                tri_loss = triplet_loss(anchors, positives, negatives)
                loss = loss1 + loss2 + tri_loss

                if prev_encoder is not None:
                    prev_reps = prev_encoder.forward_mixup(tokens).detach() # B x 2 x 2H
                    prev_reps_first = prev_reps[:,0,:]
                    prev_reps_second = prev_reps[:,1,:]
                    merged_prev_reps = torch.cat([prev_reps_first, prev_reps_second], dim=0)
                    normalized_prev_reps_emb = F.normalize(merged_prev_reps.view(-1, merged_prev_reps.size()[1]), p=2, dim=1)

                    feature_distill_loss = distill_criterion(normalized_reps_emb, normalized_prev_reps_emb,
                                                            torch.ones(tokens['ids'].size(0) * 2).to(
                                                                config.device))
                    loss += feature_distill_loss

                if prev_dropout_layer is not None and prev_classifier is not None:
                    prediction_distill_loss = None
                    dropout_output_all = []
                    prev_dropout_output_all = []
                    for i in range(config.f_pass):
                        output, _ = dropout_layer(merged_reps)
                        prev_output, _ = prev_dropout_layer(merged_reps)
                        dropout_output_all.append(output)
                        prev_dropout_output_all.append(output)
                        pre_logits = prev_classifier(output).detach()

                        pre_logits = F.softmax(pre_logits.index_select(1, prev_relation_index) / T, dim=1)

                        log_logits = F.log_softmax(logits_all[i].index_select(1, prev_relation_index) / T, dim=1)
                        if i == 0:
                            prediction_distill_loss = -torch.mean(torch.sum(pre_logits * log_logits, dim=1))
                        else:
                            prediction_distill_loss += -torch.mean(torch.sum(pre_logits * log_logits, dim=1))

                    prediction_distill_loss /= config.f_pass
                    loss += prediction_distill_loss
                    dropout_output_all = torch.stack(dropout_output_all)
                    prev_dropout_output_all = torch.stack(prev_dropout_output_all)
                    mean_dropout_output_all = torch.mean(dropout_output_all, dim=0)
                    mean_prev_dropout_output_all = torch.mean(prev_dropout_output_all,dim=0)
                    normalized_output = F.normalize(mean_dropout_output_all.view(-1, mean_dropout_output_all.size()[1]), p=2, dim=1)
                    normalized_prev_output = F.normalize(mean_prev_dropout_output_all.view(-1, mean_prev_dropout_output_all.size()[1]), p=2, dim=1)
                    hidden_distill_loss = distill_criterion(normalized_output, normalized_prev_output,
                                                            torch.ones(tokens['ids'].size(0) * 2).to(
                                                                config.device))
                    loss += hidden_distill_loss
                # loss += config.loss1_factor*loss_add1 + config.loss2_factor*loss_add2
                if not torch.isnan(loss_add1).any():
                    loss += config.loss1_factor*loss_add1
                if not torch.isnan(loss_add2).any():
                    loss += config.loss2_factor*loss_add2
                print(f"loss1 is {loss1.item()}, loss2 is {loss2.item()}, tri_loss is {tri_loss.item()}, loss_add1 is {loss_add1.item()}, loss_add2 is {loss_add2.item()}")
                    
                loss.backward()
                optimizer.second_step(zero_grad=True)

            # print("Loss: ", loss.item())
            losses.append(loss.item())
        # print(f"loss is {np.array(losses).mean()}")




def batch2device(batch_tuple, device):
    ans = []
    for var in batch_tuple:
        if isinstance(var, torch.Tensor):
            ans.append(var.to(device))
        elif isinstance(var, list):
            ans.append(batch2device(var))
        elif isinstance(var, tuple):
            ans.append(tuple(batch2device(var)))
        else:
            ans.append(var)
    return ans


def evaluate_strict_model(config, encoder, dropout_layer, classifier, test_data, seen_relations, map_relid2tempid):
    data_loader = get_data_loader(config, test_data, batch_size=1)
    encoder.eval()
    dropout_layer.eval()
    classifier.eval()
    n = len(test_data)

    correct = 0
    for step, batch_data in enumerate(data_loader):
        labels, _, tokens = batch_data
        for k in tokens.keys():
                tokens[k] = tokens[k].to(config.device) # B, {''ids', 'mask'}
        labels = labels.to(config.device)
        labels = [map_relid2tempid[x.item()] for x in labels]
        labels = torch.tensor(labels).to(config.device)

        
        # tokens = torch.stack([x.to(config.device) for x in tokens],dim=0)
        reps, _  = encoder(tokens)
        reps, _ = dropout_layer(reps)
        logits = classifier(reps)

        seen_relation_ids = [rel2id[relation] for relation in seen_relations]
        seen_relation_ids = [map_relid2tempid[relation] for relation in seen_relation_ids]
        seen_sim = logits[:,seen_relation_ids].cpu().data.numpy()
        max_smi = np.max(seen_sim,axis=1)

        label_smi = logits[:,labels].cpu().data.numpy()

        if label_smi >= max_smi:
            correct += 1

    return correct/n


def select_data(config, encoder, dropout_layer, relation_dataset):
    data_loader = get_data_loader(config, relation_dataset, shuffle=False, drop_last=False, batch_size=1)
    features = []
    encoder.eval()
    dropout_layer.eval()
    for step, batch_data in enumerate(data_loader):
        labels, _, tokens = batch_data
        # tokens = torch.stack([x.to(config.device) for x in tokens],dim=0)
        for k in tokens.keys():
                tokens[k] = tokens[k].to(config.device) # B, {''ids', 'mask'}
        with torch.no_grad():
            output , mask_output = encoder(tokens)
            feature = dropout_layer(output)[1].cpu()
        features.append(feature)

    features = np.concatenate(features)
    num_clusters = min(config.num_protos, len(relation_dataset))
    distances = KMeans(n_clusters=num_clusters, random_state=0).fit_transform(features)

    memory = []
    for k in range(num_clusters):
        sel_index = np.argmin(distances[:, k])
        instance = relation_dataset[sel_index]
        memory.append(instance)
    return memory


def get_proto(config, encoder, dropout_layer, relation_dataset):
    """
    Get the prototype of the relation dataset
    """
    data_loader = get_data_loader(config, relation_dataset, shuffle=False, drop_last=False, batch_size=1)
    features = []
    encoder.eval()
    dropout_layer.eval()
    for step, batch_data in enumerate(data_loader):
        labels, _, tokens = batch_data
        for k in tokens.keys():
            tokens[k] = tokens[k].to(config.device) # B, {''ids', 'mask'}   
        # tokens = torch.stack([x.to(config.device) for x in tokens],dim=0)
        with torch.no_grad():
            output , mask_output = encoder(tokens)
            feature = dropout_layer(output)[1]
        features.append(feature)
    features = torch.cat(features, dim=0)
    # t_distribution = utils.get_T_distribution(features) # get the t-distribution of the features for generate data

    proto = torch.mean(features, dim=0, keepdim=True).cpu()
    standard = torch.sqrt(torch.var(features, dim=0)).cpu()
    
    return proto, standard 

 

def generate_relation_data(protos, relation_standard):
    relation_data = {}
    relation_sample_nums = 10
    for id in protos.keys():
        relation_data[id] = []
        difference = np.random.normal(loc=0, scale=1, size=relation_sample_nums)
        for diff in difference:
            relation_data[id].append(protos[id] + diff * relation_standard[id])
    return relation_data


def generate_current_relation_data(config, encoder, dropout_layer, relation_dataset):
    data_loader = get_data_loader(config, relation_dataset, shuffle=False, drop_last=False, batch_size=1)
    relation_data = []
    encoder.eval()
    dropout_layer.eval()
    for step, batch_data in enumerate(data_loader):
        labels, _, tokens = batch_data
        for k in tokens.keys():
                tokens[k] = tokens[k].to(config.device) # B, {''ids', 'mask'}
        # tokens = torch.stack([x.to(config.device) for x in tokens],dim=0)
        with torch.no_grad():
            output , mask_output = encoder(tokens)
            feature = dropout_layer(output)[1].cpu()
        relation_data.append(feature)
    return relation_data

from transformers import  BertTokenizer
def data_augmentation(config, encoder, train_data, prev_train_data):
    expanded_train_data = train_data[:]
    expanded_prev_train_data = prev_train_data[:]
    encoder.eval()
    all_data = train_data + prev_train_data
    tokenizer = BertTokenizer.from_pretrained(config.bert_path, additional_special_tokens=["[E11]", "[E12]", "[E21]", "[E22]"])
    entity_index = []
    entity_mention = []
    for sample in all_data:
        if 30522 not in sample['tokens'] or 30523 not in sample['tokens'] or 30524 not in sample['tokens'] or 30525 not in sample['tokens']: # hot fix
            continue
        e11 = sample['tokens'].index(30522)
        e12 = sample['tokens'].index(30523)
        e21 = sample['tokens'].index(30524)
        e22 = sample['tokens'].index(30525)
        entity_index.append([e11,e12])
        entity_mention.append(sample['tokens'][e11+1:e12])
        entity_index.append([e21,e22])
        entity_mention.append(sample['tokens'][e21+1:e22])

    data_loader = get_data_loader(config, all_data, shuffle=False, drop_last=False, batch_size=1)
    features = []
    encoder.eval()
    for step, batch_data in enumerate(data_loader):
        labels, _, tokens = batch_data
        for k in tokens.keys():
                tokens[k] = tokens[k].to(config.device) # B, {''ids', 'mask'}
        # tokens = torch.stack([x.to(config.device) for x in tokens],dim=0)
        with torch.no_grad():
            feature,_ = encoder(tokens)
        feature1, feature2 = torch.split(feature, [config.encoder_output_size,config.encoder_output_size], dim=1)
        features.append(feature1)
        features.append(feature2)
    features = torch.cat(features, dim=0)
    # similarity_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=-1)
    similarity_matrix = []
    for i in range(len(features)):
        similarity_matrix.append([0]*len(features))

    for i in range(len(features)):
        for j in range(i,len(features)):
            similarity = F.cosine_similarity(features[i],features[j],dim=0)
            similarity_matrix[i][j] = similarity
            similarity_matrix[j][i] = similarity

    similarity_matrix = torch.tensor(similarity_matrix).to(config.device)
    zero = torch.zeros_like(similarity_matrix).to(config.device)
    diag = torch.diag_embed(torch.diag(similarity_matrix))
    similarity_matrix -= diag
    similarity_matrix = torch.where(similarity_matrix<0.95, zero, similarity_matrix)
    nonzero_index = torch.nonzero(similarity_matrix)
    expanded_train_count = 0

    for origin, replace in nonzero_index:
        sample_index = int(origin/2)
        sample = all_data[sample_index]
        if entity_mention[origin] == entity_mention[replace]:
            continue
        new_tokens = sample['tokens'][:entity_index[origin][0]+1] + entity_mention[replace] + sample['tokens'][entity_index[origin][1]:]
        if len(new_tokens) < config.max_length:
            new_tokens = new_tokens + [0]*(config.max_length-len(new_tokens))
        else:
            new_tokens = new_tokens[:config.max_length]

        new_sample = {
            'relation': sample['relation'],
            'neg_labels': sample['neg_labels'],
            'tokens': new_tokens
        }
        if sample_index < len(train_data) and expanded_train_count < 5 * len(train_data):
            expanded_train_data.append(new_sample)
            expanded_train_count += 1
        else:
            expanded_prev_train_data.append(new_sample)
    return expanded_train_data, expanded_prev_train_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="tacred", type=str)
    parser.add_argument("--shot", default=5, type=int)
    parser.add_argument('--config', default='config.ini')
    parser.add_argument("--step1_epochs", default=5, type=int)
    parser.add_argument("--step2_epochs", default=10, type=int)
    parser.add_argument("--step3_epochs", default=10, type=int)
    parser.add_argument("--loss1_factor", default=0.5, type=float)
    parser.add_argument("--loss2_factor", default=0.5, type=float)
    parser.add_argument("--mixup", action = "store_true")
    parser.add_argument("--SAM", action = "store_true")
    parser.add_argument("--rho", default=0.05, type=float)
    parser.add_argument("--SAM_type", default="", type=str, help = "full or current")
    args = parser.parse_args()
    config = Config(args.config)
    config.step1_epochs = args.step1_epochs
    config.step2_epochs = args.step2_epochs
    config.step3_epochs = args.step3_epochs
    config.loss1_factor = args.loss1_factor
    config.loss2_factor = args.loss2_factor
    config.SAM = args.SAM
    config.rho = args.rho
    config.SAM_type = args.SAM_type
    config.mixup = args.mixup
    config.device = torch.device(config.device)
    config.n_gpu = torch.cuda.device_count()
    config.batch_size_per_step = int(config.batch_size / config.gradient_accumulation_steps)
    
    #sckd
    config.pattern = 'entity_marker_mask'
    

    #--- fix lossfactor for reproduce results
    if args.task == "FewRel":
        config.infonce_lossfactor = 0.5
        config.mlm_lossfactor = 0.0
    elif args.task == "tacred":
        config.infonce_lossfactor = 0.4
        config.mlm_lossfactor = 0.0
    else:
        raise ValueError("Invalid task")
    #--- fix lossfactor
    config.task = args.task
    config.shot = args.shot
    
    config.step1_epochs = args.step1_epochs
    config.step2_epochs = args.step2_epochs
    config.step3_epochs = args.step3_epochs
    config.temperature = 0.08

    config.loss1_factor = args.loss1_factor
    config.loss2_factor = args.loss2_factor

    if config.task == "FewRel":
        config.relation_file = "data/fewrel/relation_name.txt"
        config.rel_index = "data/fewrel/rel_index.npy"
        config.rel_feature = "data/fewrel/rel_feature.npy"
        config.rel_des_file = "data/fewrel/relation_description.txt"
        config.num_of_relation = 80
        if config.shot == 5:
            config.rel_cluster_label = "data/fewrel/CFRLdata_10_100_10_5/rel_cluster_label_0.npy"
            config.training_file = "data/fewrel/CFRLdata_10_100_10_5/train_0.txt"
            config.valid_file = "data/fewrel/CFRLdata_10_100_10_5/valid_0.txt"
            config.test_file = "data/fewrel/CFRLdata_10_100_10_5/test_0.txt"
        elif config.shot == 10:
            config.rel_cluster_label = "data/fewrel/CFRLdata_10_100_10_10/rel_cluster_label_0.npy"
            config.training_file = "data/fewrel/CFRLdata_10_100_10_10/train_0.txt"
            config.valid_file = "data/fewrel/CFRLdata_10_100_10_10/valid_0.txt"
            config.test_file = "data/fewrel/CFRLdata_10_100_10_10/test_0.txt"
        else:
            config.rel_cluster_label = "data/fewrel/CFRLdata_10_100_10_2/rel_cluster_label_0.npy"
            config.training_file = "data/fewrel/CFRLdata_10_100_10_2/train_0.txt"
            config.valid_file = "data/fewrel/CFRLdata_10_100_10_2/valid_0.txt"
            config.test_file = "data/fewrel/CFRLdata_10_100_10_2/test_0.txt"
    else:
        config.relation_file = "data/tacred/relation_name.txt"
        config.rel_index = "data/tacred/rel_index.npy"
        config.rel_feature = "data/tacred/rel_feature.npy"
        config.num_of_relation = 41
        if config.shot == 5:
            config.rel_cluster_label = "data/tacred/CFRLdata_10_100_10_5/rel_cluster_label_0.npy"
            config.training_file = "data/tacred/CFRLdata_10_100_10_5/train_0.txt"
            config.valid_file = "data/tacred/CFRLdata_10_100_10_5/valid_0.txt"
            config.test_file = "data/tacred/CFRLdata_10_100_10_5/test_0.txt"
        else:
            config.rel_cluster_label = "data/tacred/CFRLdata_10_100_10_10/rel_cluster_label_0.npy"
            config.training_file = "data/tacred/CFRLdata_10_100_10_10/train_0.txt"
            config.valid_file = "data/tacred/CFRLdata_10_100_10_10/valid_0.txt"
            config.test_file = "data/tacred/CFRLdata_10_100_10_10/test_0.txt"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)
    
    file_handler = logging.FileHandler(f'SCKD-mmi-mixup-logs-task_{config.task}-shot_{config.shot}-epoch_{config.step1_epochs}_{config.step2_epochs}_{config.step3_epochs}-lossfactor_{config.loss1_factor}_{config.loss2_factor}.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    result_cur_test = []
    result_whole_test = []
    bwt_whole = []
    fwt_whole = []
    X = []
    Y = []
    relation_divides = []
    for i in range(10):
        relation_divides.append([])
    for rou in range(config.total_round):
        test_cur = []
        test_total = []
        random.seed(config.seed+rou*100)
        sampler = data_sampler(config=config, seed=config.seed+rou*100)

        config.extended_vocab_size = len(sampler.tokenizer)
        
        id2rel = sampler.id2rel
        rel2id = sampler.rel2id
        id2sentence = sampler.get_id2sent()
        encoder = Bert_EncoderMLM(config=config).to(config.device)
        dropout_layer = Dropout_Layer(config=config).to(config.device)
        num_class = len(sampler.id2rel)
        

        memorized_samples = {}
        memory = collections.defaultdict(list)
        history_relations = []
        history_data = []
        prev_relations = []
        classifier = None
        prev_classifier = None
        prev_encoder = None
        prev_dropout_layer = None
        relation_standard = {}
        forward_accs = []
        for steps, (training_data, valid_data, test_data, current_relations, historic_test_data, seen_relations) in enumerate(sampler):
            print(current_relations)

            prev_relations = history_relations[:]
            train_data_for_initial = []
            count = 0
            for relation in current_relations:
                history_relations.append(relation)
                train_data_for_initial += training_data[relation]
                relation_divides[count].append(float(rel2id[relation]))
                count += 1


            temp_rel2id = [rel2id[x] for x in seen_relations]
            map_relid2tempid = {k: v for v, k in enumerate(temp_rel2id)}
            prev_relation_index = []
            prev_samples = []
            for relation in prev_relations:
                prev_relation_index.append(map_relid2tempid[rel2id[relation]])
                prev_samples += memorized_samples[relation]
            prev_relation_index = torch.tensor(prev_relation_index).to(config.device)

            classifier = Softmax_Layer(input_size=encoder.output_size, num_class=len(history_relations)).to(
                config.device)

            temp_protos = {}
            # temp_distributions = {}
            for relation in current_relations:
                proto, _  = get_proto(config, encoder, dropout_layer, training_data[relation])
                temp_protos[rel2id[relation]] = proto
                # temp_distributions[rel2id[relation]] = t_distributon

            for relation in prev_relations:
                proto, _  = get_proto(config, encoder, dropout_layer, memorized_samples[relation])
                temp_protos[rel2id[relation]] = proto
                # temp_distributions[rel2id[relation]] = t_distributon


            test_data_1 = []
            for relation in current_relations:
                test_data_1 += test_data[relation]

            if steps != 0:
                forward_acc = evaluate_strict_model(config, prev_encoder, prev_dropout_layer, classifier, test_data_1, seen_relations, map_relid2tempid)
                forward_accs.append(forward_acc)

            if config.SAM_type == 'current' : 
                config.SAM = True
            train_simple_model(config, encoder, dropout_layer, classifier, train_data_for_initial, config.step1_epochs, map_relid2tempid)
            print(f"simple finished")
            if config.SAM_type == 'current' :
                config.SAM = False


            temp_protos = {} # key : relation id, value : prototype
            # temp_distributions = {} # key : relation id, value : t-distribution
            for relation in current_relations:
                proto, standard  = get_proto(config,encoder,dropout_layer,training_data[relation])
                temp_protos[rel2id[relation]] = proto
                relation_standard[rel2id[relation]] = standard
                # temp_distributions[rel2id[relation]] = t_distributon


            for relation in prev_relations:
                proto, _ = get_proto(config,encoder,dropout_layer,memorized_samples[relation])
                temp_protos[rel2id[relation]] = proto
                # temp_distributions[rel2id[relation]] = t_distributon

            new_relation_data = generate_relation_data(temp_protos, relation_standard)

            for relation in current_relations:
                new_relation_data[rel2id[relation]].extend(generate_current_relation_data(config, encoder,dropout_layer,training_data[relation]))

            # expanded_train_data_for_initial, expanded_prev_samples = data_augmentation(config, encoder,
            #                                                                            train_data_for_initial,
            #                                                                            prev_samples)
            torch.cuda.empty_cache()
            # print(len(train_data_for_initial))
            # print(len(expanded_train_data_for_initial))


            train_mem_model(config, encoder, dropout_layer, classifier, train_data_for_initial, config.step2_epochs, map_relid2tempid, new_relation_data,
                        prev_encoder, prev_dropout_layer, prev_classifier, prev_relation_index , prototype=temp_protos )
            print(f"first finished")

            for relation in current_relations:
                memorized_samples[relation] = select_data(config, encoder, dropout_layer, training_data[relation])
                memory[rel2id[relation]] = select_data(config, encoder, dropout_layer, training_data[relation])

            train_data_for_memory = []
            # train_data_for_memory += expanded_prev_samples
            train_data_for_memory += prev_samples
            for relation in current_relations:
                train_data_for_memory += memorized_samples[relation]
            print(len(seen_relations))
            print(len(train_data_for_memory))

            temp_protos = {}
            # temp_distributions = {}
            for relation in seen_relations:
                proto, _  = get_proto(config, encoder, dropout_layer, memorized_samples[relation])
                temp_protos[rel2id[relation]] = proto
            data_for_augment = train_data_for_initial + train_data_for_memory
            if steps > 0:
                if config.mixup:
                    mixup_samples = mixup_data_augmentation(data_for_augment)
                    print("Num mixup samples", len(mixup_samples))
                    train_mem_model_mixup(config, encoder, dropout_layer, classifier, mixup_samples, 2, map_relid2tempid, new_relation_data,
                                prev_encoder, prev_dropout_layer, prev_classifier, prev_relation_index)
            train_mem_model(config, encoder, dropout_layer, classifier, train_data_for_memory, config.step3_epochs, map_relid2tempid, new_relation_data,
                        prev_encoder, prev_dropout_layer, prev_classifier, prev_relation_index , prototype=temp_protos )
            print(f"memory finished")
            test_data_1 = []
            for relation in current_relations:
                test_data_1 += test_data[relation]

            test_data_2 = []
            for relation in seen_relations:
                test_data_2 += historic_test_data[relation]
            history_data.append(test_data_1)


            print(len(test_data_1))
            print(len(test_data_2))
            # cur_acc = evaluate_strict_model(config, encoder, classifier, test_data_1, seen_relations, map_relid2tempid)
            # total_acc = evaluate_strict_model(config, encoder, classifier, test_data_2, seen_relations, map_relid2tempid)

            cur_acc = evaluate_strict_model(config, encoder,dropout_layer,classifier, test_data_1, seen_relations, map_relid2tempid)
            total_acc = evaluate_strict_model(config, encoder, dropout_layer, classifier, test_data_2, seen_relations, map_relid2tempid)

            print(f'Restart Num {rou + 1}')
            print(f'task--{steps + 1}:')
            print(f'current test acc:{cur_acc}')
            print(f'history test acc:{total_acc}')
            # wandb.log({f"Round {rou} history test acc": total_acc})
            test_cur.append(cur_acc)
            test_total.append(total_acc)
            print(test_cur)
            print(test_total)

            logger.info('#############-----############')
            logger.info(f'Restart Num {rou + 1}')
            logger.info(f'task--{steps + 1}:')
            logger.info(f'current test acc:{cur_acc}')
            logger.info(f'history test acc:{total_acc}')
            logger.info(f'test current: {test_cur}')
            logger.info(f'test total: {test_total}')

            accuracy = []
            temp_rel2id = [rel2id[x] for x in history_relations]
            map_relid2tempid = {k: v for v, k in enumerate(temp_rel2id)}
            for data in history_data:
                # accuracy.append(
                #     evaluate_strict_model(config, encoder, classifier, data, history_relations, map_relid2tempid))
                accuracy.append(evaluate_strict_model(config, encoder, dropout_layer, classifier, data, seen_relations, map_relid2tempid))
            print(accuracy)
            logger.info(f'accuracy: {accuracy}')

            prev_encoder = deepcopy(encoder)
            prev_dropout_layer = deepcopy(dropout_layer)
            prev_classifier = deepcopy(classifier)
            torch.cuda.empty_cache()
        result_cur_test.append(np.array(test_cur))
        result_whole_test.append(np.array(test_total)*100)
        print("result_whole_test")
        print(result_whole_test)
        logger.info(f'result_whole_test: {result_whole_test}')
        avg_result_cur_test = np.average(result_cur_test, 0)
        avg_result_all_test = np.average(result_whole_test, 0)
        print("avg_result_cur_test")
        print(avg_result_cur_test)
        logger.info(f'avg_result_cur_test: {avg_result_cur_test}')
        print("avg_result_all_test")
        print(avg_result_all_test)
        logger.info(f'avg_result_all_test: {avg_result_all_test}')
        # wandb.log({"avg_result_all_test": avg_result_all_test})
        std_result_all_test = np.std(result_whole_test, 0)
        print("std_result_all_test")
        print(std_result_all_test)
        logger.info(f'std_result_all_test: {std_result_all_test}')
        # wandb.log({"std_result_all_test": std_result_all_test})
        accuracy = []
        temp_rel2id = [rel2id[x] for x in history_relations]
        map_relid2tempid = {k: v for v, k in enumerate(temp_rel2id)}
        for data in history_data:
            accuracy.append(evaluate_strict_model(config, encoder, dropout_layer, classifier, data, history_relations, map_relid2tempid))
        print(accuracy)
        logger.info(f'accuracy: {accuracy}')
        bwt = 0.0
        for k in range(len(accuracy)-1):
            bwt += accuracy[k]-test_cur[k]
        bwt /= len(accuracy)-1
        bwt_whole.append(bwt)
        fwt_whole.append(np.average(np.array(forward_accs)))
        print("bwt_whole")
        print(bwt_whole)
        logger.info(f'bwt_whole: {bwt_whole}')
        print("fwt_whole")
        print(fwt_whole)
        logger.info(f'fwt_whole: {fwt_whole}')
        avg_bwt = np.average(np.array(bwt_whole))
        print("avg_bwt_whole")
        print(avg_bwt)
        logger.info(f'avg_bwt_whole: {avg_bwt}')

        avg_fwt = np.average(np.array(fwt_whole))
        print("avg_fwt_whole")
        print(avg_fwt)
        logger.info(f'avg_fwt_whole: {avg_fwt}')




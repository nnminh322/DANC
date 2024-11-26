import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.functional import normalize
from torch.optim import AdamW
from utils import *
from configs import parse_arguments
from model import BertED
from tqdm import tqdm
from exemplars import Exemplars
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter   
import os, time
import logging
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from sam import SAM



# PERM_5 = [[0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [0, 3, 1, 4, 2], [1, 2, 0, 3, 4], [3, 4, 0, 1, 2]]

# PERM_10 = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]



def train(local_rank, args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
            "[%(asctime)s]-[%(filename)s line:%(lineno)d]:%(message)s "
        )
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    cur_time = time.strftime('%Y-%m-%d-%H-%M-%S')
    if args.log:
        if not os.path.exists(os.path.join(args.tb_dir, args.dataset, args.joint_da_loss, str(args.class_num) + "class", str(args.shot_num) + "shot", args.cl_aug, args.log_name, "perm" + str(args.perm_id))):
            os.makedirs(os.path.join(args.tb_dir, args.dataset, args.joint_da_loss, str(args.class_num) + "class", str(args.shot_num) + "shot", args.cl_aug, args.log_name, "perm" + str(args.perm_id)))
        if not os.path.exists(os.path.join(args.log_dir, args.dataset, args.joint_da_loss, str(args.class_num) + "class", str(args.shot_num) + "shot", args.cl_aug, args.log_name, "perm" + str(args.perm_id))):
            os.makedirs(os.path.join(args.log_dir, args.dataset, args.joint_da_loss, str(args.class_num) + "class", str(args.shot_num) + "shot", args.cl_aug, args.log_name, "perm" + str(args.perm_id)))
        writer = SummaryWriter(os.path.join(args.tb_dir, args.dataset, args.joint_da_loss, str(args.class_num) + "class", str(args.shot_num) + "shot", args.cl_aug, args.log_name, "perm" + str(args.perm_id), cur_time))
        fh = logging.FileHandler(os.path.join(args.log_dir, args.dataset, args.joint_da_loss, str(args.class_num) + "class", str(args.shot_num) + "shot", args.cl_aug, args.log_name, "perm" + str(args.perm_id), cur_time + '.log'), mode='a')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    for arg in vars(args):
        logger.info('{}={}'.format(arg.upper(), getattr(args, arg)))
    logger.info('')
    # set device, whether to use cuda or cpu
    device = torch.device(args.device if torch.cuda.is_available() and args.device != 'cpu' else "cpu")  # type: ignore
    # get streams from json file and permute them in pre-defined order
    # PERM = PERM_5 if args.task_num == 5 else PERM_10
    streams = collect_from_json(args.dataset, args.stream_root, 'stream')
    # streams = [streams[l] for l in PERM[int(args.perm_id)]] # permute the stream
    label2idx = {0:0}
    for st in streams:
        for lb in st:
            if lb not in label2idx:
                label2idx[lb] = len(label2idx)
    streams_indexed = [[label2idx[l] for l in st] for st in streams]
    model = BertED(args.class_num+1, args.input_map) # define model
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.decay, eps=args.adamw_eps, betas=(0.9, 0.999)) #TODO: Hyper parameters
    if args.sam:
            base_optimizer = AdamW
            optimizer = SAM(params=model.parameters(), base_optimizer=base_optimizer, rho=args.rho, adaptive=True, lr=args.lr, weight_decay=args.decay, eps=args.adamw_eps, betas=(0.9, 0.999))
    # if args.amp:
        # model, optimizer = amp.initialize(model, optimizer, opt_level="O1") 
    if args.parallel == 'DDP':
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", rank=local_rank, world_size=args.world_size)
        # device_id = [0, 1, 2, 3, 4, 5, 6, 7]
        model = DDP(model, device_ids= [local_rank], find_unused_parameters=True)
    elif args.parallel == 'DP':
        os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3, 4, 5, 6, 7' 
        model = nn.DataParallel(model, device_ids=[int(it) for it in args.device_ids.split(" ")])


    # optimizer = SGD(model.parameters(), lr=args.lr) # TODO: Use AdamW, GPU out of memory

    criterion_ce = nn.CrossEntropyLoss()
    criterion_fd = nn.CosineEmbeddingLoss()
    all_labels = []
    all_labels = list(set([t for stream in streams_indexed for t in stream if t not in all_labels]))
    task_idx = [i for i in range(len(streams_indexed))]
    labels = all_labels.copy()

    # training process
    learned_types = [0]
    prev_learned_types = [0]
    dev_scores_ls = []
    exemplars = Exemplars() # TODO: 
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        state_dict = torch.load(args.resume)
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        task_idx = task_idx[state_dict['stage']:]
        # TODO: test use
        labels = state_dict['labels']
        learned_types = state_dict['learned_types']
        prev_learned_types = state_dict['prev_learned_types']
    if args.early_stop:
        e_pth = "./checkpoints/" + args.log_name + ".pth"
    for stage in task_idx:
        # if stage > 0:
        #     break
        logger.info(f"Stage {stage}")
        logger.info(f'Loading train instances for stage {stage}')
        # stage = 1 # TODO: test use
        # exemplars = Exemplars() # TODO: test use
        if args.single_label:
            stream_dataset = collect_sldataset(args.dataset, args.data_root, 'train', label2idx, stage, streams[stage])
        else:
            stream_dataset = collect_dataset(args.dataset, args.data_root, 'train', label2idx, stage, [i for item in streams[stage:] for i in item])
        if args.parallel == 'DDP':
            stream_sampler = DistributedSampler(stream_dataset, shuffle=True)
            org_loader = DataLoader(
                dataset=stream_dataset,
                sampler=stream_sampler,
                batch_size=args.batch_size,
                # batch_size=args.shot_num + int(args.class_num / args.shot_num),
                collate_fn= lambda x:x)
        else:
            org_loader = DataLoader(
                dataset=stream_dataset,
                shuffle=True,
                batch_size=args.batch_size,
                # batch_size=args.shot_num + int(args.class_num / args.shot_num),
                collate_fn= lambda x:x)
        stage_loader = org_loader

        learning_nums = (args.class_num // args.task_num)
        event_type_ids, event_type_mask = exemplars.get_event_type_ids((stage+1)*learning_nums, device, label2idx)
        with torch.no_grad():
            event_type_reps = model.forward_backbone(event_type_ids, event_type_mask)[:, 0, :] # way x H

        if stage > 0:
            # if args.early_stop and no_better == args.patience:
            #     logger.info("Early stopping finished, loading stage: " + str(stage))
            #     model.load_state_dict(torch.load(e_pth))
            prev_model = deepcopy(model) # TODO:test use
            for item in streams_indexed[stage - 1]:
                if not item in prev_learned_types:
                    prev_learned_types.append(item)
            # TODO: test use
            # prev_model = deepcopy(model) # TODO: How does optimizer distinguish deep copy parameters
            # exclude_none_labels = [t for t in streams_indexed[stage - 1] if t != 0]
            logger.info(f'Loading train instances without negative instances for stage {stage}')
            exemplar_dataset = collect_exemplar_dataset(args.dataset, args.data_root, 'train', label2idx, stage-1, streams[stage-1])
            exemplar_loader = DataLoader(
                dataset=exemplar_dataset,
                batch_size=64,
                shuffle=True,
                collate_fn=lambda x:x)
            # exclude_none_loader = train_ecn_loaders[stage - 1]
            # TODO: test use
            # exemplars.set_exemplars(prev_model.to('cpu'), exclude_none_loader, len(learned_types), device)
            exemplars.set_exemplars(prev_model, exemplar_loader, len(learned_types), device, label2idx)
            # if not args.replay:
            if not args.no_replay:
                stage_loader = exemplars.build_stage_loader(stream_dataset)
            # else:
            #     e_loader = list(exemplars.build_stage_loader(MAVEN_Dataset([], [], [], [])))
            if args.rep_aug != "none":

                e_loader = exemplars.build_stage_loader(MAVEN_Dataset([], [], [], []))
            # prev_model.to(args.device)   # TODO: test use

        for item in streams_indexed[stage]:
            if not item in learned_types:
                learned_types.append(item)
        logger.info(f'Learned types: {learned_types}')
        logger.info(f'Previous learned types: {prev_learned_types}')
        dev_score = None
        no_better = 0
        for ep in range(args.epochs):
            if stage == 0 and args.skip_first:
                continue
            logger.info('-' * 100)
            logger.info(f"Stage {stage}: Epoch {ep}")
            logger.info("Training process")
            model.train()
            logger.info("Training batch:")
            iter_cnt = 0
            for bt, batch in enumerate(tqdm(stage_loader)):
                iter_cnt += 1

                train_x, train_y, train_masks, train_span = zip(*batch)
                train_x = torch.LongTensor(train_x).to(device)
                train_masks = torch.LongTensor(train_masks).to(device)
                train_y = [torch.LongTensor(item).to(device) for item in train_y]
                train_span = [torch.LongTensor(item).to(device) for item in train_span]
                # if args.dataset == "ACE":
                #     return_dict = model(train_x, train_masks)
                # else:
                return_dict = model(train_x, train_masks, train_span)
                outputs, context_feat, trig_feat = return_dict['outputs'], return_dict['context_feat'], return_dict['trig_feat']

                sim_event_loss = 0
                if args.sim_event_type:
                    extended_event_type_reps = []
                    for y in train_y:
                        for y_i in y:
                            if y_i != 0:
                                extended_event_type_reps.append(event_type_reps[y_i.item() - 1, :])
                    extended_event_type_reps = torch.stack(extended_event_type_reps)
                    sim_event_loss = criterion_fd(extended_event_type_reps, trig_feat[torch.cat(train_y) != 0], torch.ones(extended_event_type_reps.shape[0]).to(device))


                for i in range(len(train_y)):
                    invalid_mask_label = torch.BoolTensor([item not in learned_types for item in train_y[i]]).to(device)
                    train_y[i].masked_fill_(invalid_mask_label, 0)
                # outputs[:, 0] = 0
                loss, loss_ucl, loss_aug, loss_fd, loss_pd, loss_tlcl, loss_dr = sim_event_loss, 0, 0, 0, 0, 0, 0
                ce_y = torch.cat(train_y)
                ce_outputs = outputs
                if (args.ucl or args.tlcl) and (stage > 0 or (args.skip_first_cl != "ucl+tlcl" and stage == 0)):                        
                    reps = return_dict['reps']
                    bs, hdim = reps.shape
                    aug_repeat_times = args.aug_repeat_times
                    da_x = train_x.clone().repeat((aug_repeat_times, 1))
                    da_y = train_y * aug_repeat_times
                    da_masks = train_masks.repeat((aug_repeat_times, 1))
                    da_span = train_span * aug_repeat_times
                    tk_len = torch.count_nonzero(da_masks, dim=-1) - 2
                    perm = [torch.randperm(item).to(device) + 1 for item in tk_len]
                    if args.cl_aug == "shuffle":
                        for i in range(len(tk_len)):
                            da_span[i] = torch.where(da_span[i].unsqueeze(2) == perm[i].unsqueeze(0).unsqueeze(0))[2].view(-1, 2) + 1
                            da_x[i, 1: 1+tk_len[i]] = da_x[i, perm[i]]
                    elif args.cl_aug =="RTR":
                        rand_ratio = 0.25
                        rand_num = (rand_ratio * tk_len).int()
                        special_ids = [103, 102, 101, 100, 0]
                        all_ids = torch.arange(model.backbone.config.vocab_size).to(device)
                        special_token_mask = torch.ones(model.backbone.config.vocab_size).to(device)
                        special_token_mask[special_ids] = 0
                        all_tokens = all_ids.index_select(0, special_token_mask.nonzero().squeeze())
                        for i in range(len(rand_num)):
                            token_idx = torch.arange(tk_len[i]).to(device) + 1
                            trig_mask = torch.ones(token_idx.shape).to(device)
                            if args.dataset == "ACE":
                                span_pos = da_span[i][da_y[i].nonzero()].view(-1).unique() - 1
                            else:
                                span_pos = da_span[i].view(-1).unique() - 1
                            trig_mask[span_pos] = 0
                            token_idx_ntrig = token_idx.index_select(0, trig_mask.nonzero().squeeze())
                            replace_perm = torch.randperm(token_idx_ntrig.shape.numel())
                            replace_idx = token_idx_ntrig[replace_perm][:rand_num[i]]
                            new_tkn_idx = torch.randperm(len(all_tokens))[:rand_num[i]]
                            da_x[i, replace_idx] = all_tokens[new_tkn_idx].to(device)
                    # if args.dataset == "ACE":
                    #     da_return_dict = model(da_x, da_masks)
                    # else:
                    da_return_dict = model(da_x, da_masks, da_span)
                    da_outputs, da_reps, da_context_feat, da_trig_feat = da_return_dict['outputs'], da_return_dict['reps'], da_return_dict['context_feat'], da_return_dict['trig_feat']
                    
                    if args.ucl:
                        if not ((args.skip_first_cl == "ucl" or args.skip_first_cl == "ucl+tlcl") and stage == 0):
                            ucl_reps = torch.cat([reps, da_reps])
                            ucl_reps = normalize(ucl_reps, dim=-1)
                            Adj_mask_ucl = torch.zeros(bs * (1 + aug_repeat_times), bs * (1 + aug_repeat_times)).to(device)
                            for i in range(aug_repeat_times):
                                Adj_mask_ucl += torch.eye(bs * (1 + aug_repeat_times)).to(device)
                                Adj_mask_ucl = torch.roll(Adj_mask_ucl, bs, -1)                    
                            loss_ucl = compute_CLLoss(Adj_mask_ucl, ucl_reps, bs * (1 + aug_repeat_times))
                    if args.tlcl:
                        if not ((args.skip_first_cl == "tlcl" or args.skip_first_cl == "ucl+tlcl") and stage == 0):
                            tlcl_feature = torch.cat([trig_feat, da_trig_feat])
                            tlcl_lbs = torch.cat(train_y + da_y)

                            if (args.aug_dropout_times > 0):
                                dr_trig_feat, dr_y = dropout_augment(trig_feat, train_y)
                                tlcl_feature = torch.cat([tlcl_feature, dr_trig_feat])
                                tlcl_lbs = torch.cat([tlcl_lbs, dr_y])

                            tlcl_feature = normalize(tlcl_feature, dim=-1)
                            mat_size = tlcl_feature.shape[0]
                            tlcl_lbs_oh = F.one_hot(tlcl_lbs).float()
                            Adj_mask_tlcl = torch.matmul(tlcl_lbs_oh, tlcl_lbs_oh.T)
                            Adj_mask_tlcl = Adj_mask_tlcl * (torch.ones(mat_size) - torch.eye(mat_size)).to(device)
                            loss_tlcl = compute_CLLoss(Adj_mask_tlcl, tlcl_feature, mat_size)

                    loss = loss + loss_ucl + loss_tlcl
                    if args.joint_da_loss == "ce" or args.joint_da_loss == "mul":
                        ce_y = torch.cat(train_y + da_y)
                        ce_outputs = torch.cat([outputs, da_outputs])

                # dropout augment
                if (args.aug_dropout_times > 0) and (stage == 0):
                    dr_trig_feat, dr_y = dropout_augment(trig_feat, train_y)
                    dr_feature = torch.cat([trig_feat, dr_trig_feat])
                    dr_lbs = torch.cat([torch.cat(train_y), dr_y])
                    dr_feature = normalize(dr_feature, dim=-1)
                    mat_size = dr_feature.shape[0]
                    dr_lbs_oh = F.one_hot(dr_lbs).float()
                    Adj_mask_dr = torch.matmul(dr_lbs_oh, dr_lbs_oh.T)
                    Adj_mask_dr = Adj_mask_dr * (torch.ones(mat_size) - torch.eye(mat_size)).to(device)
                    loss_dr = compute_CLLoss(Adj_mask_dr, dr_feature, mat_size)

                    loss += loss_dr
                
                if args.pseudo_label and (stage > 0 ):
                    prev_model.eval()
                    with torch.no_grad():
                        prev_return_dict = prev_model(train_x, train_masks, train_span)
                        prev_outputs, prev_feature = prev_return_dict['outputs'], prev_return_dict['context_feat']

                        prev_predict = prev_outputs[:, prev_learned_types].argmax(-1)
                        ce_y = prev_predict * (ce_y == 0).int() + ce_y
                
                ce_y = torch.cat(train_y)
                ce_outputs = ce_outputs[:, learned_types]
                loss_ce = criterion_ce(ce_outputs, ce_y)
                loss = loss + loss_ce
                w = len(prev_learned_types) / len(learned_types)

                if args.rep_aug != "none" and stage > 0:
                    outputs_aug, aug_y = [], []
                    for e_batch in e_loader:
                        exemplar_x, exemplars_y, exemplar_masks, exemplar_span = zip(*e_batch)
                        exemplar_radius = [exemplars.radius[y[0]] for y in exemplars_y]
                        exemplar_x = torch.LongTensor(exemplar_x).to(device)
                        exemplar_masks = torch.LongTensor(exemplar_masks).to(device)
                        exemplars_y = [torch.LongTensor(item).to(device) for item in exemplars_y]
                        exemplar_span = [torch.LongTensor(item).to(device) for item in exemplar_span]            
                        if args.rep_aug == "relative":
                            aug_return_dict = model(exemplar_x, exemplar_masks, exemplar_span, torch.sqrt(torch.stack(exemplar_radius)).unsqueeze(-1))
                        else:
                            aug_return_dict = model(exemplar_x, exemplar_masks, exemplar_span, torch.sqrt(torch.stack(list(exemplars.radius.values())).mean()))
                        output_aug = aug_return_dict['outputs_aug']
                        outputs_aug.append(output_aug)
                        aug_y.extend(exemplars_y)
                    outputs_aug = torch.cat(outputs_aug)
                    if args.leave_zero:
                        outputs_aug[:, 0] = 0
                    outputs_aug = outputs_aug[:, learned_types].squeeze(-1)
                    loss_aug = criterion_ce(outputs_aug, torch.cat(aug_y))
                    # loss = loss_ce * w + loss_aug * (1 - w)
                    # loss = loss_ce * (1 - w) + loss_aug * w
                    loss = args.gamma * loss + args.theta * loss_aug
                    
                    
                if stage > 0 and args.distill != "none":
                    prev_model.eval()
                    with torch.no_grad():
                        prev_return_dict = prev_model(train_x, train_masks, train_span)
                        prev_outputs, prev_feature = prev_return_dict['outputs'], prev_return_dict['context_feat']

                        if args.joint_da_loss == "dist" or args.joint_da_loss == "mul":
                            outputs = torch.cat([outputs, da_outputs])
                            context_feat = torch.cat([context_feat, da_context_feat])
                            prev_return_dict_cl = prev_model(da_x, da_masks, da_span)
                            prev_outputs_cl, prev_feature_cl = prev_return_dict_cl['outputs'], prev_return_dict_cl['context_feat']
                            prev_outputs, prev_feature = torch.cat([prev_outputs, prev_outputs_cl]), torch.cat([prev_feature, prev_feature_cl])
                    # prev_invalid_mask_op = torch.BoolTensor([item not in prev_learned_types for item in range(args.class_num)]).to(device)
                    prev_valid_mask_op = torch.nonzero(torch.BoolTensor([item in prev_learned_types for item in range(args.class_num + 1)]).to(device))
                    if args.distill == "fd" or args.distill == "mul":
                        prev_feature = normalize(prev_feature.view(-1, prev_feature.shape[-1]), dim=-1)
                        cur_feature = normalize(context_feat.view(-1, prev_feature.shape[-1]), dim=-1)
                        loss_fd = criterion_fd(prev_feature, cur_feature, torch.ones(prev_feature.size(0)).to(device)) # TODO: Don't know whether the code is right
                    else:
                        loss_fd = 0
                    if args.distill == "pd" or args.distill == "mul":
                        T = args.temperature
                        if args.leave_zero:
                            prev_outputs[:, 0] = 0
                        prev_outputs = prev_outputs[:, prev_valid_mask_op].squeeze(-1)
                        cur_outputs = outputs[:, prev_valid_mask_op].squeeze(-1)
                        # prev_outputs[i].masked_fill_(prev_invalid_mask_op, torch.Tensor([float("-inf")]).squeeze(0))
                        prev_p = torch.softmax(prev_outputs / T, dim= -1)
                        p = torch.log_softmax(cur_outputs / T, dim = -1)
                        loss_pd = -torch.mean(torch.sum(prev_p * p, dim = -1), dim = 0)
                    else:
                        loss_pd = 0
                    # loss_pd = criterion_pd(torch.cat([item / T for item in outputs]), torch.cat([item / T for item in prev_outputs]))
                    if args.dweight_loss and stage > 0:
                        if (not args.sam) or (args.sam_type == "full"):
                            loss = loss * (1 - w) + (loss_fd + loss_pd) * w
                    else:
                        if (not args.sam) or (args.sam_type == "full"):
                            loss = loss + args.alpha * loss_fd + args.beta * loss_pd
                if not args.sam:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                else:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.first_step(zero_grad=True)

                    return_dict = model(train_x, train_masks, train_span)
                    outputs, context_feat, trig_feat = return_dict['outputs'], return_dict['context_feat'], return_dict['trig_feat']

                    sim_event_loss = 0
                    if args.sim_event_type:
                        extended_event_type_reps = []
                        for y in train_y:
                            for y_i in y:
                                if y_i != 0:
                                    extended_event_type_reps.append(event_type_reps[y_i.item() - 1, :])
                        extended_event_type_reps = torch.stack(extended_event_type_reps)
                        sim_event_loss = criterion_fd(extended_event_type_reps, trig_feat[torch.cat(train_y) != 0], torch.ones(extended_event_type_reps.shape[0]).to(device))


                    for i in range(len(train_y)):
                        invalid_mask_label = torch.BoolTensor([item not in learned_types for item in train_y[i]]).to(device)
                        train_y[i].masked_fill_(invalid_mask_label, 0)
                    # outputs[:, 0] = 0
                    loss, loss_ucl, loss_aug, loss_fd, loss_pd, loss_tlcl, loss_dr = sim_event_loss, 0, 0, 0, 0, 0, 0
                    ce_y = torch.cat(train_y)
                    ce_outputs = outputs
                    if (args.ucl or args.tlcl) and (stage > 0 or (args.skip_first_cl != "ucl+tlcl" and stage == 0)):                        
                        reps = return_dict['reps']
                        bs, hdim = reps.shape
                        aug_repeat_times = args.aug_repeat_times
                        da_x = train_x.clone().repeat((aug_repeat_times, 1))
                        da_y = train_y * aug_repeat_times
                        da_masks = train_masks.repeat((aug_repeat_times, 1))
                        da_span = train_span * aug_repeat_times
                        tk_len = torch.count_nonzero(da_masks, dim=-1) - 2
                        perm = [torch.randperm(item).to(device) + 1 for item in tk_len]
                        if args.cl_aug == "shuffle":
                            for i in range(len(tk_len)):
                                da_span[i] = torch.where(da_span[i].unsqueeze(2) == perm[i].unsqueeze(0).unsqueeze(0))[2].view(-1, 2) + 1
                                da_x[i, 1: 1+tk_len[i]] = da_x[i, perm[i]]
                        elif args.cl_aug =="RTR":
                            rand_ratio = 0.25
                            rand_num = (rand_ratio * tk_len).int()
                            special_ids = [103, 102, 101, 100, 0]
                            all_ids = torch.arange(model.backbone.config.vocab_size).to(device)
                            special_token_mask = torch.ones(model.backbone.config.vocab_size).to(device)
                            special_token_mask[special_ids] = 0
                            all_tokens = all_ids.index_select(0, special_token_mask.nonzero().squeeze())
                            for i in range(len(rand_num)):
                                token_idx = torch.arange(tk_len[i]).to(device) + 1
                                trig_mask = torch.ones(token_idx.shape).to(device)
                                if args.dataset == "ACE":
                                    span_pos = da_span[i][da_y[i].nonzero()].view(-1).unique() - 1
                                else:
                                    span_pos = da_span[i].view(-1).unique() - 1
                                trig_mask[span_pos] = 0
                                token_idx_ntrig = token_idx.index_select(0, trig_mask.nonzero().squeeze())
                                replace_perm = torch.randperm(token_idx_ntrig.shape.numel())
                                replace_idx = token_idx_ntrig[replace_perm][:rand_num[i]]
                                new_tkn_idx = torch.randperm(len(all_tokens))[:rand_num[i]]
                                da_x[i, replace_idx] = all_tokens[new_tkn_idx].to(device)
                        # if args.dataset == "ACE":
                        #     da_return_dict = model(da_x, da_masks)
                        # else:
                        da_return_dict = model(da_x, da_masks, da_span)
                        da_outputs, da_reps, da_context_feat, da_trig_feat = da_return_dict['outputs'], da_return_dict['reps'], da_return_dict['context_feat'], da_return_dict['trig_feat']
                        
                        if args.ucl:
                            if not ((args.skip_first_cl == "ucl" or args.skip_first_cl == "ucl+tlcl") and stage == 0):
                                ucl_reps = torch.cat([reps, da_reps])
                                ucl_reps = normalize(ucl_reps, dim=-1)
                                Adj_mask_ucl = torch.zeros(bs * (1 + aug_repeat_times), bs * (1 + aug_repeat_times)).to(device)
                                for i in range(aug_repeat_times):
                                    Adj_mask_ucl += torch.eye(bs * (1 + aug_repeat_times)).to(device)
                                    Adj_mask_ucl = torch.roll(Adj_mask_ucl, bs, -1)                    
                                loss_ucl = compute_CLLoss(Adj_mask_ucl, ucl_reps, bs * (1 + aug_repeat_times))
                        if args.tlcl:
                            if not ((args.skip_first_cl == "tlcl" or args.skip_first_cl == "ucl+tlcl") and stage == 0):
                                tlcl_feature = torch.cat([trig_feat, da_trig_feat])
                                tlcl_lbs = torch.cat(train_y + da_y)

                                if (args.aug_dropout_times > 0):
                                    dr_trig_feat, dr_y = dropout_augment(trig_feat, train_y)
                                    tlcl_feature = torch.cat([tlcl_feature, dr_trig_feat])
                                    tlcl_lbs = torch.cat([tlcl_lbs, dr_y])

                                tlcl_feature = normalize(tlcl_feature, dim=-1)
                                mat_size = tlcl_feature.shape[0]
                                tlcl_lbs_oh = F.one_hot(tlcl_lbs).float()
                                Adj_mask_tlcl = torch.matmul(tlcl_lbs_oh, tlcl_lbs_oh.T)
                                Adj_mask_tlcl = Adj_mask_tlcl * (torch.ones(mat_size) - torch.eye(mat_size)).to(device)
                                loss_tlcl = compute_CLLoss(Adj_mask_tlcl, tlcl_feature, mat_size)

                        loss = loss + loss_ucl + loss_tlcl
                        if args.joint_da_loss == "ce" or args.joint_da_loss == "mul":
                            ce_y = torch.cat(train_y + da_y)
                            ce_outputs = torch.cat([outputs, da_outputs])

                    # dropout augment
                    if (args.aug_dropout_times > 0) and (stage == 0):
                        dr_trig_feat, dr_y = dropout_augment(trig_feat, train_y)
                        dr_feature = torch.cat([trig_feat, dr_trig_feat])
                        dr_lbs = torch.cat([torch.cat(train_y), dr_y])
                        dr_feature = normalize(dr_feature, dim=-1)
                        mat_size = dr_feature.shape[0]
                        dr_lbs_oh = F.one_hot(dr_lbs).float()
                        Adj_mask_dr = torch.matmul(dr_lbs_oh, dr_lbs_oh.T)
                        Adj_mask_dr = Adj_mask_dr * (torch.ones(mat_size) - torch.eye(mat_size)).to(device)
                        loss_dr = compute_CLLoss(Adj_mask_dr, dr_feature, mat_size)

                        loss += loss_dr
                    
                    if args.pseudo_label and (stage > 0 ):
                        prev_model.eval()
                        with torch.no_grad():
                            prev_return_dict = prev_model(train_x, train_masks, train_span)
                            prev_outputs, prev_feature = prev_return_dict['outputs'], prev_return_dict['context_feat']

                            prev_predict = prev_outputs[:, prev_learned_types].argmax(-1)
                            ce_y = prev_predict * (ce_y == 0).int() + ce_y
                    
                    ce_y = torch.cat(train_y)
                    ce_outputs = ce_outputs[:, learned_types]
                    loss_ce = criterion_ce(ce_outputs, ce_y)
                    loss = loss + loss_ce
                    w = len(prev_learned_types) / len(learned_types)

                    if args.rep_aug != "none" and stage > 0:
                        outputs_aug, aug_y = [], []
                        for e_batch in e_loader:
                            exemplar_x, exemplars_y, exemplar_masks, exemplar_span = zip(*e_batch)
                            exemplar_radius = [exemplars.radius[y[0]] for y in exemplars_y]
                            exemplar_x = torch.LongTensor(exemplar_x).to(device)
                            exemplar_masks = torch.LongTensor(exemplar_masks).to(device)
                            exemplars_y = [torch.LongTensor(item).to(device) for item in exemplars_y]
                            exemplar_span = [torch.LongTensor(item).to(device) for item in exemplar_span]            
                            if args.rep_aug == "relative":
                                aug_return_dict = model(exemplar_x, exemplar_masks, exemplar_span, torch.sqrt(torch.stack(exemplar_radius)).unsqueeze(-1))
                            else:
                                aug_return_dict = model(exemplar_x, exemplar_masks, exemplar_span, torch.sqrt(torch.stack(list(exemplars.radius.values())).mean()))
                            output_aug = aug_return_dict['outputs_aug']
                            outputs_aug.append(output_aug)
                            aug_y.extend(exemplars_y)
                        outputs_aug = torch.cat(outputs_aug)
                        if args.leave_zero:
                            outputs_aug[:, 0] = 0
                        outputs_aug = outputs_aug[:, learned_types].squeeze(-1)
                        loss_aug = criterion_ce(outputs_aug, torch.cat(aug_y))
                        # loss = loss_ce * w + loss_aug * (1 - w)
                        # loss = loss_ce * (1 - w) + loss_aug * w
                        loss = args.gamma * loss + args.theta * loss_aug
                        
                        
                    if stage > 0 and args.distill != "none":
                        prev_model.eval()
                        with torch.no_grad():
                            prev_return_dict = prev_model(train_x, train_masks, train_span)
                            prev_outputs, prev_feature = prev_return_dict['outputs'], prev_return_dict['context_feat']

                            if args.joint_da_loss == "dist" or args.joint_da_loss == "mul":
                                outputs = torch.cat([outputs, da_outputs])
                                context_feat = torch.cat([context_feat, da_context_feat])
                                prev_return_dict_cl = prev_model(da_x, da_masks, da_span)
                                prev_outputs_cl, prev_feature_cl = prev_return_dict_cl['outputs'], prev_return_dict_cl['context_feat']
                                prev_outputs, prev_feature = torch.cat([prev_outputs, prev_outputs_cl]), torch.cat([prev_feature, prev_feature_cl])
                        # prev_invalid_mask_op = torch.BoolTensor([item not in prev_learned_types for item in range(args.class_num)]).to(device)
                        prev_valid_mask_op = torch.nonzero(torch.BoolTensor([item in prev_learned_types for item in range(args.class_num + 1)]).to(device))
                        if args.distill == "fd" or args.distill == "mul":
                            prev_feature = normalize(prev_feature.view(-1, prev_feature.shape[-1]), dim=-1)
                            cur_feature = normalize(context_feat.view(-1, prev_feature.shape[-1]), dim=-1)
                            loss_fd = criterion_fd(prev_feature, cur_feature, torch.ones(prev_feature.size(0)).to(device)) # TODO: Don't know whether the code is right
                        else:
                            loss_fd = 0
                        if args.distill == "pd" or args.distill == "mul":
                            T = args.temperature
                            if args.leave_zero:
                                prev_outputs[:, 0] = 0
                            prev_outputs = prev_outputs[:, prev_valid_mask_op].squeeze(-1)
                            cur_outputs = outputs[:, prev_valid_mask_op].squeeze(-1)
                            # prev_outputs[i].masked_fill_(prev_invalid_mask_op, torch.Tensor([float("-inf")]).squeeze(0))
                            prev_p = torch.softmax(prev_outputs / T, dim= -1)
                            p = torch.log_softmax(cur_outputs / T, dim = -1)
                            loss_pd = -torch.mean(torch.sum(prev_p * p, dim = -1), dim = 0)
                        else:
                            loss_pd = 0
                        # loss_pd = criterion_pd(torch.cat([item / T for item in outputs]), torch.cat([item / T for item in prev_outputs]))
                        if args.dweight_loss and stage > 0:
                            loss = loss * (1 - w) + (loss_fd + loss_pd) * w
                        else:
                            loss = loss + args.alpha * loss_fd + args.beta * loss_pd

                    loss.backward()
                    optimizer.second_step(zero_grad=True)
                    


                
            logger.info(f'loss_ce: {loss_ce}')
            logger.info(f'loss_ucl: {loss_ucl}')
            logger.info(f'loss_tlcl: {loss_tlcl}')
            logger.info(f'loss_dr: {loss_dr}')
            logger.info(f'loss_aug: {loss_aug}')
            logger.info(f'sim_event_loss: {sim_event_loss}')
            logger.info(f'loss_fd: {loss_fd}')
            logger.info(f'loss_pd: {loss_pd}')
            logger.info(f'loss_all: {loss}')

            if ((ep + 1) % args.eval_freq == 0 and args.early_stop) or (ep + 1) == args.epochs: # TODO TODO
                # Evaluation process
                logger.info("Evaluation process")
                model.eval()
                with torch.no_grad():
                    if args.single_label:
                        eval_dataset = collect_eval_sldataset(args.dataset, args.data_root, 'test', label2idx, None, [i for item in streams for i in item])
                    else:
                        eval_dataset = collect_dataset(args.dataset, args.data_root, 'test', label2idx, None, [i for item in streams for i in item])
                    eval_loader = DataLoader(
                        dataset=eval_dataset,
                        shuffle=False,
                        batch_size=128,
                        collate_fn=lambda x:x)
                    calcs = Calculator()
                    for batch in tqdm(eval_loader):
                        eval_x, eval_y, eval_masks, eval_span = zip(*batch)
                        eval_x = torch.LongTensor(eval_x).to(device)
                        eval_masks = torch.LongTensor(eval_masks).to(device)
                        eval_y = [torch.LongTensor(item).to(device) for item in eval_y]
                        eval_span = [torch.LongTensor(item).to(device) for item in eval_span]  
                        eval_return_dict = model(eval_x, eval_masks, eval_span)
                        eval_outputs = eval_return_dict['outputs']
                        valid_mask_eval_op = torch.BoolTensor([idx in learned_types for idx in range(args.class_num + 1)]).to(device)
                        for i in range(len(eval_y)):
                            invalid_mask_eval_label = torch.BoolTensor([item not in learned_types for item in eval_y[i]]).to(device)
                            eval_y[i].masked_fill_(invalid_mask_eval_label, 0)
                        if args.leave_zero:
                            eval_outputs[:, 0] = 0
                        eval_outputs = eval_outputs[:, valid_mask_eval_op].squeeze(-1)
                        calcs.extend(eval_outputs.argmax(-1), torch.cat(eval_y))
                    bc, (precision, recall, micro_F1) = calcs.by_class(learned_types)
                    if args.log:
                        writer.add_scalar(f'score/epoch/marco_F1', micro_F1,  ep + 1 + args.epochs * stage)
                    if args.log and (ep + 1) == args.epochs:
                        writer.add_scalar(f'score/stage/marco_F1', micro_F1, stage)
                    
                    if args.early_stop:
                        if dev_score is None or dev_score < micro_F1:
                            no_better = 0
                            dev_score = micro_F1
                            # torch.save(model.state_dict(), e_pth)
                        else:
                            no_better += 1
                            logger.info(f'No better: {no_better}/{args.patience}')
                        # if no_better >= args.patience:
                        #     logger.info("Early stopping with dev_score: " + str(dev_score))
                        #     logger.info(f'marco F1 {micro_F1}')
                        #     dev_scores_ls.append(dev_score)
                        #     logger.info(f"Dev scores list: {dev_scores_ls}")
                        #     logger.info(f"bc:{bc}")
                        #     if args.log:
                        #         writer.add_scalar(f'score/stage/marco_F1', micro_F1, stage)
                        #     break
                    
                    if (ep + 1) == args.epochs:
                        logger.info("Early stopping with dev_score: " + str(dev_score))
                        logger.info(f'marco F1 {micro_F1}')
                        dev_scores_ls.append(dev_score if dev_score else micro_F1)
                        logger.info(f"Dev scores list: {dev_scores_ls}")
                        logger.info(f"bc:{bc}")
                    


        for tp in streams_indexed[stage]:
            if not tp == 0:
                labels.pop(labels.index(tp))
        save_stage = stage
        if args.save_dir and local_rank == 0:
            state = {'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'stage':stage + 1, 
                            'labels':labels, 'learned_types':learned_types, 'prev_learned_types':prev_learned_types}
            save_pth = os.path.join(args.save_dir, "perm" + str(args.perm_id))
            save_name = f"stage_{save_stage}_{cur_time}.pth"
            if not os.path.exists(save_pth):
                os.makedirs(save_pth)
            logger.info(f'state_dict saved to: {os.path.join(save_pth, save_name)}')
            torch.save(state, os.path.join(save_pth, save_name))
            os.remove(e_pth)





if __name__ == "__main__":
    args = parse_arguments()
    if args.parallel == 'DDP':
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        mp.spawn(train,
            args=(args, ),
            nprocs=args.world_size,
            join=True)
    else:
        train(0, args)
        

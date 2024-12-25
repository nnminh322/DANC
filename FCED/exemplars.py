import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from configs import parse_arguments
from transformers import AutoTokenizer
# from openai import OpenAI
import json
import google.generativeai as genai
# from kaggle_secrets import UserSecretsClient

# user_secrets = UserSecretsClient()
# api_key = user_secrets.get_secret("GEMINI_API_KEY")
genai.configure(api_key = "AIzaSyBirlOrEJBDn9irZTST7LAISKfhRgaJy9o")

# Generate content
model = genai.GenerativeModel(model_name='gemini-1.5-flash-latest')


# client = OpenAI(api_key="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")

args = parse_arguments()
tokenizer = AutoTokenizer.from_pretrained(args.backbone)

LABEL2EVENT_TYPE = {
    "MAVEN": {
        85: "Competition",
        6: "Causation",
        20: "Hostile encounter",
        22: "Conquering",
        32: "Process start",
        11: "Motion",
        83: "Social event",
        4: "Catastrophe",
        21: "Killing",
        24: "Attack"
    },
    "ACE": {
        9: "Contact: Phone-Write",
        2: "Movement: Transport",
        6: "Personnel: Elect",
        3: "Life: Die",
        1: "Conflict: Attack",
        5: "Personnel: End-Position",
        7: "Transaction: Transfer-Money",
        10: "Justice: Trial-Hearing",
        8: "Life: Injure",
        4: "Contact: Meet",
        11: "Justice: Charge-Indict",
        12: "Transaction: Transfer-Ownership",
        13: "Personnel: Start-Position",
        14: "Justice: Sentence",
        15: "Justice: Arrest-Jail",
        16: "Life: Marry",
        17: "Conflict: Demonstrate",
        18: "Justice:Convict",
        19: "Justice:Sue",
        20: "Life:Be-Born"
    }
}

PROMPT = """A sample in event detection datasets comprises a context, a trigger word, and an event type that corresponds to the trigger phrase.
Here is an example:
{{
    "context": "{context}",
    "trigger_word": "{trigger}",
    "event_type": "{event_type}"
}}

Please generate {n} samples for event type "{event_type}" in LIST of JSON format:
"""

def api_call(input_prompt: str):
    retry = 3
    while retry > 0:
        try:
            # completion = client.chat.completions.create(
            #     model="gpt-4o",
            #     messages=[
            #         {"role": "system", "content": "You are a helpful assistant."},
            #         {
            #             "role": "user",
            #             "content": input_prompt
            #         }
            #     ]
            # )
            # content = completion.choices[0].message.content
            response = model.generate_content(input_prompt)
            content = response.text
            content = "[" + content.split("[")[-1].split("]")[0] + "]"
            return json.loads(content)
        except:
            retry -= 1
    raise Exception("Error: Retried 3 times!")

def get_span(piece_ids, trigger):
    mid = len(tokenizer.tokenize(trigger))
    for step in range(max(1, mid - 2), mid + 3):
        for i in range(len(piece_ids) - step + 1):
            if tokenizer.decode(piece_ids[i:i+step]) == trigger:
                span = [i, i+step-1]
                return span
    
    raise Exception(f"Error: trigger: {trigger} \t context: {tokenizer.decode(piece_ids, skip_special_tokens=True)}")

def llm_augment(x: list[int], y: list[int], span: list[list[int]], label2idx):
    print("______________LLM Augmenting________________")
    context = tokenizer.decode(x, skip_special_tokens=True)
    idx2label = {v: k for k, v in label2idx.items()}
    event_type = LABEL2EVENT_TYPE[args.dataset][idx2label[y[0]]]
    span = span[0]
    trigger = tokenizer.decode(x[span[0]:span[1] + 1])
    
    prompt = PROMPT.format(context=context, trigger=trigger, event_type=event_type, n=args.llm_augment_times)

    augmented_samples = api_call(prompt)


    max_seqlen = len(x)
    augmented_x, augmented_y, augmented_mask, augmented_span = [], [], [], []
    for sample in augmented_samples:
        input_ids = tokenizer(sample["context"], padding="max_length", max_length=max_seqlen, truncation=True)
        piece_ids = input_ids['input_ids']
        mask = input_ids['attention_mask']
        try:
            a_span = [get_span(piece_ids, sample['trigger_word'])]
            augmented_x.append(piece_ids)
            augmented_y.append(y)
            augmented_mask.append(mask)
            augmented_span.append(a_span)
        except:
            pass
    return augmented_x, augmented_y, augmented_mask, augmented_span





class Exemplars():
    def __init__(self) -> None:
        # self.exemplars = {}
        self.learned_nums = 0
        self.memory_size = args.enum * self.learned_nums if args.fixed_enum else args.enum
        self.exemplars_x = []
        self.exemplars_mask = []
        self.exemplars_y = []
        self.exemplars_span = []
        self.radius ={}

        self.llm_augmenteds_x = []
        self.llm_augmenteds_mask = []
        self.llm_augmenteds_y = []
        self.llm_augmenteds_span = []


    def __len__(self):
        return self.memory_size
    def get_exemplar_loader(self):
        x = [item for t in self.exemplars_x for item in t]
        y = [item for t in self.exemplars_y for item in t]
        mask = [item for t in self.exemplars_mask for item in t]
        span = [item for t in self.exemplars_span for item in t]    
        return (x, mask, y, span ,self.radius)

    def rm_exemplars(self, exemplar_num):
        if self.exemplars_x != [] and exemplar_num > len(self.exemplars_x[0]):
            self.exemplars_x = [i[:exemplar_num] for i in self.exemplars_x]
            self.exemplars_mask = [i[:exemplar_num] for i in self.exemplars_mask]
            self.exemplars_y = [i[:exemplar_num] for i in self.exemplars_y]
            self.exemplars_span = [i[:exemplar_num] for i in self.exemplars_span]

    def get_event_type_ids(self, learned_nums, device, label2idx):
        idx2label = {v: k for k, v in label2idx.items()}
        del idx2label[0]
        label_names = []
        for idx in range(learned_nums):
            label_name = LABEL2EVENT_TYPE[args.dataset][idx2label[idx+1]]
            label_names.append(label_name)

        label_ids = tokenizer(label_names, padding="max_length", max_length=20, truncation=True, return_tensors="pt").to(device)
        return label_ids['input_ids'], label_ids['attention_mask']

    
    def set_exemplars(self, model: nn.Module, exemplar_loader: DataLoader, learned_nums, device, label2idx=None):
        self.learned_nums = learned_nums - 1 if learned_nums > 0 else 1
        if args.fixed_enum:
            exemplar_num = args.enum
            self.memory_size = exemplar_num * self.learned_nums
        else:
            exemplar_num = int(self.memory_size / self.learned_nums)
            self.rm_exemplars(exemplar_num)
        rep_dict, data_dict = {}, {}
        model.eval()
        with torch.no_grad():
            print("Setting exemplars, loading exemplar batch:")
            for batch in tqdm(exemplar_loader):
                data_x, data_y, data_masks, data_span = zip(*batch)
                # tensor_x = torch.LongTensor(data_x).to('cpu')
                # tensor_masks = torch.LongTensor(data_masks).to('cpu')
                tensor_x = torch.LongTensor(data_x).to(device)
                tensor_masks = torch.LongTensor(data_masks).to(device)
                if args.parallel == 'DP':
                    rep = model.module.forward_backbone(tensor_x, tensor_masks)
                else:
                    rep = model.forward_backbone(tensor_x, tensor_masks)

                for i in range(rep.size(0)):
                    for j, label in enumerate(data_y[i]):
                        if label != 0:
                            if not label in rep_dict:
                                rep_dict[label], data_dict[label] = [], []
                            # data_dict[label].append([data_x[i], data_y[i], data_masks[i], data_span[i]])
                            data_dict[label].append([data_x[i], [label], data_masks[i], [data_span[i][j]]])
                            rep_dict[label].append(rep[i, 0, :].squeeze(0))
                # if len(rep_dict) > 20: # TODO: test use
                #     break
            for l, reps in rep_dict.items():
                reps = torch.stack(reps)
                radius = torch.mean(torch.var(reps, dim=0)) if reps.shape[0] > 1 else torch.tensor(0).to(device)
                # dt, lb, sp = zip(*data_dict[l])
                data_ls = data_dict[l]
                if exemplar_num > reps.size(0): # if reps num is not enough, up sampling 
                    repeat_times = int(exemplar_num / reps.size(0)) + 1
                    reps = reps.repeat(repeat_times, 1)
                    data_ls = data_ls * repeat_times
                # data_ls = np.asarray(data_ls)
                prototype_rep = reps.mean(0)
                dist = torch.sqrt(torch.sum(torch.square(prototype_rep - reps), dim=1))
                reps_num = exemplar_num
                topk_dist_idx = torch.topk(dist, reps_num, largest=False).indices.to('cpu')
                # self.exemplars[label] = torch.cat([self.exemplars[label], reps[topk_dist_idx, :]], 0)
                # data_topk = dt[topk_dist_idx]
                # label_topk = lb[topk_dist_idx]
                # span_topk = sp[topk_dist_idx]
                exemplar_x, exemplar_y, exemplar_mask, exemplar_span = [], [], [], []
                for idx in list(topk_dist_idx):
                    exemplar_x.append(data_ls[idx][0])
                    exemplar_y.append(data_ls[idx][1])
                    exemplar_mask.append(data_ls[idx][2])
                    exemplar_span.append(data_ls[idx][3])
                    
                    if args.llm_augment:
                        augmented_x, augmented_y, augmented_mask, augmented_span = llm_augment(data_ls[idx][0], data_ls[idx][1], data_ls[idx][3], label2idx)
                        self.llm_augmenteds_x += augmented_x
                        self.llm_augmenteds_y += augmented_y
                        self.llm_augmenteds_mask += augmented_mask
                        self.llm_augmenteds_span += augmented_span
                        
                
                self.exemplars_x.append(list(exemplar_x))
                self.exemplars_y.append(list(exemplar_y))
                # self.exemplars_y.append(list(data_topk[:, 1]))
                self.exemplars_mask.append(list(exemplar_mask))
                self.exemplars_span.append(list(exemplar_span))
                self.radius[l] = radius
        
    def build_stage_loader(self, dataset, collate_fn=lambda x:x):
        if len(dataset) > 0:
            dataset.extend(self.llm_augmenteds_x, self.llm_augmenteds_y, self.llm_augmenteds_mask, self.llm_augmenteds_span)
        x = [item for t in self.exemplars_x for item in t]
        y = [item for t in self.exemplars_y for item in t]
        mask = [item for t in self.exemplars_mask for item in t]
        span = [item for t in self.exemplars_span for item in t]    
        dataset.extend(x, y, mask, span)
        return DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=False)
        





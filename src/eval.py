import os
import json
import tqdm

import torch
import numpy as np
import torch.nn.functional as F

import sys
import utils
from src.datasets.common import get_dataloader, get_dataloader_shuffle, maybe_dictionarize
from heads import get_classification_head
from src.modeling import ImageClassifier
from transformers import AutoTokenizer, Trainer, TrainingArguments
from src.datasets.registry import get_dataset
from sklearn.metrics import accuracy_score
from datasets import load_dataset


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}


def test_imdb(model, tokenizer, training_args, quick_test=False):
    dataset_imdb = load_dataset('imdb')
    
    test_dataset = dataset_imdb['test']

    if quick_test:
        bs = getattr(training_args, "per_device_eval_batch_size", 1) or 1
        take_n = min(bs, len(test_dataset))
        test_dataset = test_dataset.select(range(take_n))
        
    test_dataset = test_dataset.map(lambda x: tokenizer(x['text'], truncation=True, padding='max_length'), batched=True)
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    imdb_eval_result = trainer.evaluate()
    print(f"(finetuned model) imdb:", imdb_eval_result['eval_accuracy'])
    return imdb_eval_result['eval_accuracy']


def test_sst2(model, tokenizer, training_args, quick_test=False):
    # SST2
    dataset_sst2 = load_dataset('sst2')

    test_dataset = dataset_sst2['validation']

    if quick_test:
        bs = getattr(training_args, "per_device_eval_batch_size", 1) or 1
        take_n = min(bs, len(test_dataset))
        test_dataset = test_dataset.select(range(take_n))

    test_dataset = test_dataset.map(lambda x: tokenizer(x['sentence'], truncation=True, padding='max_length'), batched=True)
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    imdb_eval_result = trainer.evaluate()
    print(f"(finetuned model) sst2:", imdb_eval_result['eval_accuracy'])
    return imdb_eval_result['eval_accuracy']


def test_yelp(model, tokenizer, training_args, quick_test=False):
    # Yelp Review Polarity
    dataset_yelp = load_dataset('yelp_polarity')
    
    test_dataset = dataset_yelp['test']

    if quick_test:
        bs = getattr(training_args, "per_device_eval_batch_size", 1) or 1
        take_n = min(bs, len(test_dataset))
        test_dataset = test_dataset.select(range(take_n))

    test_dataset = test_dataset.map(lambda x: tokenizer(x['text'], truncation=True, padding='max_length'), batched=True)
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    imdb_eval_result = trainer.evaluate()
    print(f"(finetuned model) yelp:", imdb_eval_result['eval_accuracy'])
    return imdb_eval_result['eval_accuracy']


def test_ag_news(model, tokenizer, training_args, quick_test=False):
    # AG News
    dataset_new = load_dataset("ag_news")
    dataset_new = dataset_new.filter(lambda example: example['label'] in [0, 1])

    if quick_test:
        bs = getattr(training_args, "per_device_eval_batch_size", 1) or 1
        take_n = min(bs, len(dataset_new['test']))
        dataset_new['test'] = dataset_new['test'].select(range(take_n))

    test_dataset = dataset_new['test']
    test_dataset = test_dataset.map(lambda x: tokenizer(x['text'], truncation=True, padding='max_length'), batched=True)
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    imdb_eval_result = trainer.evaluate()
    print(f"(finetuned model) ag_news:", imdb_eval_result['eval_accuracy'])
    return imdb_eval_result['eval_accuracy']


def test_rotten_tomatoes(model, tokenizer, training_args, quick_test=False):
    # Rotten Tomatoes
    dataset_tomatoes = load_dataset('rotten_tomatoes')

    test_dataset = dataset_tomatoes['test']

    if quick_test:
        bs = getattr(training_args, "per_device_eval_batch_size", 1) or 1
        take_n = min(bs, len(test_dataset))
        test_dataset = test_dataset.select(range(take_n))

    test_dataset = test_dataset.map(lambda x: tokenizer(x['text'], truncation=True, padding='max_length'), batched=True)
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    imdb_eval_result = trainer.evaluate()
    print(f"(finetuned model) rotten_tomatoes:", imdb_eval_result['eval_accuracy'])
    return imdb_eval_result['eval_accuracy']


def test_cola(model, tokenizer, training_args, quick_test=False):
    # CoLA
    dataset_cola = load_dataset('glue', 'cola')

    test_dataset = dataset_cola['validation']

    if quick_test:
        bs = getattr(training_args, "per_device_eval_batch_size", 1) or 1
        take_n = min(bs, len(test_dataset))
        test_dataset = test_dataset.select(range(take_n))

    test_dataset = test_dataset.map(lambda x: tokenizer(x['sentence'], truncation=True, padding='max_length'), batched=True)
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    imdb_eval_result = trainer.evaluate()
    print(f"(finetuned model) cola:", imdb_eval_result['eval_accuracy'])
    return imdb_eval_result['eval_accuracy']


def test_sms(model, tokenizer, training_args, quick_test=False):
    # SMS Spam Collection
    dataset_sms = load_dataset('sms_spam')
    dataset_sms = dataset_sms['train'].train_test_split(test_size=0.1, seed=42)

    test_dataset = dataset_sms['test']

    if quick_test:
        bs = getattr(training_args, "per_device_eval_batch_size", 1) or 1
        take_n = min(bs, len(test_dataset))
        test_dataset = test_dataset.select(range(take_n))

    test_dataset = test_dataset.map(lambda x: tokenizer(x['sms'], truncation=True, padding='max_length'), batched=True)
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    eval_result = trainer.evaluate()
    print(f"(finetuned model) sms:", eval_result.get('eval_accuracy', None))
    return eval_result.get('eval_accuracy', None)
    

def eval_single_dataset(image_encoder, dataset_name, args, use_train=False, no_print=False,
        use_shuffle_test=False, use_val=False, val_test=False, quick_iter=False, constrain_batch_size=None):
    if dataset_name in ["imdb", "sst2", "yelp", "ag_news", "rotten_tomatoes", "cola", "sms"]:
        return eval_single_dataset_NLP(image_encoder, dataset_name)
    else:
        return eval_single_dataset_CV(image_encoder, dataset_name, args, use_train, no_print,
        use_shuffle_test, use_val, val_test, quick_iter, constrain_batch_size)


def eval_single_dataset_NLP(image_encoder, dataset_name, quick_test=False):
    training_args = TrainingArguments(
        output_dir='./results',          
        num_train_epochs=1,              
        per_device_train_batch_size=16, 
        per_device_eval_batch_size=1 if quick_test else 128, # 64,  
        warmup_steps=500,                
        weight_decay=0.01,               
        logging_dir='./logs',
        report_to="none",
    )
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    test_funcs = {
        "ag_news": test_ag_news,
        "rotten_tomatoes": test_rotten_tomatoes,
        "cola": test_cola,
        "sms": test_sms,
    }
    if dataset_name in test_funcs:
        top1 = test_funcs[dataset_name](image_encoder, tokenizer, training_args, quick_test=quick_test)
        metrics = {'top1': top1}
        metrics['loss'] = None
        return metrics
    

def eval_single_dataset_CV(image_encoder, dataset_name, args, use_train=False, no_print=False,
        use_shuffle_test=False, use_val=False, val_test=False, quick_iter=False, constrain_batch_size=None):
    classification_head = get_classification_head(args, dataset_name)
    model = ImageClassifier(image_encoder, classification_head)

    model.eval()

    dataset = get_dataset(
        dataset_name,
        model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=14, # TODO
        use_val=use_val
    )
    
    if use_shuffle_test:
        dataloader = get_dataloader_shuffle(dataset)
    else:
        dataloader = get_dataloader(
            dataset, is_train=use_train, args=args, image_encoder=None, use_val=use_val, val_test=val_test)
    
    device = args.device

    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        total_loss = 0.
        for i, data in enumerate(tqdm.tqdm(dataloader)):
            data = maybe_dictionarize(data)
            x = data['images'].to(device)
            y = data['labels'].to(device) # [batch_size]
            if constrain_batch_size is not None:
                x = x[:constrain_batch_size]
                y = y[:constrain_batch_size]

            logits = utils.get_logits(x, model) # [batch_size, num_classes]
            loss = F.cross_entropy(logits, y, reduction='sum')

            pred = logits.argmax(dim=1, keepdim=True).to(device)

            correct += pred.eq(y.view_as(pred)).sum().item()
            total_loss += loss.item()
            
            n += y.size(0)
            
            if quick_iter:
                break

        top1 = correct / n
        loss = total_loss / n

    metrics = {'top1': top1}
    metrics['loss'] = loss
    if not no_print:
        print(f'Done evaluating on {dataset_name}. Accuracy: {round(100*top1,1)}')
    
    return metrics


def inference_single_dataset(image_encoder, dataset_name, args, use_train=False, no_print=False,
        use_shuffle_test=False, use_val=False, val_test=False, quick_iter=False, constrain_batch_size=None):
    classification_head = get_classification_head(args, dataset_name)
    model = ImageClassifier(image_encoder, classification_head)

    model.eval()

    dataset = get_dataset(
        dataset_name,
        model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=14, # TODO
        use_val=use_val
    )
    
    if use_shuffle_test:
        dataloader = get_dataloader_shuffle(dataset)
    else:
        dataloader = get_dataloader(
            dataset, is_train=use_train, args=args, image_encoder=None, use_val=use_val, val_test=val_test)
    
    device = args.device
    component2grad = {}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    for i, data in enumerate(tqdm.tqdm(dataloader)):
        data = maybe_dictionarize(data)
        x = data['images'].to(device)
        y = data['labels'].to(device) # [batch_size]
        if constrain_batch_size is not None:
            x = x[:constrain_batch_size]
            y = y[:constrain_batch_size]

        optimizer.zero_grad()
        logits = utils.get_logits(x, model) # [batch_size, num_classes]
        loss = F.cross_entropy(logits, y, reduction='sum')
        loss.backward()
        # Extract the gradients of each layer of the model.
        for name, param in model.named_parameters():
            if param.grad is not None:
                component2grad[name] = param.grad.norm().item()
                print(f"Layer: {name}, Gradient Norm: {param.grad.norm().item()}")
        
        if quick_iter:
            break
    
    return component2grad



def cal_single_dataset_feature(image_encoder, dataset_name, args, constrain_batch_size=None):
    classification_head = get_classification_head(args, dataset_name)
    model = ImageClassifier(image_encoder, classification_head)

    model.eval()

    dataset = get_dataset(
        dataset_name,
        image_encoder.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=14, # TODO
    )
  
    dataloader = get_dataloader(
        dataset, is_train=False, args=args, image_encoder=None)
    
    device = args.device

    with torch.no_grad():
        for i, data in enumerate(tqdm.tqdm(dataloader)):
            data = maybe_dictionarize(data)
            x = data['images'].to(device)
            y = data['labels'].to(device) # [batch_size]
            if constrain_batch_size is not None:
                x = x[:constrain_batch_size]
                y = y[:constrain_batch_size]
            break            
    return x, y


def eval_single_dataset_head(image_encoder, head, dataset_name, args):
    model = ImageClassifier(image_encoder, head)

    model.eval()

    dataset = get_dataset(dataset_name, model.val_preprocess, location=args.data_location,  batch_size=args.batch_size)
    dataloader = get_dataloader(dataset, is_train=False, args=args, image_encoder=None)
    device = args.device

    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        for i, data in enumerate(tqdm.tqdm(dataloader)):
            data = maybe_dictionarize(data)
            x = data['images'].to(device)
            y = data['labels'].to(device)

            logits = utils.get_logits(x, model)

            pred = logits.argmax(dim=1, keepdim=True).to(device)

            correct += pred.eq(y.view_as(pred)).sum().item()

            n += y.size(0)

        top1 = correct / n

    metrics = {'top1': top1}
    print(f'Done evaluating on {dataset_name}. Accuracy: {100 * top1:.2f}%')

    return metrics

def eval_single_dataset_preprocess_head(image_encoder, head, dataset_name, args):
    model = ImageClassifier(image_encoder, head)

    model.eval()

    dataset = get_dataset(dataset_name, model.val_preprocess, location=args.data_location,  batch_size=args.batch_size)
    dataloader = get_dataloader(dataset, is_train=False, args=args, image_encoder=None)
    device = args.device

    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        for i, data in enumerate(tqdm.tqdm(dataloader)):
            data = maybe_dictionarize(data)
            x = data['images'].to(device)
            y = data['labels'].to(device)

            logits = utils.get_logits(x, model)

            pred = logits.argmax(dim=1, keepdim=True).to(device)

            correct += pred.eq(y.view_as(pred)).sum().item()

            n += y.size(0)

        top1 = correct / n

    metrics = {'top1': top1}
    print(f'Done evaluating on {dataset_name}. Accuracy: {100 * top1:.2f}%')

    return metrics

def evaluate(image_encoder, args):
    if args.eval_datasets is None:
        return
    info = vars(args)
    for i, dataset_name in enumerate(args.eval_datasets):
        print('Evaluating on', dataset_name)

        results = eval_single_dataset_CV(image_encoder, dataset_name, args)

        if 'top1' in results:
            print(f"{dataset_name} Top-1 accuracy: {results['top1']:.4f}")
        for key, val in results.items():
            if 'worst' in key or 'f1' in key.lower() or 'pm0' in key:
                print(f"{dataset_name} {key}: {val:.4f}")
            info[dataset_name + ':' + key] = val

    if args.results_db is not None:
        dirname = os.path.dirname(args.results_db)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(args.results_db, 'a+') as f:
            f.write(json.dumps(info) + '\n')
        print(f'Results saved to {args.results_db}.')
    else:
        print('Results not saved (to do so, use --results_db to specify a path).')

    return info
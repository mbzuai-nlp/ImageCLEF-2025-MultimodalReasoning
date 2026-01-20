import json
import sys
import argparse
from collections import defaultdict
from inspect import signature    

ID_FIELD_NAME = 'id'
ANSWER_FIELD_NAME = 'answer_key'
    
def call_validations(functions, *args, **kwargs):
    results = []
    for func in functions:
        sig = signature(func)
        func_args = {}
        
        for param in sig.parameters:
            if param in kwargs:
                func_args[param] = kwargs[param]
            elif args:
                func_args[param] = args[0]
                args = args[1:]
        results.append(func(**func_args))
    return results

################### Validation functions ###################
def are_keys_correct(pred):
    for answer in pred.values():
        if not(answer == 'A' or answer == 'B' or answer == 'C' or answer == 'D' or answer == 'E'):
            raise Exception('Unsupported answer_key: {}. Cannot score predictions.'.format(answer))
        
def dataset_size_match(pred, gold):
    if not len(pred) == len(gold):
        raise Exception('Invalid pred file. Prediction size does not match test or duplicate ids were found.')
    
def ids_match(pred, gold):
    for key in pred.keys():
        try:
            gold[key]
        except:
            raise Exception('Invalid submission. Test data does not contain id: {}'.format(key))
################### Validation functions ###################
def read_data(path, lang=None):
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        jsn_data = json.loads(f.read())
    for el in jsn_data:
        if ID_FIELD_NAME not in el.keys() or ANSWER_FIELD_NAME not in el.keys():
            raise Exception('Prediction file format is invalid. Each object must contain the following fields: [id, answer_key]')
        if lang:
            if lang == el['language']:
                data[el[ID_FIELD_NAME]] = el[ANSWER_FIELD_NAME]
        else:
            data[el[ID_FIELD_NAME]] = el[ANSWER_FIELD_NAME]
    return data
  
def load_pred_gold_per_lang(pred_path, gold_path, lang):
    #Load data
    pred = read_data(pred_path, lang)
    gold = read_data(gold_path, lang)
    
    # Validate data
    validations = [dataset_size_match, are_keys_correct, ids_match]
    val_results = call_validations(validations, pred=pred, gold=gold)

    if False in val_results:
        return None, None
    
    return pred, gold
  
def load_pred_gold(pred_path, gold_path):
    #Load data
    pred = read_data(pred_path)
    gold = read_data(gold_path)
    
    # Validate data
    validations = [dataset_size_match, are_keys_correct, ids_match]
    val_results = call_validations(validations, pred=pred, gold=gold)

    if False in val_results:
        return None, None
    
    return pred, gold

def evaluate(pred_path, gold_path, language):
    pred, gold = load_pred_gold_per_lang(pred_path, gold_path, language)
    correct = 0.0
    total = len(gold)
    for qstn_id in gold.keys():
        if pred[qstn_id] == gold[qstn_id]:
            correct += 1
    return correct / total
    
'''
    Example script usage:
        python evaluate.py --pred_file="./pred.json" --gold_file="./gold.json" --print_score="True"
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_file', type=str, help='path to submission file')
    parser.add_argument('--gold_file', type=str, help='path to gold truth file')
    parser.add_argument('--print_score', type=bool, default=False, help='path to gold truth file')
    parser.add_argument('--language', type=str, default=False, help='language to evaluate')
    args = parser.parse_args()
    
    evaluate(args.pred_file, args.gold_file, args.language)
    if args.print_score:
        print(evaluate(args.pred_file, args.gold_file, args.language))
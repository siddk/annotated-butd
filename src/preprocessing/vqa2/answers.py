"""
answers.py

Core script for pre-processing VQA-2 Answer Data --> Truncates Answer Labels based on number of occurrences, and
computes soft-labels.

Reference: https://github.com/hengyuan-hu/bottom-up-attention-vqa/blob/master/tools/compute_softscore.py
"""
import json
import os
import pickle
import re

# CONSTANTS
CONTRACTIONS = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't",
                "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't",
                "dont": "don't", "hadnt": "hadn't", "hadnt've": "hadn't've", "hadn'tve": "hadn't've",
                "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've",
                "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've",
                "I'dve": "I'd've", "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've",
                "it'dve": "it'd've", "itll": "it'll", "let's": "let's", "maam": "ma'am", "mightnt": "mightn't",
                "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've", "mustnt": "mustn't",
                "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock",
                "oughtnt": "oughtn't", "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at",
                "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've", "she's": "she's",
                "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've",
                "shouldn'tve": "shouldn't've", "somebody'd": "somebodyd", "somebodyd've": "somebody'd've",
                "somebody'dve": "somebody'd've", "somebodyll": "somebody'll", "somebodys": "somebody's",
                "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've",
                "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd",
                "somethingd've": "something'd've", "something'dve": "something'd've",
                "somethingll": "something'll", "thats": "that's", "thered": "there'd", "thered've": "there'd've",
                "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd",
                "theyd've": "they'd've", "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're",
                "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", "wed've": "we'd've", "we'dve": "we'd've",
                "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're", "whats": "what's",
                "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's",
                "whereve": "where've", "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've",
                "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", "whyre": "why're",
                "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't",
                "wouldnt've": "wouldn't've", "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll",
                "y'allll": "y'all'll", "yall'd've": "y'all'd've", "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've",
                "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", "youll": "you'll", "youre": "you're",
                "youve": "you've"}

MANUAL_MAP = {'none': '0', 'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5', 'six': '6',
              'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10'}
ARTICLES = ['a', 'an', 'the']
PUNCT = [';', r"/", '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-', '>', '<', '@', '`', ',', '?', '!']

PERIOD_STRIP = re.compile("(?!<=\d)(\.)(?!\d)")
COMMA_STRIP = re.compile("(\d)(\,)(\d)")


def process_punctuation(inText):
    outText = inText
    for p in PUNCT:
        if (p + ' ' in inText or ' ' + p in inText) or (re.search(COMMA_STRIP, inText) != None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = PERIOD_STRIP.sub("", outText, re.UNICODE)
    return outText


def process_digit_article(inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = MANUAL_MAP.setdefault(word, word)
        if word not in ARTICLES:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in CONTRACTIONS:
            outText[wordId] = CONTRACTIONS[word]
    outText = ' '.join(outText)
    return outText


def preprocess_answer(answer):
    answer = process_digit_article(process_punctuation(answer))
    answer = answer.replace(',', '')
    return answer


def get_score(occurrences):
    if occurrences == 0:
        return 0
    elif occurrences == 1:
        return 0.3
    elif occurrences == 2:
        return 0.6
    elif occurrences == 3:
        return 0.9
    else:
        return 1


def filter_answers(answers, min_occurrences=9):
    """ Count the number of occurrences of each answer in the train/validation set """
    occurrence = {}
    for entry in answers:
        ground_truth = preprocess_answer(entry['multiple_choice_answer'])

        # Add Answer, Question ID to Occurrences
        if ground_truth not in occurrence:
            occurrence[ground_truth] = set()
        occurrence[ground_truth].add(entry['question_id'])

    # Iterate through Occurrences and Filter any Answers w/ Fewer than `min_occurrences` occurrences
    for ans in list(occurrence.keys()):
        if len(occurrence[ans]) < min_occurrences:
            occurrence.pop(ans)

    print('\t[*] Filtered Rare Answers --> %d Answers appear >= %d Times...' % (len(occurrence), min_occurrences))
    return occurrence


def create_labels(occurrence):
    """ Create Mappings between Answers and Label ID """
    ans2label, label2ans = {}, []
    for answer in occurrence:
        ans2label[answer] = len(ans2label)
        label2ans.append(answer)
    return ans2label, label2ans


def compute_target(answers, ans2label):
    """ Associate each example in the dataset with a soft distribution over answers """
    target = []
    for entry in answers:
        ans, ans_count = entry['answers'], {}
        for a in ans:
            a_ = a['answer']
            ans_count[a_] = ans_count.get(a_, 0) + 1

        labels, scores = [], []
        for a in ans_count:
            if a not in ans2label:
                continue
            labels.append(ans2label[a])
            scores.append(get_score(ans_count[a]))

        target.append({
            'question_id': entry['question_id'],
            'image_id': entry['image_id'],
            'labels': labels,
            'scores': scores
        })

    return target


def vqa2_create_soft_answers(vqa2_q='data/VQA2-Questions', cache='data/VQA2-Cache', min_occurrences=9):
    """Create set of possible answers for VQA2 based on Occurrences & Compute Soft-Labels"""
    train_ans = os.path.join(vqa2_q, 'v2_mscoco_train2014_annotations.json')
    val_ans = os.path.join(vqa2_q, 'v2_mscoco_val2014_annotations.json')

    # Create File Paths and Load from Disk (if exists)
    a2lfile, tfile = os.path.join(cache, 'ans2label.pkl'), os.path.join(cache, 'target.pkl')
    if os.path.exists(a2lfile) and os.path.exists(tfile):
        with open(a2lfile, 'rb') as f:
            ans2label, label2ans = pickle.load(f)
        with open(tfile, 'rb') as f:
            train_target, val_target = pickle.load(f)

        return ans2label, label2ans, train_target, val_target

    # Load from JSON
    print('\t[*] Reading Training and Validation Answers for Pre-processing...')
    with open(train_ans, 'r') as f:
        train_answers = json.load(f)['annotations']

    with open(val_ans, 'r') as f:
        val_answers = json.load(f)['annotations']

    # Aggregate and Filter Answers based on Number of Occurrences
    answers = train_answers + val_answers
    occurrence = filter_answers(answers, min_occurrences=min_occurrences)

    # Create Answer Labels
    ans2label, label2ans = create_labels(occurrence)

    # Compute Per-Example Target Distribution over Answers (Soft Answers)
    train_target = compute_target(train_answers, ans2label)
    val_target = compute_target(val_answers, ans2label)

    # Dump Ans2Label and Targets to File
    with open(a2lfile, 'wb') as f:
        pickle.dump((ans2label, label2ans), f)

    with open(tfile, 'wb') as f:
        pickle.dump((train_target, val_target), f)

    # Return Mapping and Targets
    return ans2label, label2ans, train_target, val_target

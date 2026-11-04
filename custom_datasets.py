import random
import datasets
import pandas as pd
SEPARATOR = '<<<SEP>>>'


DATASETS = ['writing', 'english', 'german', 'pubmed','kaggle']


def load_pubmed(cache_dir):
    data = datasets.load_dataset('pubmed_qa', 'pqa_labeled', split='train', cache_dir=cache_dir)
    
    # combine question and long_answer
    data = [f'Question: {q} Answer:{SEPARATOR}{a}' for q, a in zip(data['question'], data['long_answer'])]

    return data


def process_prompt(prompt):
    return prompt.replace('[ WP ]', '').replace('[ OT ]', '')


def process_spaces(story):
    return story.replace(
        ' ,', ',').replace(
        ' .', '.').replace(
        ' ?', '?').replace(
        ' !', '!').replace(
        ' ;', ';').replace(
        ' \'', '\'').replace(
        ' ’ ', '\'').replace(
        ' :', ':').replace(
        '<newline>', '\n').replace(
        '`` ', '"').replace(
        ' \'\'', '"').replace(
        '\'\'', '"').replace(
        '.. ', '... ').replace(
        ' )', ')').replace(
        '( ', '(').replace(
        ' n\'t', 'n\'t').replace(
        ' i ', ' I ').replace(
        ' i\'', ' I\'').replace(
        '\\\'', '\'').replace(
        '\n ', '\n').strip()


def load_writing(cache_dir=None):
    writing_path = 'data/writingPrompts'
    
    with open(f'{writing_path}/valid.wp_source', 'r', encoding='utf-8') as f:
        prompts = f.readlines()
    with open(f'{writing_path}/valid.wp_target', 'r', encoding='utf-8') as f:
        stories = f.readlines()
    
    prompts = [process_prompt(prompt) for prompt in prompts]
    joined = [process_spaces(prompt + " " + story) for prompt, story in zip(prompts, stories)]
    filtered = [story for story in joined if 'nsfw' not in story and 'NSFW' not in story]

    random.seed(0)
    random.shuffle(filtered)

    return filtered


def load_language(language, cache_dir):
    # load either the english or german portion of the wmt16 dataset
    assert language in ['en', 'de']
    d = datasets.load_dataset('wmt16', 'de-en', split='train', cache_dir=cache_dir)
    docs = d['translation']
    desired_language_docs = [d[language] for d in docs]
    lens = [len(d.split()) for d in desired_language_docs]
    sub = [d for d, l in zip(desired_language_docs, lens) if l > 100 and l < 150]
    return sub


def load_german(cache_dir):
    return load_language('de', cache_dir)


def load_english(cache_dir):
    return load_language('en', cache_dir)

def load_kaggle(cache_dir=None, path='data/kaggle/train_essays.csv', max_words=250, prompt_id=None):
    """Human-written student essays from the Kaggle 'LLM - Detect AI Generated Text' training file.
    Keeps generated == 0 only (the 3 LLM essays are too few to use), cuts each essay to its first
    `max_words` words so it fits the 512-token limit of the T5 mask-filling model, and optionally
    keeps a single prompt (0 = Car-free cities, 1 = Does the electoral college work?)."""
    df = pd.read_csv(path)
    df = df[df['generated'] == 0]
    if prompt_id is not None:
        df = df[df['prompt_id'] == prompt_id]
    essays = [' '.join(t.split()[:max_words]) for t in df['text']]
    random.seed(0)
    random.shuffle(essays)
    return essays
 
def load(name, cache_dir, **kwargs):
    if name in DATASETS:
        load_fn = globals()[f'load_{name}']
        return load_fn(cache_dir=cache_dir, **kwargs)
    else:
        raise ValueError(f'Unknown dataset {name}')
import os
import pickle
import json
import argparse
from six.moves import cPickle
from collections import defaultdict
from tqdm import tqdm

def precook(s, n=4, out=False):
  """
  Takes a string as input and returns an object that can be given to
  either cook_refs or cook_test. This is optional: cook_refs and cook_test
  can take string arguments as well.
  :param s: string : sentence to be converted into ngrams
  :param n: int    : number of ngrams for which representation is calculated
  :return: term frequency vector for occuring ngrams
  """
  words = s.split()
  counts = defaultdict(int)
  for k in xrange(1,n+1):
    for i in xrange(len(words)-k+1):
      ngram = tuple(words[i:i+k])
      counts[ngram] += 1
  return counts

def cook_refs(refs, n=4): ## lhuang: oracle will call with "average"
    '''Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.
    :param refs: list of string : reference sentences for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (list of dict)
    '''
    return [precook(ref, n) for ref in refs]

def create_crefs(refs):
  crefs = []
  for ref in tqdm(refs):
    # ref is a list of 5 captions
    crefs.append(cook_refs(ref))
  return crefs

def compute_doc_freq(crefs):
  '''
  Compute term frequency for reference data.
  This will be used to compute idf (inverse document frequency later)
  The term frequency is stored in the object
  :return: None
  '''
  document_frequency = defaultdict(float)
  for refs in tqdm(crefs):
    # refs, k ref captions of one image
    for ngram in set([ngram for ref in refs for (ngram,count) in ref.iteritems()]):
      document_frequency[ngram] += 1
      # maxcounts[ngram] = max(maxcounts.get(ngram,0), count)
  return document_frequency

# def build_dict(imgs, wtoi, params):
#   wtoi['<eos>'] = 0
#
#   count_imgs = 0
#
#   refs_words = []
#   refs_idxs = []
#   for img in imgs:
#     if (params['split'] == img['split']) or \
#       (params['split'] == 'train' and img['split'] == 'restval') or \
#       (params['split'] == 'all'):
#       #(params['split'] == 'val' and img['split'] == 'restval') or \
#       ref_words = []
#       ref_idxs = []
#       for sent in img['sentences']:
#         tmp_tokens = sent['tokens'] + ['<eos>']
#         tmp_tokens = [_ if _ in wtoi else 'UNK' for _ in tmp_tokens]
#         ref_words.append(' '.join(tmp_tokens))
#         ref_idxs.append(' '.join([str(wtoi[_]) for _ in tmp_tokens]))
#       refs_words.append(ref_words)
#       refs_idxs.append(ref_idxs)
#       count_imgs += 1
#   print('total imgs:', count_imgs)
#
#   ngram_words = compute_doc_freq(create_crefs(refs_words))
#   ngram_idxs = compute_doc_freq(create_crefs(refs_idxs))
#   return ngram_words, ngram_idxs, count_imgs

def build_dict(data):

    refs_idxs = [data]

    ngram_idxs = compute_doc_freq(create_crefs(refs_idxs))

    cPickle.dump({'document_frequency': ngram_idxs, 'ref_len': len(data)+1}, open('./cider' + '-idxs.p', 'w'),
                 protocol=cPickle.HIGHEST_PROTOCOL)
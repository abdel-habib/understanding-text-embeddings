import pandas as pd
import torch 
import numpy as np
from typing import Union

def show_tokenization(inputs, tokenizer):
    ''' Show tokenization of a returned input. '''

    return pd.DataFrame(
        [(id, tokenizer.decode(id)) for id in inputs["input_ids"][0]],
        columns=["id", "token"],
    )

def search_through_vocab_dict_using_id(tokenizer, id):
    ''' Search for a token in the vocabulary of the tokenizer by its id. 
    
    If the id is not found in the vocabulary, a message is returned.
    
    Args:
    
        tokenizer: The tokenizer used to tokenize the text.
        id: The id to search for in the vocabulary dictionary.

    Returns:
        The token of the id in the vocabulary if found, or a message if the id is not found.
        '''
    vocab_dict = tokenizer.get_vocab()

    return list(vocab_dict.keys())[list(vocab_dict.values()).index(id)] if id in list(vocab_dict.values()) else 'Id not found in the vocabulary.'


def search_through_vocab_dict_using_token(tokenizer, token):
    ''' Search for an id in the vocabulary of the tokenizer by its token. 
    
    If the token is not found in the vocabulary, a message is returned.

    Args:
        tokenizer: The tokenizer used to tokenize the text.
        token: The token to search for in the vocabulary.

    Returns:
        The id of the token in the vocabulary if found, or a message if the token is not found.
    '''
    vocab_dict = tokenizer.get_vocab()

    return vocab_dict[token] if token in list(vocab_dict.keys()) else 'Token not found in the vocabulary.'


def embedding_pooling(embeddings, attention_mask = None, pooling_strategy = 'cls'):
    ''' Pool the embeddings outputed by the model using the specified pooling strategy.

    The embeddings has a shape of (batch_size, sequence_length, hidden_size). 
    The attention mask is used in some pooling strategies to ignore the padding tokens. Those text tokens are extracted from the inputs using the 1's in the attention mask, where any 
    padded values will be given a  0 attention value.
    
    Args:
        embeddings: The embeddings to pool.
        attention_mask: The attention mask to use to ignore the padding tokens. It is used in some pooling strategies.
        pooling_strategy: The pooling strategy to use. It can be either 'mean' or 'max'.

    Returns:
        The pooled embeddings.
    '''
    def _validate_attention_mask(attention_mask):
        if attention_mask is None:
            raise ValueError('`attention_mask` must be provided for some pooling strategy to extract the attention mask and handle the padded tokens with zero attention.')

    if pooling_strategy == 'cls':
        return embeddings[:, 0, :] # take the first token of the sequence: [CLS] token, shape: (batch_size, hidden_size)

    elif pooling_strategy == 'eos':
        # validate if the inputs are given to ensure the attention mask is available
        _validate_attention_mask(attention_mask)
        
        # using the attention mask to ignore the padding tokens and take the last token of the sequence: [EOS] token
        eos_token_indices = attention_mask.sum(dim=-1) - 1 # the -1 is as the indices start from 0, and if the last token is [EOS], then the index is the length of the sequence - 1
        return embeddings[torch.arange(embeddings.shape[0], device=embeddings.device), eos_token_indices] # shape: (batch_size, hidden_size)
        
    elif pooling_strategy == 'max':
        # validate if the inputs are given to ensure the attention mask is available
        _validate_attention_mask(attention_mask)

        # the unsqueeze is to make the attention mask have the same shape as the embeddings
        # the multiplication is to zero out the embeddings of the padding tokens
        max, _ = torch.max(embeddings * attention_mask.unsqueeze(-1), dim=1) 
        return max
    
    elif pooling_strategy == 'mean':
        # validate if the inputs are given to ensure the attention mask is available
        _validate_attention_mask(attention_mask)
        
        # the unsqueeze is to make the attention mask have the same shape as the embeddings
        # the multiplication is to zero out the embeddings of the padding tokens
        sum = torch.sum(embeddings * attention_mask.unsqueeze(-1), dim=1)
        mean = sum / attention_mask.sum(dim=1).unsqueeze(1)
        return mean
    
    else:
        # `cls_avg`, `cls_max`, `last`, `avg`, `mean`, `max`, `all`, int]'
        raise NotImplementedError(
            'please specify pooling_strategy from [`cls`, `eos`, `max`, `mean`]')

def cosine_distance(v1: np.ndarray, v2: np.ndarray) -> Union[float, np.ndarray]:
    '''
    Compute the cosine distance between two vectors.

    Parameters
    ----------
    v1 : np.ndarray
        First vector.
    v2 : np.ndarray
        Second vector.

    Returns
    -------
    float
        Cosine distance between `v1` and `v2`.
    '''
    vecs = (v1, v2) if len(v1.shape) >= len(v2.shape) else (v2, v1)
    return 1 - np.dot(*vecs) / (
            np.linalg.norm(v1, axis=len(v1.shape)-1) *
            np.linalg.norm(v2, axis=len(v2.shape)-1)
    )

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> Union[float, np.ndarray]:
    '''
    Compute the cosine similarity between two vectors usin the cosine distance function.

    Parameters
    ----------
    v1 : np.ndarray
        First vector.
    v2 : np.ndarray
        Second vector.

    Returns
    -------
    float
        Cosine similarity between `v1` and `v2`.
    '''
    return 1 - cosine_distance(v1, v2)

def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute the Euclidean distance between two vectors.

    Parameters
    ----------
    v1 : np.ndarray
        First vector.
    v2 : np.ndarray
        Second vector.

    Returns
    -------
    float
        Euclidean distance between `v1` and `v2`.
    """
    dist = v1 - v2
    return np.linalg.norm(dist, axis=len(dist.shape)-1)
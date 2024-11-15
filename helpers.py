import pandas as pd

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
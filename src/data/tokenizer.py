from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import TemplateProcessing

from typing import List
import json

class LMTokenizer:
    """
    Byte-level BPE tokenizer wrapper.
    """
    
    def __init__(self):
        """
        Initialize an untrained byte-level BPE tokenizer instance.
        """
        self.tokenizer = Tokenizer(BPE(byte_fallback=True, unk_token='<UNK>'))
        self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        self.tokenizer.decoder = ByteLevelDecoder()
        self.special_tokens = {}

    def train(self, corpus_sents, vocab_size=10_000, special_tokens=("<PAD>", "<EOS>", "<UNK>")):
        """
        Train the tokenizer on an iterable corpus and configure special tokens.
        Args:
            corpus_sents: Iterable of training sentences/text samples.
            vocab_size: Maximum vocabulary size to learn.
            special_tokens: Special tokens to register during training.
        """
        trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=list(special_tokens), show_progress=True)
        self.tokenizer.train_from_iterator(corpus_sents, trainer=trainer)

        vocab = self.tokenizer.get_vocab()
        self.special_tokens = {token: int(vocab[token]) for token in special_tokens if token in vocab}

        if '<EOS>' in self.special_tokens:
            self.tokenizer.post_processor = TemplateProcessing(
                single="$A <EOS>",
                pair="$A <EOS> $B:1 <EOS>:1",
                special_tokens=[("<EOS>", self.special_tokens["<EOS>"])]
            )

    @property
    def vocabulary(self):
        """
        Set[int]: Token ID vocabulary as a set of integer IDs.
        """
        return set(self.tokenizer.get_vocab().values())
    
    @property
    def vocab_size(self):
        """
        int: Size of the learned vocabulary.
        """
        return int(self.tokenizer.get_vocab_size())
    
    def encode(self, text: str, add_eos: bool = False):
        """
        Encode text into token IDs.
        Args:
            text: Input text to tokenize.
            add_eos: If True, return tokenizer output as-is (including EOS if
                post-processor appends it). If False, strip a trailing EOS token
                when present.
        Returns:
            List[int]: Encoded token IDs.
        """
        if add_eos:
            return self.tokenizer.encode(text).ids
        ids = self.tokenizer.encode(text).ids
        if "<EOS>" in self.special_tokens and ids and ids[-1] == self.special_tokens["<EOS>"]:
            return ids[:-1]
        return ids

    def decode(self, ids: List[int]):
        """
        Decode token IDs back into text.
        Args:
            ids: Sequence of token IDs.
        Returns:
            str: Decoded text.
        """
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def to_state(self):
        """
        Save tokenizer state into object.
        """
        return {
            'tokenizer': self.tokenizer.to_str(),
            'special_tokens': self.special_tokens,
        }
    
    def to_json(self, path):
        """
        Save tokenizer state into JSON file.
        """
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_state(), f, indent=4)
    
    @classmethod
    def from_state(cls, state):
        """
        Load tokenizer state from object.
        """
        tok = cls()
        tok.tokenizer = Tokenizer.from_str(state['tokenizer'])
        tok.special_tokens = {k: int(v) for k, v in state.get('special_tokens', {}).items()}
        return tok
    
    @classmethod
    def from_json(cls, path):
        """
        Load tokenizer state from JSON file.
        """
        tok = cls()
        with open(path, 'r', encoding='utf-8') as f:
            state = json.load(f)
        tok.tokenizer = Tokenizer.from_str(state['tokenizer'])
        tok.special_tokens = {k: int(v) for k, v in state.get('special_tokens', {}).items()}
        return tok
